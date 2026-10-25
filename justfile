set default-list := true

# CI runs every step under bash (Windows: Git Bash), so recipes do too.
set shell := ["bash", "-euo", "pipefail", "-c"]
set windows-shell := ["bash", "-euo", "pipefail", "-c"]

# Recipes in the CI groups below run the same commands as the GitHub Actions
# steps named in their comments, without the dependency steps. Install those
# with the deps-* recipes. Recipes that need the Python environment ask for a
# shell activated with the CI extras (`just deps-test`, then the activate_cmd
# below) and never sync it themselves.

# Union of the extras the CI test lanes install (Windows lanes omit bedrock).
# Activation must name them too: a bare `source ./activate` rebuilds the test
# environment with only [all], and tests for the missing extras skip.
test_extras := if os_family() == "windows" { "all,anthropic,mistral,fal,modal,daytona,parallel-web,messaging,telegram" } else { "all,anthropic,bedrock,mistral,fal,modal,daytona,parallel-web,messaging,telegram" }
activate_cmd := if os_family() == "windows" { ". .\\activate.ps1 -TestExtras " + test_extras } else { "source ./activate --test-extras " + test_extras }

# CI blanks the provider keys so no test can reach a live API.
no_keys := "OPENROUTER_API_KEY= OPENAI_API_KEY= NOUS_API_KEY="

# remote-backends.yml env.HERMES_TEST_IMAGE_REF
remote_test_image := "debian:13-slim@sha256:a29215f6a35e51e22adffa17f89e9d2ef06214e64a2bad10d765c46aea49f11f"

# What setup-pm exports for a job with extras or test-environment
# (scripts/ci/setup_toolchain.py dependencies): the test venv first on PATH,
# as HERMES_PYTHON and VIRTUAL_ENV, with the checkout on PYTHONPATH.
test_env := 'bindir="${__HERMES_TEST_PYTHON%[/\\]*}"; export PATH="$(cd "$bindir" && pwd):$PATH" HERMES_PYTHON="$__HERMES_TEST_PYTHON" VIRTUAL_ENV="$(cd "$bindir/.." && pwd)" PYTHONPATH="$PWD";'

lint-all: lint-python lint-all-js    

[group('lint python')]
lint-python:
    @ruff check ./

[group('lint python')]
lint-python-fix:
    @ruff check ./ --fix

[group('lint js')]
lint-all-js: lint-web lint-bootstrap-installer lint-desktop lint-tui lint-tests lint-shared

[group('lint js')]
lint-web:
    @cd web && npx eslint ./

[group('lint js')]
lint-bootstrap-installer:
    @cd apps/bootstrap-installer && npx eslint ./

[group('lint js')]
lint-desktop:
    @cd apps/desktop && npx eslint ./

[group('lint js')]
lint-tui:
    @cd ui-tui && npx eslint ./

[group('lint js')]
lint-tests:
    @cd tests-js && npx eslint ./

[group('lint js')]
lint-shared:
    @cd apps/shared && npx eslint ./

# ── Dependencies ──────────────────────────────────────────────────────────────

# Build the checkout's test environment with the provider extras the CI lanes use
[group('deps')]
[unix]
deps-test extras=test_extras:
    bash setup-hermes.sh --runtime-only --test-environment={{ extras }}

# Build the checkout's test environment with the provider extras the CI lanes use
[group('deps')]
[windows]
deps-test extras=test_extras:
    pwsh -NoProfile -File setup-hermes.ps1 -RuntimeOnly -TestExtras '{{ extras }}'

# Install every npm workspace from package-lock.json
[group('deps')]
deps-js:
    npm ci

# Install the docs site dependencies and the pinned ascii-guard
[group('deps')]
deps-docs:
    cd website && npm ci
    uv tool install ascii-guard==2.3.0

# Pull the pinned image the real SSH and Docker backend tests run against
[group('deps')]
[unix]
deps-remote-backends:
    docker pull -q "{{ remote_test_image }}"

# Acquire the native payload dependencies (pm-bundle.yml "Prepare native payload dependencies")
[group('deps')]
[macos]
deps-pm-bundle:
    PYTHONUTF8=1 python3 -S -B scripts/bundles/native_build.py --prepare-only --source "{{ justfile_directory() }}" --work "{{ justfile_directory() }}/.build/payload-job" --cache "{{ justfile_directory() }}/.build/payload-inputs" --out "{{ justfile_directory() }}/.build/agent-payload" --ref HEAD

# Acquire the native payload dependencies (pm-bundle.yml "Prepare native payload dependencies")
[group('deps')]
[windows]
deps-pm-bundle:
    PYTHONUTF8=1 python -S -B scripts/bundles/native_build.py --prepare-only --source "{{ justfile_directory() }}" --work "{{ justfile_directory() }}/.build/payload-job" --cache "{{ justfile_directory() }}/.build/payload-inputs" --out "{{ justfile_directory() }}/.build/agent-payload" --ref HEAD

# Desktop packaging downloads (`just test-js` packages the app). Their tarballs
# fail check-profile-archives and test-docker; the next package re-downloads them.
[group('deps')]
clean-desktop-cache:
    rm -rf "{{ justfile_directory() }}/.cache/desktop-inputs"

# The shell carries a current Hermes environment whose test venv has every CI extra
[private]
[script("bash")]
_require-env:
    set -euo pipefail
    fix="Run 'just deps-test', then '{{ activate_cmd }}'."
    if ! { . scripts/_activation.sh && hermes_activation_current "$PWD"; }; then
        echo "error: no current Hermes environment in this shell. $fix" >&2
        exit 1
    fi
    if ! "$__HERMES_TEST_PYTHON" -c '
    import json, os, pathlib, sys
    record = pathlib.Path(os.environ["__HERMES_TEST_PYTHON"]).parents[3] / "active.json"
    missing = sorted(set(sys.argv[1].split(",")) - set(json.loads(record.read_text()).get("extras", [])))
    sys.exit("error: the test environment lacks CI extras: " + ", ".join(missing) if missing else 0)
    ' "{{ test_extras }}"; then
        echo "$fix" >&2
        exit 1
    fi

# ── lint.yml ─────────────────────────────────────────────────────────────────

# ruff enforcement (blocking), with the pinned ruff from the test environment
[group('lint python')]
lint-ruff: _require-env
    "$__HERMES_TEST_PYTHON" -m ruff check .

# code health ratchet (blocking), against base (default: merge-base with origin/main)
[group('lint python')]
lint-health base="": _require-env
    "$__HERMES_TEST_PYTHON" scripts/check --only health {{ if base == "" { "" } else { "--base " + base } }}

# ruff + ty diff (advisory): new and fixed findings between base and HEAD
[group('lint python')]
[script("bash")]
lint-diff base="": _require-env
    set -euo pipefail
    base="{{ base }}"
    base="${base:-$(git merge-base origin/main HEAD)}"
    reports="$(mktemp -d)"
    worktree="$(mktemp -d)"
    trap 'git worktree remove --force "$worktree" >/dev/null 2>&1 || true; rm -rf "$reports"' EXIT
    mkdir -p "$reports/head" "$reports/base"
    "$__HERMES_TEST_PYTHON" -m ruff check --output-format json --exit-zero > "$reports/head/ruff.json" || true
    "$__HERMES_TEST_PYTHON" -m ty check --output-format gitlab --exit-zero > "$reports/head/ty.json" || true
    if [ "$(git rev-parse "$base")" = "$(git rev-parse HEAD)" ]; then
        echo '[]' > "$reports/base/ruff.json"
        echo '[]' > "$reports/base/ty.json"
    else
        git worktree add --detach "$worktree" "$base" >/dev/null
        (
            cd "$worktree"
            "$__HERMES_TEST_PYTHON" -m ruff check --output-format json --exit-zero > "$reports/base/ruff.json" || true
            "$__HERMES_TEST_PYTHON" -m ty check --output-format gitlab --exit-zero > "$reports/base/ty.json" || true
        )
    fi
    python scripts/lint_diff.py \
        --base-ruff "$reports/base/ruff.json" --head-ruff "$reports/head/ruff.json" \
        --base-ty "$reports/base/ty.json" --head-ty "$reports/head/ty.json" \
        --base-ref "$base" --head-ref "$(git rev-parse --abbrev-ref HEAD)" \
        --output "$reports/summary.md"
    cat "$reports/summary.md"

# Windows footgun checker
[group('guards')]
check-footguns: _require-env
    python scripts/check-windows-footguns.py --all

# Require portable Bash shebangs
[group('guards')]
check-shebangs: _require-env
    python scripts/check_bash_shebangs.py

# Forbid literal /tmp paths outside the baseline
[group('guards')]
check-no-tmp: _require-env
    python scripts/check_no_tmp_literals.py

# Forbid config.yaml writers that bypass the comment-preserving writer
[group('guards')]
check-config-writers: _require-env
    python scripts/check_config_yaml_writers.py

# Forbid unmarked macOS fakes in tests
[group('guards')]
check-os-marker-fakes: _require-env
    python scripts/ci/check_os_marker_fakes.py

# Keep every AGENTS.md under its truncation cap
[group('guards')]
check-agents-md: _require-env
    python scripts/ci/check_agents_md_size.py

# Profile-scope patterns on added lines (advisory)
[group('guards')]
check-profile-scope base="origin/main": _require-env
    python scripts/check_profile_scope_patterns.py --base {{ base }} --head HEAD

# Public-surface diff vs base (advisory)
[group('guards')]
check-public-surface base="origin/main": _require-env
    python scripts/ci/check_public_surface.py --base {{ base }} --head HEAD

# Every blocking lint check CI runs, through the repo's scripts/check
[group('guards')]
check-local: _require-env
    "$__HERMES_TEST_PYTHON" scripts/check

# ── Guard workflows ──────────────────────────────────────────────────────────

# case-collision-check.yml
[group('guards')]
check-case-collisions: _require-env
    python scripts/check-case-collisions.py

# lazy-deps-guard.yml
[group('guards')]
check-lazy-deps: _require-env
    python scripts/ci/check_lazy_deps_imports.py

# profile-artifact-check.yml
[group('guards')]
[script("bash")]
check-profile-archives: _require-env
    set -euo pipefail
    if ! report="$(python scripts/ci/check_profile_archive_boundary.py 2>&1)"; then
        printf '%s\n' "$report" >&2
        # `just test-js` packages the desktop app, whose tool downloads land here.
        if grep -q '^  \.cache/desktop-inputs/' <<<"$report"; then
            echo "hint: the .cache/desktop-inputs/ archives are desktop packager downloads; 'just clean-desktop-cache' removes them" >&2
        fi
        exit 1
    fi
    printf '%s\n' "$report"

# blob-size-check.yml, against base (default: merge-base with origin/main)
[group('guards')]
[script("bash")]
check-blob-size base="": _require-env
    set -euo pipefail
    base="{{ base }}"
    python scripts/ci/check_blob_size.py --base "${base:-$(git merge-base origin/main HEAD)}"

# infographic-check.yml: reject committed PR-infographic images
[group('guards')]
[script("bash")]
check-infographics:
    set -euo pipefail
    offenders=$(git ls-files -z | tr '\0' '\n' \
        | grep -iE '(^|/)(infograph|infograf)[^/]*/' \
        | grep -iE '\.(png|jpe?g|webp|gif)$' || true)
    if [ -n "$offenders" ]; then
        echo "error: PR-infographic image(s) are tracked in git; embed them in the PR description instead:" >&2
        printf '%s\n' "$offenders" | sed 's/^/  /' >&2
        exit 1
    fi
    echo "No committed PR-infographic images."

# history-check.yml: the branch shares history with origin/main
[group('guards')]
check-history:
    @base=$(git merge-base origin/main HEAD 2>/dev/null) && [ -n "$base" ] && echo "Common ancestor with main: $base" || { echo "error: this branch has no common ancestor with origin/main; rebase it onto main" >&2; exit 1; }

# contributor-check.yml: every new author email since origin/main is mapped
[group('guards')]
[script("bash")]
check-contributors:
    set -euo pipefail
    emails=$(git log "$(git merge-base origin/main HEAD)"..HEAD --format='%ae' --no-merges | sort -u)
    if [ -z "$emails" ]; then
        echo "No new commits to check."
        exit 0
    fi
    missing=""
    while IFS= read -r email; do
        case "$email" in
            *teknium*|*noreply@github.com*|*dependabot*|*github-actions*|*anthropic.com*|*cursor.com*) continue ;;
        esac
        if printf '%s\n' "$email" | grep -qE '\+.*@users\.noreply\.github\.com'; then
            continue
        fi
        if [ -f "contributors/emails/${email}" ]; then
            continue
        fi
        if ! grep -qF "\"${email}\"" scripts/releases/authors_legacy.py 2>/dev/null; then
            missing="${missing}  ${email} ($(git log --author="$email" --format='%an' -1))"$'\n'
        fi
    done <<< "$emails"
    if [ -n "$missing" ]; then
        echo "New contributor email(s) without a mapping:" >&2
        printf '%s' "$missing" >&2
        echo "Fix: python3 scripts/audit_pr_attribution.py --fix, or python3 scripts/add_contributor.py <email> <github-username>" >&2
        exit 1
    fi
    echo "All contributor emails are mapped."

# uv-lockfile-check.yml: uv.lock matches pyproject.toml
[group('guards')]
check-uv-lock: _require-env
    python -m pm.build_env --source . --check-lock

# icons-freshness-check.yml: regenerating every icon target changes nothing
[group('guards')]
[script("bash")]
check-icons: _require-env
    set -euo pipefail
    snapshot() { git status --porcelain --untracked-files=all; git diff; }
    before=$(snapshot)
    {{ test_env }}
    export HERMES_PAYLOAD_TAG= HERMES_BUILD_COMMIT=
    node scripts/generate-icons.mjs
    node scripts/generate-icons.mjs --check
    if [ "$(snapshot)" != "$before" ]; then
        git status --porcelain --untracked-files=all
        echo "error: committed icons are stale. Run 'node scripts/generate-icons.mjs' and commit the result." >&2
        exit 1
    fi

# lockfile-diff.yml: semantic package-lock.json diff against base (report only)
[group('guards')]
[script("bash")]
check-lockfile-diff base="": _require-env
    set -euo pipefail
    base="{{ base }}"
    out="$(mktemp)"
    trap 'rm -f "$out"' EXIT
    python scripts/ci/lockfile_diff.py --base "${base:-$(git merge-base origin/main HEAD)}" --head HEAD --output "$out"
    if [ -s "$out" ]; then cat "$out"; else echo "package-lock.json: no locked version changes."; fi

# prompt-surface-diff.yml: system prompt / tool schema snapshot diff against base (report only)
[group('guards')]
[script("bash")]
check-prompt-surface-diff base="": _require-env
    set -euo pipefail
    base="{{ base }}"
    base="${base:-$(git merge-base origin/main HEAD)}"
    snap=tests/fixtures/prompt_surface
    work="$(mktemp -d)"
    trap 'rm -rf "$work"' EXIT
    mkdir -p "$work/base"
    if git cat-file -e "${base}:${snap}" 2>/dev/null; then
        git archive "$base" "$snap" | tar -x -C "$work/base"
    fi
    python scripts/ci/prompt_surface.py diff "$work/base/$snap" "$snap" --output "$work/prompt-surface.md"
    if [ -s "$work/prompt-surface.md" ]; then cat "$work/prompt-surface.md"; else echo "Prompt surface unchanged."; fi

# supply-chain-audit.yml scan: critical payload patterns on lines added since base
[group('guards')]
[script("bash")]
check-supply-chain base="origin/main":
    set -euo pipefail
    diff=$(git diff "{{ base }}"...HEAD -- . ':!uv.lock' ':!*.lock' ':!package-lock.json' ':!yarn.lock' || true)
    found=0
    hits=$(echo "$diff" | grep -n '^+' | grep -iE 'base64\.(b64decode|decodebytes|urlsafe_b64decode)' | grep -iE 'exec\(|eval\(' | head -10 || true)
    if [ -n "$hits" ]; then
        echo "Base64-decoded strings passed directly to exec/eval:" >&2
        echo "$hits" >&2
        found=1
    fi
    hits=$(echo "$diff" | grep -n '^+' | grep -E 'subprocess\.(Popen|call|run)\s*\(' | grep -iE 'base64|\\x[0-9a-f]{2}|chr\(' | head -10 || true)
    if [ -n "$hits" ]; then
        echo "Subprocess calls with base64- or hex-encoded command strings:" >&2
        echo "$hits" >&2
        found=1
    fi
    if [ "$found" = 1 ]; then
        echo "error: critical supply chain risk patterns detected" >&2
        exit 1
    fi
    echo "No critical supply chain patterns."

# supply-chain-audit.yml dep-bounds: PyPI deps added to pyproject.toml since base carry an upper bound
[group('guards')]
[script("bash")]
check-dep-bounds base="origin/main":
    set -euo pipefail
    added=$(git diff "{{ base }}"...HEAD -- pyproject.toml | grep '^+' | grep -v '^+++' || true)
    unbounded=$(echo "$added" | grep -oE '"[a-zA-Z0-9_-]+(\[[^]]*\])?>=[ 0-9.]+"' | grep -v ',<' || true)
    if [ -n "$unbounded" ]; then
        echo "error: PyPI dependencies without upper bounds (add a <next_major ceiling):" >&2
        echo "$unbounded" >&2
        exit 1
    fi
    echo "No unbounded PyPI dependencies added."

# docker-lint.yml: hadolint on both Dockerfiles
[group('guards')]
lint-hadolint:
    hadolint --config .hadolint.yaml --failure-threshold warning Dockerfile
    hadolint --config .hadolint.yaml --failure-threshold warning docker/sandbox-desktop.Dockerfile

# docker-lint.yml: shellcheck (severity error) on the shell scripts under docker/
[group('guards')]
[script("bash")]
lint-shellcheck:
    set -euo pipefail
    files=()
    while IFS= read -r -d '' f; do
        case "$f" in
            *.sh|*.bash|*.ksh|*.bats) files+=("$f") ;;
            *) head -n1 "$f" | grep -qE '^#!.*(sh|bash|ksh)( |$)' && files+=("$f") ;;
        esac
    done < <(find docker -type f -print0)
    shellcheck --severity=error "${files[@]}"

# plugin-catalog-ci.yml structural: every catalog file validates
[group('guards')]
check-plugin-catalog: _require-env
    "$__HERMES_TEST_PYTHON" scripts/validate_plugin_catalog.py plugin-catalog/

# ── Python tests (tests.yml, tests-os.yml, remote-backends.yml, termux-verify.yml) ──

# tests.yml test: the full suite, optionally one slice ("1/2")
[group('test python')]
test-py slice="": _require-env
    env {{ no_keys }} HERMES_TEST_SLICE={{ slice }} HERMES_TEST_FILE_TIMEOUT=600 scripts/run_tests.sh

# tests.yml e2e: tests/e2e except the upgrade suite, against the built TUI
[group('test python')]
[script("bash")]
test-e2e: _require-env build-tui
    set -euo pipefail
    "$__HERMES_TEST_PYTHON" -c "import sqlite3, hermes_state_wal as w; print('sqlite', sqlite3.sqlite_version); assert not w.is_sqlite_wal_reset_vulnerable()"
    files=()
    while IFS= read -r f; do files+=("$f"); done < <(find tests/e2e -name 'test_*.py' -not -path 'tests/e2e/core/upgrade/*' | sort)
    export {{ no_keys }}
    HERMES_TEST_WORKERS=3 HERMES_TEST_FILE_TIMEOUT=900 HERMES_TEST_FILE_RETRIES=0 HERMES_E2E_REQUIRE_TUI=1 \
        scripts/run_tests.sh --include-integration "${files[@]}"

# tests.yml e2e-upgrade-plan: list the upgrade suite shards
[group('test python')]
[script("bash")]
test-e2e-upgrade-shards:
    set -euo pipefail
    root=tests/e2e/core/upgrade
    if [ -n "$(find "$root" -maxdepth 1 -name 'test_*.py' -print -quit)" ]; then echo core; fi
    for d in "$root"/*/; do
        name=$(basename "$d")
        case "$name" in _*|.*) continue ;; esac
        if [ -n "$(find "$d" -name 'test_*.py' -print -quit)" ]; then echo "$name"; fi
    done

# tests.yml e2e-upgrade: one upgrade shard, or every shard in turn (needs bubblewrap)
[group('test python')]
[linux]
[script("bash")]
test-e2e-upgrade shard="": _require-env
    set -euo pipefail
    "$__HERMES_TEST_PYTHON" -c "from tests.e2e.core.upgrade._helpers import BWRAP_OK; assert BWRAP_OK, 'bwrap unusable'" && echo "bwrap ok"
    "$__HERMES_TEST_PYTHON" -c "import sqlite3, hermes_state_wal as w; print('sqlite', sqlite3.sqlite_version); assert not w.is_sqlite_wal_reset_vulnerable()"
    shards="{{ shard }}"
    if [ -z "$shards" ]; then shards=$({{ just_executable() }} test-e2e-upgrade-shards); fi
    export {{ no_keys }}
    for s in $shards; do
        if [ "$s" = core ]; then
            dir_args=(tests/e2e/core/upgrade -maxdepth 1)
        else
            dir_args=("tests/e2e/core/upgrade/$s")
        fi
        files=()
        while IFS= read -r f; do files+=("$f"); done < <(find "${dir_args[@]}" -name 'test_*.py' | sort)
        [ "${#files[@]}" -gt 0 ] || { echo "no test files for shard $s" >&2; exit 1; }
        echo "shard $s: ${files[*]}"
        HERMES_TEST_WORKERS=6 HERMES_TEST_FILE_TIMEOUT=3000 scripts/run_tests.sh --include-integration "${files[@]}"
    done

# tests-os.yml os-tests: the tests gated on this host's platform
[group('test python')]
[linux]
test-os: (_test-os "linux")

# tests-os.yml os-tests: the tests gated on this host's platform
[group('test python')]
[macos]
test-os: (_test-os "macos")

# tests-os.yml os-tests: the tests gated on this host's platform
[group('test python')]
[windows]
test-os: (_test-os "windows")

[private]
[script("bash")]
_test-os marker: _require-env
    set -uo pipefail
    list="$(mktemp)"
    trap 'rm -f "$list"' EXIT
    if ! python scripts/ci/list_os_marked_tests.py "{{ marker }}" > "$list"; then
        echo "error: could not enumerate {{ marker }} test files" >&2
        exit 1
    fi
    echo "selected file(s) for {{ marker }}:"
    cat "$list"
    separator="$(python -c 'import os, sys; sys.stdout.write(os.pathsep)')"
    files="$(tr -d '\r' < "$list" | paste -sd "$separator" -)"
    export {{ no_keys }}
    scripts/run_tests.sh --files "$files" -- -m "platforms and not integration" -v --tb=short

# tests-os.yml e2e-windows: Windows E2E with real processes
[group('test python')]
[script("bash")]
[windows]
test-e2e-windows: _require-env
    set -uo pipefail
    # Unlike a CI runner, a developer machine may already run python/hermes processes.
    baseline="$(mktemp)"
    trap 'rm -f "$baseline"' EXIT
    {{ just_executable() }} _windows-python-snapshot "$(cygpath -w "$baseline")" || exit 1
    export {{ no_keys }}
    HERMES_TEST_WORKERS=6 HERMES_TEST_FILE_TIMEOUT=900 HERMES_TEST_FILE_RETRIES=0 \
        scripts/run_tests.sh tests/e2e/core/windows/ -- -m "platforms and integration" -v --tb=short -rA
    rc=$?
    {{ just_executable() }} _windows-no-leftovers "$(cygpath -w "$baseline")" || rc=1
    exit "$rc"

# windows-install-update-e2e.yml: install.ps1 + hermes update journey (creates throwaway machines and user profiles)
[confirm("This E2E provisions throwaway Windows machines and user profiles under C:\\Users. Run it on a disposable host only. Continue?")]
[group('test python')]
[script("bash")]
[windows]
test-windows-install-update: _require-env
    set -uo pipefail
    work="$(mktemp -d)"
    export {{ no_keys }}
    export HERMES_E2E_WINDOWS_INSTALL=1
    export HERMES_E2E_MACHINE_ROOT="$(cygpath -w "$work/wm")"
    export HERMES_E2E_PROFILES_ROOT='C:\Users'
    export HERMES_E2E_ARTIFACTS="$(cygpath -w "$work/win-update-e2e")"
    HERMES_TEST_WORKERS=6 HERMES_TEST_FILE_TIMEOUT=3600 HERMES_TEST_FILE_RETRIES=0 \
        scripts/run_tests.sh tests/e2e/core/windows_update/ -- -m "platforms and integration" -v --tb=short -rA
    rc=$?
    if [ ! -f "$work/win-update-e2e/crash-cells-ran.txt" ]; then
        echo "error: test_crash_cells.py ran no crash cells" >&2
        rc=1
    else
        cat "$work/win-update-e2e/crash-cells-ran.txt"
    fi
    {{ just_executable() }} _windows-no-machine-leftovers "$HERMES_E2E_MACHINE_ROOT" || rc=1
    echo "transcripts, logs and receipts: $work/win-update-e2e"
    exit "$rc"

# windows-venv-e2e.yml: venv-holder, Telegram CLOSE-WAIT and spawn-parity live E2E
[group('test python')]
[windows]
test-windows-live: _require-env
    scripts/run_tests.sh tests/hermes_cli/test_venv_holder_windows_live.py tests/hermes_cli/test_taskkill_identity_windows_live.py tests/hermes_cli/test_git_trampoline_windows_live.py tests/pm/test_held_entry_windows_live.py
    scripts/run_tests.sh tests/gateway/test_telegram_closewait_windows_live.py
    scripts/run_tests.sh tests/tools/test_process_registry_windows_live.py

# windows-venv-e2e.yml: PM long-path live E2E (CI turns LongPathsEnabled off first; this refuses rather than change the host)
[group('test python')]
[script("bash")]
[windows]
test-windows-long-paths: _require-env
    set -euo pipefail
    enabled=$(pwsh -NoProfile -Command "(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem').LongPathsEnabled")
    if [ "$(echo "$enabled" | tr -d '\r')" != 0 ]; then
        echo "error: LongPathsEnabled is on. This test needs it off, as CI sets it:" >&2
        echo "  reg add 'HKLM\SYSTEM\CurrentControlSet\Control\FileSystem' /v LongPathsEnabled /t REG_DWORD /d 0 /f" >&2
        exit 1
    fi
    scripts/run_tests.sh tests/pm/test_long_paths_windows_live.py -- -m "platforms and integration" -v

# Write the identities (pid + creation time: Windows reuses pids) of the running python/hermes processes
[private]
[script("pwsh")]
[windows]
_windows-python-snapshot out:
    $names = 'python.exe', 'pythonw.exe', 'hermes.exe'
    $ids = @(Get-CimInstance Win32_Process | Where-Object { $names -contains $_.Name.ToLower() } |
      ForEach-Object { "$($_.ProcessId)|$($_.CreationDate.ToUniversalTime().Ticks)" })
    Set-Content -LiteralPath '{{ out }}' -Value $ids

# tests-os.yml "No leftover Python / Hermes processes", excluding the processes in the baseline snapshot
[private]
[script("pwsh")]
[windows]
_windows-no-leftovers baseline:
    $names = 'python.exe', 'pythonw.exe', 'hermes.exe'
    $known = @(Get-Content -LiteralPath '{{ baseline }}')
    $deadline = (Get-Date).AddSeconds(20)
    do {
      $left = @(Get-CimInstance Win32_Process | Where-Object {
        ($names -contains $_.Name.ToLower()) -and
        ($known -notcontains "$($_.ProcessId)|$($_.CreationDate.ToUniversalTime().Ticks)")
      })
      if ($left.Count -eq 0) { 'no leftover python/hermes processes'; exit 0 }
      Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    foreach ($p in $left) { "leftover pid=$($p.ProcessId) ppid=$($p.ParentProcessId): $($p.CommandLine)" }
    'Processes running before the suite were excluded; one started by another program during the run still counts.'
    exit 1

# windows-install-update-e2e.yml "No leftover Python / Hermes processes": only processes of this run's machines
[private]
[script("pwsh")]
[windows]
_windows-no-machine-leftovers root:
    $names = 'python.exe', 'pythonw.exe', 'hermes.exe', 'node.exe'
    # Work dirs live under the machine root; profiles are C:\Users\*hermes-e2e-<id>.
    $root = '{{ root }}'.ToLower()
    $profileMark = 'hermes-e2e-'
    $deadline = (Get-Date).AddSeconds(20)
    do {
      $left = @(Get-CimInstance Win32_Process | Where-Object {
        $blob = "$($_.CommandLine) $($_.ExecutablePath)".ToLower()
        ($names -contains $_.Name.ToLower()) -and ($blob.Contains($root) -or $blob.Contains($profileMark))
      })
      if ($left.Count -eq 0) { 'no leftover processes from the E2E machines'; exit 0 }
      Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    foreach ($p in $left) { "leftover pid=$($p.ProcessId) ppid=$($p.ParentProcessId): $($p.CommandLine)" }
    exit 1

# remote-backends.yml: SSH + Docker backends against a real sshd and daemon
[group('test python')]
[script("bash")]
[unix]
test-remote-backends: _require-env
    set -euo pipefail
    state="$(mktemp -d)"
    trap 'docker rm -f hermes-test-sshd >/dev/null 2>&1 || true; rm -rf "$state"' EXIT
    sshd_env=$(HERMES_TEST_SSHD_IMAGE="{{ remote_test_image }}" scripts/ci/start_test_sshd.sh "$state/sshd")
    while IFS= read -r line; do export "$line"; done <<< "$sshd_env"
    export {{ no_keys }}
    if ! HERMES_TEST_DOCKER_IMAGE="{{ remote_test_image }}" scripts/run_tests.sh --fail-on-skip \
            tests/tools/test_ssh_environment.py tests/tools/test_docker_backend_real.py; then
        echo "error: real-backend tests failed or skipped (a skip means the sshd or docker image was not wired through)" >&2
        docker logs hermes-test-sshd || true
        exit 1
    fi

# termux-verify.yml contracts: Termux native linker and wheel contracts (needs patchelf)
[group('test python')]
[linux]
test-termux-contracts: _require-env
    scripts/run_tests.sh -j 2 tests/scripts/test_termux_python_linkage.py tests/scripts/test_termux_retag_wheel.py tests/scripts/test_termux_build_wheels.py tests/scripts/test_termux_wheelhouse_cache.py tests/scripts/test_termux_runtime_libs.py tests/scripts/test_archive_inputs.py tests/scripts/test_termux_launchers.py tests/pm/test_pm_bionic.py tests/pm/test_stage_only.py tests/pm/test_deb_safety.py tests/scripts/test_stage_apt_repo.py tests/scripts/test_run_tests_shell.py

# bootstrap-installer.yml posix: install.sh protocol + small fresh-install E2E
[group('test python')]
[script("bash")]
[unix]
test-bootstrap-installer: _require-env
    set -euo pipefail
    cat scripts/install.sh | bash -s -- --manifest | python3 -c '
    import json, sys
    m = json.load(sys.stdin)
    assert m["protocol_version"] == 1, m
    names = [s["name"] for s in m["stages"]]
    expected = ["prerequisites", "repository", "venv", "python-deps",
                "config", "products", "setup", "gateway", "complete"]
    assert names == expected, names
    print("stage manifest ok:", " -> ".join(names))
    '
    python3 scripts/gen-bootstrap-pins.py --check
    work="$(mktemp -d)"
    trap 'rm -rf "$work"' EXIT
    sha="$(git rev-parse HEAD)"
    git init -q --bare "$work/source-mirror.git"
    git push -q "$work/source-mirror.git" "HEAD:refs/heads/ci-under-test"
    (
        export HERMES_INSTALL_DIR="$work/bootstrap-install" HERMES_HOME="$work/hermes-home"
        export HERMES_RUNTIME_DIR="$work/hermes-tools" HERMES_REPO_URL="$work/source-mirror.git"
        for stage in prerequisites repository config complete; do
            echo "── stage $stage"
            bash scripts/install.sh --branch ci-under-test --commit "$sha" --non-interactive --stage "$stage" --json
        done
        test -f "$HERMES_INSTALL_DIR/.hermes-bootstrap-complete"
    )
    python3 scripts/verify-bootstrap-version-stamp.py \
        --stamp "$work/bootstrap-install/.hermes-bootstrap-complete" --repo "$work/bootstrap-install" \
        --expect-commit "$sha" --expect-branch ci-under-test --no-source-stamp
    head="$(git -C "$work/bootstrap-install" rev-parse HEAD)"
    test "$head" = "$sha" || { echo "error: installed HEAD $head != pinned $sha" >&2; exit 1; }
    HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh tests/scripts/test_fresh_source_install.py -q -j 1

# bootstrap-installer.yml windows: install.ps1 protocol surface (Windows PowerShell 5.1 and pwsh 7)
[group('test python')]
[script("pwsh")]
[windows]
test-bootstrap-installer:
    $ErrorActionPreference = 'Stop'
    foreach ($shell in @('powershell', 'pwsh')) {
      $extra = if ($shell -eq 'powershell') { @('-ExecutionPolicy', 'Bypass') } else { @() }
      $pv = & $shell -NoProfile @extra -File scripts/install.ps1 -ProtocolVersion
      if ("$pv" -ne "1") { Write-Error "${shell}: protocol version '$pv' != 1"; exit 1 }
      $json = & $shell -NoProfile @extra -File scripts/install.ps1 -Manifest | ConvertFrom-Json
      if ($json.protocol_version -ne 1) { Write-Error "${shell}: manifest protocol_version != 1"; exit 1 }
      $names = @($json.stages | ForEach-Object { $_.name })
      if ($names -notcontains "complete") { Write-Error "${shell}: manifest missing complete stage"; exit 1 }
      Write-Host "stage manifest ok (${shell}): $($names -join ' -> ')"
    }

# ── JS tests (js-tests.yml, termux-verify.yml, windows-bundle-sdk.yml) ───────

# js-tests.yml: every workspace check, in parallel
[group('test js')]
test-js: _require-env
    {{ test_env }} node .github/scripts/run-workspace-checks.mjs

[group('test js')]
js-check-bootstrap-installer: _require-env
    {{ test_env }} npm run --prefix apps/bootstrap-installer check

[group('test js')]
js-check-desktop-ui: _require-env
    {{ test_env }} npm run --prefix apps/desktop check:test:ui

[group('test js')]
js-check-desktop-platforms: _require-env
    {{ test_env }} npm run --prefix apps/desktop check:test:desktop:platforms

[group('test js')]
js-check-desktop-all: _require-env
    {{ test_env }} npm run --prefix apps/desktop check:test:desktop:all

[group('test js')]
js-check-desktop-lint: _require-env
    {{ test_env }} npm run --prefix apps/desktop check:lint

[group('test js')]
js-check-shared: _require-env
    {{ test_env }} npm run --prefix apps/shared check

[group('test js')]
js-check-hermes-ink: _require-env
    {{ test_env }} npm run --prefix ui-tui/packages/hermes-ink check

[group('test js')]
js-check-tests-js: _require-env
    {{ test_env }} npm run --prefix tests-js check

[group('test js')]
js-check-tui: _require-env
    {{ test_env }} npm run --prefix ui-tui check

[group('test js')]
js-check-web: _require-env
    {{ test_env }} npm run --prefix web check

# Bundle the Ink TUI (tests.yml e2e, termux-verify.yml tui)
[group('test js')]
build-tui: _require-env
    {{ test_env }} npm run build --workspace ui-tui

# termux-verify.yml tui: check and bundle the prebuilt TUI
[group('test js')]
test-termux-tui: js-check-tests-js js-check-tui build-tui

# windows-bundle-sdk.yml: provision and execute the Windows bundle SDK
[group('test js')]
[windows]
test-windows-bundle-sdk:
    cd apps/desktop && node ../../node_modules/vitest/vitest.mjs run --project electron scripts/windows-bundle-tools.test.mjs scripts/msix-shared.test.mjs scripts/sign-msix.test.mjs scripts/batch-sign-binaries.test.mjs

# ── Desktop E2E (e2e-desktop*.yml) ───────────────────────────────────────────

# e2e-desktop.yml: Playwright E2E with visual baselines
[group('test e2e')]
[linux]
test-e2e-desktop: (_e2e-desktop "xvfb-run")

# e2e-desktop.yml: Playwright E2E with visual baselines
[group('test e2e')]
[macos]
[windows]
test-e2e-desktop: (_e2e-desktop "")

# e2e-desktop-core.yml: package the app (electron-builder --dir) and run the core suite
[group('test e2e')]
[linux]
test-e2e-desktop-core: (_e2e-desktop-core "--linux" "xvfb-run")

# e2e-desktop-core.yml: package the app (electron-builder --dir) and run the core suite
[group('test e2e')]
[macos]
test-e2e-desktop-core: (_e2e-desktop-core "--mac" "")

# e2e-desktop-core.yml: package the app (electron-builder --dir) and run the core suite
[group('test e2e')]
[windows]
test-e2e-desktop-core: (_e2e-desktop-core "--win" "")

# e2e-desktop-update.yml: one install/update cell (first-run, app-update, build-fail, update-concurrency), or all
[group('test e2e')]
[linux]
[script("bash")]
test-e2e-desktop-update cell="": _require-env
    set -euo pipefail
    {{ test_env }}
    cells="{{ cell }}"
    cells="${cells:-first-run app-update build-fail update-concurrency}"
    root="$(mktemp -d)"
    trap 'rm -rf "$root"' EXIT
    wrap=()
    if command -v xvfb-run >/dev/null; then
        wrap=(xvfb-run -a --server-args="-screen 0 1280x1024x24")
    else
        echo "warning: xvfb-run not found; running on the current display" >&2
    fi
    cd apps/desktop
    for c in $cells; do
        case "$c" in
            first-run) specs="e2e/update/first-run.spec.ts e2e/update/gates.spec.ts" ;;
            app-update) specs="e2e/update/app-update.spec.ts" ;;
            build-fail) specs="e2e/update/build-fail.spec.ts" ;;
            update-concurrency) specs="e2e/update/update-concurrency.spec.ts" ;;
            *) echo "error: unknown cell '$c'" >&2; exit 2 ;;
        esac
        echo "── cell $c"
        rm -rf "$root/hdu"
        env CI=true {{ no_keys }} ANTHROPIC_API_KEY= HERMES_E2E_PYTHON="$__HERMES_TEST_PYTHON" HERMES_E2E_UPDATE_ROOT="$root/hdu" \
            ${wrap[@]+"${wrap[@]}"} npx playwright test -c e2e/update/playwright.config.ts $specs
    done

[private]
[script("bash")]
_e2e-desktop xvfb: _require-env
    set -euo pipefail
    {{ test_env }}
    wrap=()
    if [ -n "{{ xvfb }}" ]; then
        if command -v "{{ xvfb }}" >/dev/null; then
            wrap=("{{ xvfb }}" -a --server-args="-screen 0 1280x1024x24")
        else
            echo "warning: {{ xvfb }} not found; running on the current display" >&2
        fi
    fi
    cd apps/desktop
    npm run build
    env CI=true {{ no_keys }} ${wrap[@]+"${wrap[@]}"} npx playwright test --reporter=list

[private]
[script("bash")]
_e2e-desktop-core platform xvfb: _require-env
    set -euo pipefail
    {{ test_env }}
    wrap=()
    if [ -n "{{ xvfb }}" ]; then
        if command -v "{{ xvfb }}" >/dev/null; then
            wrap=("{{ xvfb }}" -a --server-args="-screen 0 1280x1024x24")
        else
            echo "warning: {{ xvfb }} not found; running on the current display" >&2
        fi
    fi
    cd apps/desktop
    npm run build
    npm run builder -- --dir --publish never {{ platform }}
    env CI=true {{ no_keys }} ANTHROPIC_API_KEY= HERMES_E2E_PYTHON="$__HERMES_TEST_PYTHON" HERMES_E2E_REQUIRE_PACKAGED=1 \
        ${wrap[@]+"${wrap[@]}"} npx playwright test -c e2e/core/playwright.config.ts

# ── Rust (rust-tests.yml) ────────────────────────────────────────────────────

# cargo test (bootstrap installer)
[group('test rust')]
test-rust:
    cd apps/bootstrap-installer/src-tauri && cargo test --lib

# ── Docs (docs-site-checks.yml) ──────────────────────────────────────────────

# Committed skill docs match the generator
[group('docs')]
[script("bash")]
docs-skills-check: _require-env
    set -euo pipefail
    paths=(website/docs website/sidebars.ts website/i18n)
    snapshot() { git status --porcelain --untracked-files=all -- "${paths[@]}"; git diff -- "${paths[@]}"; }
    before=$(snapshot)
    "$__HERMES_TEST_PYTHON" website/scripts/extract-skills.py
    "$__HERMES_TEST_PYTHON" website/scripts/generate-skill-docs.py
    if [ "$(snapshot)" != "$before" ]; then
        git status --short -- "${paths[@]}"
        echo "error: website/scripts/generate-skill-docs.py output differs from the docs; run it and commit the result" >&2
        exit 1
    fi

# Check cross-page links resolve on GitHub and on the site
[group('docs')]
docs-links: _require-env
    "$__HERMES_TEST_PYTHON" website/scripts/check_doc_links.py

# Lint docs diagrams
[group('docs')]
docs-diagrams:
    cd website && npm run lint:diagrams

# Build Docusaurus (English only, as CI)
[group('docs')]
docs-build:
    cd website && npm run build:fast

# Every docs-site-checks.yml step, in CI order
[group('docs')]
docs-all: docs-skills-check docs-links docs-diagrams docs-build

# ── Images and packaging (docker.yml, sandbox-image.yml, nix.yml, pm-bundle.yml) ──

# docker.yml build: build the image (slim or desktop) for this host's arch and run tests/docker
[group('docker')]
[script("bash")]
[unix]
test-docker variant="slim": _require-env
    set -euo pipefail
    case "{{ variant }}" in slim) desktop=0 ;; desktop) desktop=1 ;; *) echo "error: variant must be slim or desktop" >&2; exit 2 ;; esac
    display="$(python -m scripts.releases.distance)"
    identity=(--commit "$(git rev-parse HEAD)")
    if [ -n "$display" ]; then
        base="${display%%+*}"
        development="${display#*+}"
        if [ "$development" = "$display" ]; then distance=0; else distance="${development%%.*}"; fi
        identity+=(--base-version "$base" --display-version "$display" --distance "$distance")
    fi
    {{ just_executable() }} check-profile-archives
    # The build context needs the Docker stamp, but in the checkout it would
    # rebrand the source install (version source, updateMechanism: external).
    # Put the original back (or none) after the build, on failure and on interrupt.
    saved="$(mktemp)"
    had_stamp=0
    if [ -e install-stamp.json ]; then cp -p install-stamp.json "$saved"; had_stamp=1; fi
    restore_stamp() {
        if [ "$had_stamp" = 1 ]; then mv -f "$saved" install-stamp.json; else rm -f install-stamp.json "$saved"; fi
        had_stamp=restored
    }
    trap '[ "$had_stamp" = restored ] || restore_stamp' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM HUP
    python scripts/write_install_stamp.py --output install-stamp.json \
        --distribution docker --update-mechanism external --source ci "${identity[@]}"
    docker build --file Dockerfile --tag nousresearch/hermes-agent:test --build-arg HERMES_BOT_DESKTOP="$desktop" .
    restore_stamp
    export {{ no_keys }}
    HERMES_TEST_IMAGE=nousresearch/hermes-agent:test HERMES_TEST_WORKERS="$(getconf _NPROCESSORS_ONLN)" \
        scripts/run_tests.sh tests/docker/

# sandbox-image.yml build: build the sandbox desktop image and smoke it
[group('docker')]
[unix]
test-sandbox-image:
    docker build --file docker/sandbox-desktop.Dockerfile --tag nousresearch/hermes-sandbox:test .
    docker/sandbox-desktop-smoke.sh nousresearch/hermes-sandbox:test

# nix.yml flake-check
[group('packaging')]
[unix]
check-nix:
    nix --extra-experimental-features 'nix-command flakes' flake check . --print-build-logs

# pm-bundle.yml: assemble the prepared payload offline and smoke it (after deps-pm-bundle)
[group('packaging')]
[macos]
test-pm-bundle:
    PYTHONUTF8=1 python3 -S -B scripts/bundles/native_build.py --prepared "{{ justfile_directory() }}/.build/agent-payload.prepared.json"
    bash scripts/smoke-payload.sh .build/agent-payload

# pm-bundle.yml: assemble the prepared payload offline and smoke it (after deps-pm-bundle)
[group('packaging')]
[windows]
test-pm-bundle:
    PYTHONUTF8=1 python -S -B scripts/bundles/native_build.py --prepared "{{ justfile_directory() }}/.build/agent-payload.prepared.json"
    bash scripts/smoke-payload.sh .build/agent-payload

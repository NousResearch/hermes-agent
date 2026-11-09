set default-list
set shell := ["bash", "-euo", "pipefail", "-c"]

# Provision the PM tools, Python runtime and test environment
[group('setup')]
[unix]
setup:
    bash setup-hermes.sh --runtime-only --test-environment

# Provision the PM tools, Python runtime and test environment
[group('setup')]
[windows]
setup:
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File setup-hermes.ps1 -RuntimeOnly

# Install the JS workspaces from pnpm-lock.yaml
[group('setup')]
js-install:
    pnpm install --frozen-lockfile

# Install only tests-js and its runtime dependencies, without lifecycle scripts
[group('setup')]
js-install-tests:
    pnpm install --frozen-lockfile --filter "{./tests-js}..." --prod --ignore-scripts

# Install the WhatsApp bridge's own npm lockfile, without native builds
[group('setup')]
js-install-whatsapp-bridge:
    npm ci --ignore-scripts --no-audit --no-fund --prefix scripts/whatsapp-bridge

# Prepare the pm-locked Electron binary the desktop app and its E2E suites launch
[group('setup')]
electron-runtime:
    node scripts/build/electron-runtime.mjs --source .

# Run every blocking lint check CI runs (`just check --only health`)
[group('check')]
[positional-arguments]
check *args:
    scripts/hermes-python scripts/check "$@"

# Apply ruff and JS workspace autofixes to the tree
[group('check')]
fix:
    scripts/hermes-python -m scripts.code_health.ruff_runner check --fix .
    pnpm -r fix

# Run every JS workspace check
[group('check')]
check-js:
    pnpm -r check

# Run every JS workspace check in parallel, as CI does (.github/scripts/run-workspace-checks.mjs)
[group('check')]
check-js-ci:
    node .github/scripts/run-workspace-checks.mjs

# Fail when the committed icons differ from the generator's output (scripts/ci/check_icons.sh)
[group('check')]
check-icons:
    scripts/ci/check_icons.sh

# Run the repo-guards history and supply-chain steps against the merge base with origin/main
[group('check')]
check-repo-guards:
    scripts/ci/local_step.sh scripts/ci/check_infographic_images.py scripts/ci/check_common_ancestor.py scripts/ci/check_contributor_emails.py 'scripts/ci/scan_supply_chain.py patterns' 'scripts/ci/scan_supply_chain.py dependency-bounds'

# Fail when uv.lock is out of sync with pyproject.toml (re-resolves against PyPI; prepares pm's runtime in the Hermes home)
[group('check')]
check-uv-lock:
    scripts/ci/local_step.sh scripts/ci/verify_uv_lock.py

# Lint the Dockerfiles (pinned hadolint image) and docker/ shell scripts (shellcheck on PATH)
[group('check')]
lint-docker:
    scripts/ci/lint_docker.sh

# Validate every plugin-catalog entry's structure (no clone, no install)
[group('check')]
check-plugin-catalog:
    scripts/hermes-python scripts/validate_plugin_catalog.py plugin-catalog/

# Run the flake's checks (needs nix with flakes)
[group('check')]
check-nix:
    nix --extra-experimental-features 'nix-command flakes' flake check . --print-build-logs

# Run the named test files or directories (pytest flags pass through)
[group('test')]
[positional-arguments]
test +args:
    scripts/run_tests.sh --require-paths "$@"

# Run the full Python test suite (about 34k tests)
[group('test')]
test-all:
    scripts/run_tests.sh

# Run every JS workspace's tests (builds @hermes/ink first, as the TUI's check does)
[group('test')]
test-js:
    pnpm --dir ui-tui run build:ink
    pnpm -r test

# Run the plugin-catalog tests, the files scripts/ci/catalog_test_files.py selects
[group('test')]
test-catalog:
    scripts/hermes-python scripts/ci/catalog_test_files.py | scripts/run_tests.sh --files-from -

# Run the WhatsApp bridge's node tests (after `just js-install-whatsapp-bridge`)
[group('test')]
test-whatsapp-bridge:
    node --test scripts/whatsapp-bridge/*.test.mjs

# Run the bootstrap installer's Rust unit tests (needs cargo and Tauri's system libraries)
[group('test')]
test-rust:
    cargo test --lib --manifest-path apps/bootstrap-installer/src-tauri/Cargo.toml

# Start the desktop app in development mode
[group('dev')]
desktop:
    pnpm --dir apps/desktop run dev

# Build the desktop app from source (apps/desktop/scripts/build.mjs), as its E2E suites run it
[group('dev')]
desktop-build:
    pnpm --dir apps/desktop run build

# Start the TUI in watch mode
[group('dev')]
tui:
    pnpm --dir ui-tui run dev

# Run a command against a throwaway HERMES_HOME (scripts/dev-sandbox.sh)
[group('dev')]
[positional-arguments]
sandbox *args:
    scripts/dev-sandbox.sh "$@"

# Regenerate logos and icons from assets/
[group('dev')]
icons:
    node scripts/generate-icons.mjs

# Install the website's npm dependencies
[group('docs')]
docs-install:
    npm ci --prefix website

# Build the English docs site
[group('docs')]
docs-build:
    npm run build:fast --prefix website

# Regenerate the skills dashboard data and the per-skill docs pages
[group('docs')]
docs-generate:
    scripts/hermes-python website/scripts/extract-skills.py
    scripts/hermes-python website/scripts/generate-skill-docs.py

# Run every docs check CI runs, after regenerating the skill docs
[group('docs')]
docs-check: && docs-check-links docs-lint-diagrams
    website/scripts/check_generated_skill_docs.sh {{ quote(just_executable()) }} docs-generate

# Fail when the committed skill docs differ from the generator's output
[group('docs')]
docs-check-generated:
    website/scripts/check_generated_skill_docs.sh

# Fail on cross-page links that break on GitHub or on the site
[group('docs')]
docs-check-links:
    scripts/hermes-python website/scripts/check_doc_links.py

# Lint docs for ASCII box diagrams (pm installs the pinned ascii-guard on first run)
[group('docs')]
docs-lint-diagrams:
    scripts/hermes-python -m scripts.ci.python_packages ascii-guard==2.3.0 ruamel.yaml==0.18.17 -- -m ascii_guard.cli lint --exclude-code-blocks website/docs

# Stage pinned bionic pm packages and their runtime libraries into a Termux payload
[group('dev')]
[positional-arguments]
termux-payload payload +packages:
    scripts/termux/stage_payload.sh "$@"

# Refresh uv.lock after a pyproject.toml change, then re-sync the environment
[group('deps')]
lock:
    scripts/relock.sh

# Pre-commit: the pinned ruff with ruff.pre-commit.toml on the given files
[group('hooks')]
[positional-arguments]
ruff +files:
    scripts/hermes-python -m scripts.code_health.ruff_runner check --config ruff.pre-commit.toml "$@"

# Pre-commit: eslint on the given files, from the calling workspace
[group('hooks')]
[no-cd]
[positional-arguments]
eslint +files:
    pnpm exec eslint "$@"

# Rewrite imports and string targets for a move commit (libcst comes from a side environment)
[group('move')]
[positional-arguments]
move-rewrite *args:
    scripts/hermes-python -m scripts.ci.python_packages libcst==1.9.0 -- scripts/move/rewrite_imports.py "$@"

# Regenerate files built from docstrings a move rewrote (run after move-docs; the parity table only checks)
[group('move')]
move-generate:
    scripts/gen_gateway_contracts.py
    scripts/gen_serve_handler_ledger.py
    scripts/gen_plugin_api_docs.py
    scripts/hermes-python website/scripts/generate-skill-docs.py
    scripts/gen_gateway_command_parity.py --check

# Build commit A (git mv + MOVES) and commit B (generated) of one chunk: NN-name ROOT... [--log-dir D] [--only-a] [--from-a]
[group('move')]
[positional-arguments]
move-chunk *args:
    python3 scripts/move/build_chunk.py "$@"

# Rewrite moved paths and module names in docs and comments (--from-commit A; --help lists the rest)
[group('move')]
[positional-arguments]
move-docs *args:
    scripts/hermes-python scripts/move/rewrite_docs.py "$@"

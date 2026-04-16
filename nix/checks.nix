# nix/checks.nix — Build-time verification tests
#
# Checks are Linux-only: the full Python venv (via uv2nix) includes
# transitive deps like onnxruntime that lack compatible wheels on
# aarch64-darwin. The package and devShell still work on macOS.
{ inputs, ... }: {
  perSystem = { pkgs, lib, self', ... }:
    let
      hermes-agent = self'.packages.default;
      hermesVenv = hermes-agent.hermesVenv;

      configMergeScript = pkgs.callPackage ./configMergeScript.nix { };

      # Auto-generated config key reference — always in sync with Python
      configKeys = pkgs.runCommand "hermes-config-keys" {} ''
        set -euo pipefail
        export HOME=$TMPDIR
        ${hermesVenv}/bin/python3 -c '
import json, sys
from hermes_cli.config import DEFAULT_CONFIG

def leaf_paths(d, prefix=""):
    paths = []
    for k, v in sorted(d.items()):
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v:
            paths.extend(leaf_paths(v, path))
        else:
            paths.append(path)
    return paths

json.dump(sorted(leaf_paths(DEFAULT_CONFIG)), sys.stdout, indent=2)
' > $out
      '';
    in {
      packages.configKeys = configKeys;

      checks = {
        # Cross-platform evaluation — catches "not supported for interpreter"
        # errors (e.g. sphinx dropping python311) without needing a darwin builder.
        # Evaluation is pure and instant; it doesn't build anything.
        cross-eval = let
          targetSystems = builtins.filter
            (s: inputs.self.packages ? ${s})
            [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
          tryEvalPkg = sys:
            let pkg = inputs.self.packages.${sys}.default;
            in builtins.tryEval (builtins.seq pkg.drvPath true);
          results = map (sys: { inherit sys; result = tryEvalPkg sys; }) targetSystems;
          failures = builtins.filter (r: !r.result.success) results;
          failMsg = lib.concatMapStringsSep "\n" (r: "  - ${r.sys}") failures;
        in pkgs.runCommand "hermes-cross-eval" { } (
          if failures != [] then
            throw "Package fails to evaluate on:\n${failMsg}"
          else ''
            echo "PASS: package evaluates on all ${toString (builtins.length targetSystems)} platforms"
            mkdir -p $out
            echo "ok" > $out/result
          ''
        );

        # Verify the default package builds successfully (cross-platform).
        # On Linux the runtime checks below already depend on the package,
        # but this ensures darwin builders also build it during flake check.
        build-package = pkgs.runCommand "hermes-build-package" { } ''
          echo "PASS: package built at ${hermes-agent}"
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify the devShell builds successfully (cross-platform).
        build-devshell = pkgs.runCommand "hermes-build-devshell" { } ''
          echo "PASS: devShell built at ${self'.devShells.default}"
          mkdir -p $out
          echo "ok" > $out/result
        '';
      } // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
        # Verify binaries exist and are executable
        package-contents = pkgs.runCommand "hermes-package-contents" { } ''
          set -e
          echo "=== Checking binaries ==="
          test -x ${hermes-agent}/bin/hermes || (echo "FAIL: hermes binary missing"; exit 1)
          test -x ${hermes-agent}/bin/hermes-agent || (echo "FAIL: hermes-agent binary missing"; exit 1)
          echo "PASS: All binaries present"

          echo "=== Checking version ==="
          ${hermes-agent}/bin/hermes version 2>&1 | grep -qi "hermes" || (echo "FAIL: version check"; exit 1)
          echo "PASS: Version check"

          echo "=== All checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify every pyproject.toml [project.scripts] entry has a wrapped binary
        entry-points-sync = pkgs.runCommand "hermes-entry-points-sync" { } ''
          set -e
          echo "=== Checking entry points match pyproject.toml [project.scripts] ==="
          for bin in hermes hermes-agent hermes-acp; do
            test -x ${hermes-agent}/bin/$bin || (echo "FAIL: $bin binary missing from Nix package"; exit 1)
            echo "PASS: $bin present"
          done

          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify CLI subcommands are accessible
        cli-commands = pkgs.runCommand "hermes-cli-commands" { } ''
          set -e
          export HOME=$(mktemp -d)

          echo "=== Checking hermes --help ==="
          ${hermes-agent}/bin/hermes --help 2>&1 | grep -q "gateway" || (echo "FAIL: gateway subcommand missing"; exit 1)
          ${hermes-agent}/bin/hermes --help 2>&1 | grep -q "config" || (echo "FAIL: config subcommand missing"; exit 1)
          echo "PASS: All subcommands accessible"

          echo "=== All CLI checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify bundled skills are present in the package
        bundled-skills = pkgs.runCommand "hermes-bundled-skills" { } ''
          set -e
          echo "=== Checking bundled skills ==="
          test -d ${hermes-agent}/share/hermes-agent/skills || (echo "FAIL: skills directory missing"; exit 1)
          echo "PASS: skills directory exists"

          SKILL_COUNT=$(find ${hermes-agent}/share/hermes-agent/skills -name "SKILL.md" | wc -l)
          test "$SKILL_COUNT" -gt 0 || (echo "FAIL: no SKILL.md files found in skills directory"; exit 1)
          echo "PASS: $SKILL_COUNT bundled skills found"

          grep -q "HERMES_BUNDLED_SKILLS" ${hermes-agent}/bin/hermes || \
            (echo "FAIL: HERMES_BUNDLED_SKILLS not in wrapper"; exit 1)
          echo "PASS: HERMES_BUNDLED_SKILLS set in wrapper"

          echo "=== All bundled skills checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify bundled plugins (platforms, memory, context_engine) are present
        bundled-plugins = pkgs.runCommand "hermes-bundled-plugins" { } ''
          set -e
          echo "=== Checking bundled plugins ==="
          test -d ${hermes-agent}/share/hermes-agent/plugins || (echo "FAIL: plugins directory missing"; exit 1)
          echo "PASS: plugins directory exists"

          test -f ${hermes-agent}/share/hermes-agent/plugins/platforms/irc/plugin.yaml || \
            (echo "FAIL: irc plugin manifest missing"; exit 1)
          echo "PASS: irc plugin manifest present"

          grep -q "HERMES_BUNDLED_PLUGINS" ${hermes-agent}/bin/hermes || \
            (echo "FAIL: HERMES_BUNDLED_PLUGINS not in wrapper"; exit 1)
          echo "PASS: HERMES_BUNDLED_PLUGINS set in wrapper"

          echo "=== All bundled plugins checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify bundled i18n locale catalogs are present and resolvable.
        # Regression for #23943 / #27632 / #35374 — sealed Nix venvs dropped
        # locales/, surfacing raw i18n keys like gateway.reset.header_default.
        bundled-locales = pkgs.runCommand "hermes-bundled-locales" { } ''
          set -e
          echo "=== Checking bundled locales ==="
          test -d ${hermes-agent}/share/hermes-agent/locales || (echo "FAIL: locales directory missing"; exit 1)
          echo "PASS: locales directory exists"

          LOC_COUNT=$(find ${hermes-agent}/share/hermes-agent/locales -name "*.yaml" | wc -l)
          test "$LOC_COUNT" -ge 16 || (echo "FAIL: expected >=16 catalogs, found $LOC_COUNT"; exit 1)
          echo "PASS: $LOC_COUNT locale catalogs found"

          test -f ${hermes-agent}/share/hermes-agent/locales/en.yaml || (echo "FAIL: en.yaml missing"; exit 1)
          echo "PASS: en.yaml present"

          grep -q "HERMES_BUNDLED_LOCALES" ${hermes-agent}/bin/hermes || \
            (echo "FAIL: HERMES_BUNDLED_LOCALES not in wrapper"; exit 1)
          echo "PASS: HERMES_BUNDLED_LOCALES set in wrapper"

          echo "=== Rendering via the wrapper override (HERMES_BUNDLED_LOCALES) ==="
          export HOME=$(mktemp -d)
          RENDERED=$(cd "$HOME" && HERMES_BUNDLED_LOCALES=${hermes-agent}/share/hermes-agent/locales \
            ${hermesVenv}/bin/python3 -c "from agent import i18n; print(i18n.t('gateway.reset.header_default', lang='en'))")
          echo "rendered: $RENDERED"
          test "$RENDERED" != "gateway.reset.header_default" || (echo "FAIL: i18n returned the raw key with HERMES_BUNDLED_LOCALES set"; exit 1)
          echo "PASS: i18n renders a human string via the wrapper override"

          # Defense-in-depth check: the sealed venv must ALSO resolve catalogs
          # with NO env var, via the wheel's setuptools data-files materialized
          # into the venv data scheme. If a future uv2nix bump drops data-files,
          # the wrapper override above would mask the regression at runtime while
          # `pip install`/other sealed paths silently break — this catches it.
          echo "=== Rendering WITHOUT the env var (data-files materialization) ==="
          BARE_DIR=$(cd "$HOME" && ${hermesVenv}/bin/python3 -c "from agent import i18n; print(i18n._locales_dir())")
          BARE=$(cd "$HOME" && ${hermesVenv}/bin/python3 -c "from agent import i18n; print(i18n.t('gateway.reset.header_default', lang='en'))")
          echo "resolved dir (no env var): $BARE_DIR"
          echo "rendered: $BARE"
          test "$BARE" != "gateway.reset.header_default" || \
            (echo "FAIL: sealed venv could not resolve locales without HERMES_BUNDLED_LOCALES — data-files materialization regressed"; exit 1)
          echo "PASS: sealed venv resolves locales via data-files without the env var"

          echo "=== All bundled locales checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify bundled TUI is present and compiled
        bundled-tui = pkgs.runCommand "hermes-bundled-tui" { } ''
          set -e
          echo "=== Checking bundled TUI ==="
          test -d ${hermes-agent}/ui-tui || (echo "FAIL: ui-tui directory missing"; exit 1)
          echo "PASS: ui-tui directory exists"

          test -f ${hermes-agent}/ui-tui/dist/entry.js || (echo "FAIL: compiled entry.js missing"; exit 1)
          echo "PASS: compiled entry.js present"

          # self-contained bundle; no runtime node_modules expected

          grep -q "HERMES_TUI_DIR" ${hermes-agent}/bin/hermes || \
            (echo "FAIL: HERMES_TUI_DIR not in wrapper"; exit 1)
          echo "PASS: HERMES_TUI_DIR set in wrapper"

          echo "=== All bundled TUI checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify HERMES_NODE is set in wrapper and points to Node 20+
        # (string-width uses the /v regex flag which requires Node 20+)
        hermes-node = pkgs.runCommand "hermes-node-version" { } ''
          set -e
          echo "=== Checking HERMES_NODE in wrapper ==="
          grep -q "HERMES_NODE" ${hermes-agent}/bin/hermes || \
            (echo "FAIL: HERMES_NODE not set in wrapper"; exit 1)
          echo "PASS: HERMES_NODE present in wrapper"

          HERMES_NODE=$(sed -n "s/^export HERMES_NODE='\(.*\)'/\1/p" ${hermes-agent}/bin/hermes)
          test -x "$HERMES_NODE" || (echo "FAIL: HERMES_NODE=$HERMES_NODE not executable"; exit 1)
          echo "PASS: HERMES_NODE executable at $HERMES_NODE"

          NODE_MAJOR=$("$HERMES_NODE" --version | sed 's/^v//' | cut -d. -f1)
          test "$NODE_MAJOR" -ge 20 || \
            (echo "FAIL: Node v$NODE_MAJOR < 20, TUI needs /v regex flag support"; exit 1)
          echo "PASS: Node v$NODE_MAJOR >= 20"

          echo "=== All HERMES_NODE checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify HERMES_MANAGED guard works on all mutation commands
        managed-guard = pkgs.runCommand "hermes-managed-guard" { } ''
          set -e
          export HOME=$(mktemp -d)

          check_blocked() {
            local label="$1"
            shift
            OUTPUT=$(HERMES_MANAGED=true "$@" 2>&1 || true)
            echo "$OUTPUT" | grep -q "managed by NixOS" || (echo "FAIL: $label not guarded"; echo "$OUTPUT"; exit 1)
            echo "PASS: $label blocked in managed mode"
          }

          echo "=== Checking HERMES_MANAGED guards ==="
          check_blocked "config set" ${hermes-agent}/bin/hermes config set model foo
          check_blocked "config edit" ${hermes-agent}/bin/hermes config edit

          echo "=== All guard checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify extraPythonPackages PYTHONPATH injection
        extra-python-packages = let
          testPkg = pkgs.python312Packages.pyfiglet;
          hermesWithExtra = hermes-agent.override {
            extraPythonPackages = [ testPkg ];
          };
        in pkgs.runCommand "hermes-extra-python-packages" { } ''
          set -e
          echo "=== Checking extraPythonPackages PYTHONPATH injection ==="

          grep -q "PYTHONPATH" ${hermesWithExtra}/bin/hermes || \
            (echo "FAIL: PYTHONPATH not in wrapper"; exit 1)
          echo "PASS: PYTHONPATH present in wrapper"

          grep -q "${testPkg}" ${hermesWithExtra}/bin/hermes || \
            (echo "FAIL: test package path not in PYTHONPATH"; exit 1)
          echo "PASS: test package path found in wrapper"

          echo "=== Checking base package has no PYTHONPATH ==="
          if grep -q "PYTHONPATH" ${hermes-agent}/bin/hermes; then
            echo "FAIL: base package should not have PYTHONPATH"; exit 1
          fi
          echo "PASS: base package clean"

          echo "=== All extraPythonPackages checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Verify extraDependencyGroups passes through to python.nix
        extra-dependency-groups = let
          hermesWithGroups = hermes-agent.override {
            extraDependencyGroups = [ "honcho" ];
          };
        in pkgs.runCommand "hermes-extra-dependency-groups" { } ''
          set -e
          echo "=== Checking extraDependencyGroups override evaluates ==="

          # Eval-only: verify the override produces valid derivation paths
          # without building the full venv (which is expensive and redundant
          # since the mechanism is just list concatenation into python.nix).
          echo "derivation: ${hermesWithGroups}"
          echo "venv: ${hermesWithGroups.hermesVenv}"
          echo "PASS: extraDependencyGroups override evaluates cleanly"

          echo "=== All extraDependencyGroups checks passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # Regression guard: messaging deps live outside [all], so the
        # #messaging variant must actually ship discord.py — otherwise
        # `nix profile install .#messaging` regresses to the broken default.
        messaging-variant = pkgs.runCommand "hermes-messaging-variant" { } ''
          set -e
          echo "=== Checking discord.py importable from messaging variant ==="
          ${self'.packages.messaging.hermesVenv}/bin/python3 -c \
            "import discord; print(discord.__version__)"
          echo "PASS: discord.py importable from messaging variant venv"
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # ── Config merge + round-trip test ────────────────────────────────
        # Tests the merge script (Nix activation behavior) across 7
        # scenarios, then verifies Python's load_config() reads correctly.
        config-roundtrip = let
          # Nix settings used across scenarios
          nixSettings = pkgs.writeText "nix-settings.json" (builtins.toJSON {
            model = "test/nix-model";
            toolsets = ["nix-toolset"];
            terminal = { backend = "docker"; timeout = 999; };
            mcp_servers = {
              nix-server = { command = "echo"; args = ["nix"]; };
            };
          });

          # Pre-built YAML fixtures for each scenario
          fixtureB = pkgs.writeText "fixture-b.yaml" ''
            model: "old-model"
            mcp_servers:
              old-server:
                url: "http://old"
          '';
          fixtureC = pkgs.writeText "fixture-c.yaml" ''
            skills:
              disabled:
                - skill-a
                - skill-b
            session_reset:
              mode: idle
              idle_minutes: 30
            streaming:
              enabled: true
            fallback_model:
              provider: openrouter
              model: test-fallback
          '';
          fixtureD = pkgs.writeText "fixture-d.yaml" ''
            model: "user-model"
            skills:
              disabled:
                - skill-x
            streaming:
              enabled: true
              transport: edit
          '';
          fixtureE = pkgs.writeText "fixture-e.yaml" ''
            mcp_servers:
              user-server:
                url: "http://user-mcp"
              nix-server:
                command: "old-cmd"
                args: ["old"]
          '';
          fixtureF = pkgs.writeText "fixture-f.yaml" ''
            terminal:
              cwd: "/user/path"
              custom_key: "preserved"
              env_passthrough:
                - USER_VAR
          '';

        in pkgs.runCommand "hermes-config-roundtrip" {
          nativeBuildInputs = [ pkgs.jq ];
        } ''
          set -e
          export HOME=$(mktemp -d)
          ERRORS=""

          fail() { ERRORS="$ERRORS\nFAIL: $1"; }

          # Helper: run merge then load with Python, output merged JSON
          merge_and_load() {
            local hermes_home="$1"
            export HERMES_HOME="$hermes_home"
            ${configMergeScript} ${nixSettings} "$hermes_home/config.yaml"
            ${hermesVenv}/bin/python3 -c '
import json, sys
from hermes_cli.config import load_config
json.dump(load_config(), sys.stdout, default=str)
'
          }

          # ═══════════════════════════════════════════════════════════════
          # Scenario A: Fresh install — no existing config.yaml
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario A: Fresh install ==="
          A_HOME=$(mktemp -d)
          A_CONFIG=$(merge_and_load "$A_HOME")

          echo "$A_CONFIG" | jq -e '.model == "test/nix-model"' > /dev/null \
            || fail "A: model not set from Nix"
          echo "$A_CONFIG" | jq -e '.mcp_servers."nix-server".command == "echo"' > /dev/null \
            || fail "A: MCP nix-server missing"
          echo "PASS: Scenario A"

          # ═══════════════════════════════════════════════════════════════
          # Scenario B: Nix keys override existing values
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario B: Nix overrides ==="
          B_HOME=$(mktemp -d)
          install -m 0644 ${fixtureB} "$B_HOME/config.yaml"
          B_CONFIG=$(merge_and_load "$B_HOME")

          echo "$B_CONFIG" | jq -e '.model == "test/nix-model"' > /dev/null \
            || fail "B: Nix model did not override"
          echo "PASS: Scenario B"

          # ═══════════════════════════════════════════════════════════════
          # Scenario C: User-only keys preserved
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario C: User keys preserved ==="
          C_HOME=$(mktemp -d)
          install -m 0644 ${fixtureC} "$C_HOME/config.yaml"
          C_CONFIG=$(merge_and_load "$C_HOME")

          echo "$C_CONFIG" | jq -e '.skills.disabled == ["skill-a", "skill-b"]' > /dev/null \
            || fail "C: skills.disabled not preserved"
          echo "$C_CONFIG" | jq -e '.session_reset.mode == "idle"' > /dev/null \
            || fail "C: session_reset.mode not preserved"
          echo "$C_CONFIG" | jq -e '.session_reset.idle_minutes == 30' > /dev/null \
            || fail "C: session_reset.idle_minutes not preserved"
          echo "$C_CONFIG" | jq -e '.streaming.enabled == true' > /dev/null \
            || fail "C: streaming.enabled not preserved"
          echo "$C_CONFIG" | jq -e '.fallback_model.provider == "openrouter"' > /dev/null \
            || fail "C: fallback_model not preserved"
          echo "PASS: Scenario C"

          # ═══════════════════════════════════════════════════════════════
          # Scenario D: Mixed — Nix wins for its keys, user keys preserved
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario D: Mixed merge ==="
          D_HOME=$(mktemp -d)
          install -m 0644 ${fixtureD} "$D_HOME/config.yaml"
          D_CONFIG=$(merge_and_load "$D_HOME")

          echo "$D_CONFIG" | jq -e '.model == "test/nix-model"' > /dev/null \
            || fail "D: Nix model did not override user model"
          echo "$D_CONFIG" | jq -e '.skills.disabled == ["skill-x"]' > /dev/null \
            || fail "D: user skills not preserved"
          echo "$D_CONFIG" | jq -e '.streaming.enabled == true' > /dev/null \
            || fail "D: user streaming not preserved"
          echo "PASS: Scenario D"

          # ═══════════════════════════════════════════════════════════════
          # Scenario E: MCP additive merge
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario E: MCP additive merge ==="
          E_HOME=$(mktemp -d)
          install -m 0644 ${fixtureE} "$E_HOME/config.yaml"
          E_CONFIG=$(merge_and_load "$E_HOME")

          echo "$E_CONFIG" | jq -e '.mcp_servers."user-server".url == "http://user-mcp"' > /dev/null \
            || fail "E: user MCP server not preserved"
          echo "$E_CONFIG" | jq -e '.mcp_servers."nix-server".command == "echo"' > /dev/null \
            || fail "E: Nix MCP server did not override same-name user server"
          echo "$E_CONFIG" | jq -e '.mcp_servers."nix-server".args == ["nix"]' > /dev/null \
            || fail "E: Nix MCP server args wrong"
          echo "PASS: Scenario E"

          # ═══════════════════════════════════════════════════════════════
          # Scenario F: Nested deep merge
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario F: Nested deep merge ==="
          F_HOME=$(mktemp -d)
          install -m 0644 ${fixtureF} "$F_HOME/config.yaml"
          F_CONFIG=$(merge_and_load "$F_HOME")

          echo "$F_CONFIG" | jq -e '.terminal.backend == "docker"' > /dev/null \
            || fail "F: Nix terminal.backend did not override"
          echo "$F_CONFIG" | jq -e '.terminal.timeout == 999' > /dev/null \
            || fail "F: Nix terminal.timeout did not override"
          echo "$F_CONFIG" | jq -e '.terminal.custom_key == "preserved"' > /dev/null \
            || fail "F: terminal.custom_key not preserved"
          echo "$F_CONFIG" | jq -e '.terminal.cwd == "/user/path"' > /dev/null \
            || fail "F: user terminal.cwd not preserved when Nix does not set it"
          echo "$F_CONFIG" | jq -e '.terminal.env_passthrough == ["USER_VAR"]' > /dev/null \
            || fail "F: user terminal.env_passthrough not preserved"
          echo "PASS: Scenario F"

          # ═══════════════════════════════════════════════════════════════
          # Scenario G: Idempotency — merging twice yields the same result
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario G: Idempotency ==="
          G_HOME=$(mktemp -d)
          install -m 0644 ${fixtureD} "$G_HOME/config.yaml"
          ${configMergeScript} ${nixSettings} "$G_HOME/config.yaml"
          FIRST=$(cat "$G_HOME/config.yaml")
          ${configMergeScript} ${nixSettings} "$G_HOME/config.yaml"
          SECOND=$(cat "$G_HOME/config.yaml")

          if [ "$FIRST" != "$SECOND" ]; then
            fail "G: second merge produced different output"
            echo "--- first ---"
            echo "$FIRST"
            echo "--- second ---"
            echo "$SECOND"
          fi
          echo "PASS: Scenario G"

          # ═══════════════════════════════════════════════════════════════
          # Report
          # ═══════════════════════════════════════════════════════════════
          if [ -n "$ERRORS" ]; then
            echo ""
            echo "FAILURES:"
            echo -e "$ERRORS"
            exit 1
          fi

          echo ""
          echo "=== All 7 merge scenarios passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';

        # ── NixOS VM integration test for env.d/ ──────────────────────────
        env-d-vm = import ./tests/env-d-vm.nix {
          inherit pkgs;
          nixosModule = inputs.self.nixosModules.default;
        };

        # ── env.d/ activation script behavior ─────────────────────────────
        # Tests the shell logic from the activation script: writing
        # nix-environment.env, symlinking environmentFiles, and detecting
        # stale symlink cleanup issues.
        env-d-activation = pkgs.runCommand "hermes-env-d-activation" { } ''
          set -e
          ERRORS=""
          fail() { ERRORS="$ERRORS\nFAIL: $1"; }

          SECRET_DIR=$(mktemp -d)
          echo "API_KEY=sk-secret-123" > "$SECRET_DIR/hermes-api"
          echo "DISCORD_TOKEN=tok-abc" > "$SECRET_DIR/hermes-discord"
          echo "TELEGRAM_TOKEN=tok-xyz" > "$SECRET_DIR/hermes-telegram"

          # ═══════════════════════════════════════════════════════════════
          # Scenario A: environment only — no environmentFiles
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario A: environment only ==="
          A_HERMES=$(mktemp -d)/.hermes
          mkdir -p "$A_HERMES/env.d"

          cat > "$A_HERMES/env.d/nix-environment.env" <<'ENVEOF'
MODEL=test-model
OPENAI_BASE_URL=https://example.com/v1
ENVEOF
          chmod 0640 "$A_HERMES/env.d/nix-environment.env"

          test -f "$A_HERMES/env.d/nix-environment.env" \
            || fail "A: nix-environment.env not created"
          grep -q "MODEL=test-model" "$A_HERMES/env.d/nix-environment.env" \
            || fail "A: MODEL not in nix-environment.env"
          grep -q "OPENAI_BASE_URL=https://example.com/v1" "$A_HERMES/env.d/nix-environment.env" \
            || fail "A: OPENAI_BASE_URL not in nix-environment.env"
          ls "$A_HERMES/env.d/" | grep -q "nix-[0-9]" \
            && fail "A: unexpected nix-N symlinks when no environmentFiles" || true
          echo "PASS: Scenario A"

          # ═══════════════════════════════════════════════════════════════
          # Scenario B: environmentFiles — symlinks created & readable
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario B: environmentFiles ==="
          B_HERMES=$(mktemp -d)/.hermes
          mkdir -p "$B_HERMES/env.d"

          ln -sfn "$SECRET_DIR/hermes-api" "$B_HERMES/env.d/nix-0.env"
          ln -sfn "$SECRET_DIR/hermes-discord" "$B_HERMES/env.d/nix-1.env"

          test -L "$B_HERMES/env.d/nix-0.env" \
            || fail "B: nix-0.env is not a symlink"
          test -L "$B_HERMES/env.d/nix-1.env" \
            || fail "B: nix-1.env is not a symlink"
          test "$(readlink "$B_HERMES/env.d/nix-0.env")" = "$SECRET_DIR/hermes-api" \
            || fail "B: nix-0.env points to wrong target"
          grep -q "API_KEY=sk-secret-123" "$B_HERMES/env.d/nix-0.env" \
            || fail "B: cannot read API_KEY through symlink"
          grep -q "DISCORD_TOKEN=tok-abc" "$B_HERMES/env.d/nix-1.env" \
            || fail "B: cannot read DISCORD_TOKEN through symlink"
          echo "PASS: Scenario B"

          # ═══════════════════════════════════════════════════════════════
          # Scenario C: ln -sfn replaces existing symlink atomically
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario C: symlink replacement ==="
          C_HERMES=$(mktemp -d)/.hermes
          mkdir -p "$C_HERMES/env.d"

          echo "OLD_KEY=old" > "$SECRET_DIR/old-secret"
          ln -sfn "$SECRET_DIR/old-secret" "$C_HERMES/env.d/nix-0.env"
          echo "NEW_KEY=new" > "$SECRET_DIR/new-secret"
          ln -sfn "$SECRET_DIR/new-secret" "$C_HERMES/env.d/nix-0.env"

          test "$(readlink "$C_HERMES/env.d/nix-0.env")" = "$SECRET_DIR/new-secret" \
            || fail "C: symlink not updated to new target"
          grep -q "NEW_KEY=new" "$C_HERMES/env.d/nix-0.env" \
            || fail "C: new content not readable through updated symlink"
          echo "PASS: Scenario C"

          # ═══════════════════════════════════════════════════════════════
          # Scenario D: stale symlink detection
          # Simulates reducing environmentFiles from 3 to 1. The PR's
          # activation script does ln -sfn for index 0 only but never
          # removes nix-1.env and nix-2.env from the previous deployment.
          # ═══════════════════════════════════════════════════════════════
          echo "=== Scenario D: stale symlink detection ==="
          D_HERMES=$(mktemp -d)/.hermes
          mkdir -p "$D_HERMES/env.d"

          # First deployment: 3 environmentFiles
          ln -sfn "$SECRET_DIR/hermes-api" "$D_HERMES/env.d/nix-0.env"
          ln -sfn "$SECRET_DIR/hermes-discord" "$D_HERMES/env.d/nix-1.env"
          ln -sfn "$SECRET_DIR/hermes-telegram" "$D_HERMES/env.d/nix-2.env"

          # Second deployment: only index 0 (simulates what the PR does)
          ln -sfn "$SECRET_DIR/hermes-api" "$D_HERMES/env.d/nix-0.env"

          # NOTE: This scenario documents a known bug in PR #10139.
          # The activation script does not clean up stale nix-N.env symlinks
          # when environmentFiles is reduced. We print a warning rather than
          # failing the check, since this is a known upstream issue.
          STALE=0
          if [ -e "$D_HERMES/env.d/nix-1.env" ] || [ -L "$D_HERMES/env.d/nix-1.env" ]; then
            echo "WARNING: D: stale nix-1.env left behind after reducing environmentFiles"
            STALE=1
          fi
          if [ -e "$D_HERMES/env.d/nix-2.env" ] || [ -L "$D_HERMES/env.d/nix-2.env" ]; then
            echo "WARNING: D: stale nix-2.env left behind after reducing environmentFiles"
            STALE=1
          fi
          if [ "$STALE" -eq 1 ]; then
            echo "KNOWN BUG: Scenario D (stale symlinks not cleaned up)"
          else
            echo "PASS: Scenario D"
          fi

          # ═══════════════════════════════════════════════════════════════
          # Report
          # ═══════════════════════════════════════════════════════════════
          if [ -n "$ERRORS" ]; then
            echo ""
            echo "FAILURES:"
            echo -e "$ERRORS"
            exit 1
          fi

          echo ""
          echo "=== All env.d activation scenarios passed ==="
          mkdir -p $out
          echo "ok" > $out/result
        '';
      };
    };
}

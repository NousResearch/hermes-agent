# nix/checks.nix — Build-time verification tests
#
# Checks are Linux-only: the full Python venv (via uv2nix) includes
# transitive deps like onnxruntime that lack compatible wheels on
# aarch64-darwin. The package and devShell still work on macOS.
{ inputs, ... }: {
  perSystem = { pkgs, system, lib, ... }:
    let
      hermes-agent = inputs.self.packages.${system}.default;
      hermesVenv = pkgs.callPackage ./python.nix {
        inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
      };

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
            builtins.throw "Package fails to evaluate on:\n${failMsg}"
          else ''
            echo "PASS: package evaluates on all ${toString (builtins.length targetSystems)} platforms"
            mkdir -p $out
            echo "ok" > $out/result
          ''
        );
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

        # Verify bundled TUI is present and compiled
        bundled-tui = pkgs.runCommand "hermes-bundled-tui" { } ''
          set -e
          echo "=== Checking bundled TUI ==="
          test -d ${hermes-agent}/ui-tui || (echo "FAIL: ui-tui directory missing"; exit 1)
          echo "PASS: ui-tui directory exists"

          test -f ${hermes-agent}/ui-tui/dist/entry.js || (echo "FAIL: compiled entry.js missing"; exit 1)
          echo "PASS: compiled entry.js present"

          test -d ${hermes-agent}/ui-tui/node_modules || (echo "FAIL: node_modules missing"; exit 1)
          echo "PASS: node_modules present"

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
      };
    };
}

# nix/checks.nix — Build-time verification tests
{ inputs, ... }: {
  perSystem = { pkgs, system, ... }:
    let
      hermes-agent = inputs.self.packages.${system}.default;
    in {
      checks = {
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
          # These must match [project.scripts] in pyproject.toml
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
      };
    };
}

# nix/devShell.nix — Dev shell that delegates setup to each package
#
# Each npm workspace package exposes passthru.packageJsonPath (e.g.
# "ui-tui/package.json").  This file collects them all and passes the
# list to mkNpmDevShellHook, which stamps all package.jsons at once,
# then runs a single `npm i --package-lock-only` if any changed and
# `npm ci` if the lockfile changed.
{ ... }:
{
  perSystem =
    { pkgs, self', ... }:
    let
      packages = builtins.attrValues self'.packages;
      hermesNpmLib = self'.packages.default.passthru.hermesNpmLib;

      # Collect all packageJsonPath values from npm workspace packages.
      npmPackageJsonPaths = builtins.filter (p: p != null) (
        map (p: p.passthru.packageJsonPath or null) packages
      );

      # Non-npm packages may have their own devShellHook (e.g. hermes-agent
      # stamps pyproject.toml + uv.lock for Python venv setup).
      nonNpmHooks = map (p: p.passthru.devShellHook or "") packages;
      combinedNonNpm = pkgs.lib.concatStringsSep "\n" (builtins.filter (h: h != "") nonNpmHooks);
    in
    {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          (pkgs.runCommand "hermes" { } ''
            mkdir -p $out/bin
            install -Dm755 ${../hermes} $out/bin/hermes
          '')
          (pkgs.runCommand "dev-sandbox" { } ''
            mkdir -p $out/bin
            install -Dm755 ${../scripts/dev-sandbox.sh} $out/bin/sandbox
          '')
          uv
          # Headless Wayland compositor for E2E tests (test:e2e:visual).
          # cage renders a single client with no window management, so
          # the Electron window opens at a fixed size without tiling.
          # libglvnd provides libEGL.so.1 that cage needs on NixOS.
          cage
          libglvnd
        ]
        ++ self'.packages.default.passthru.devDeps;
        shellHook = ''
          ${combinedNonNpm}
          ${hermesNpmLib.mkNpmDevShellHook npmPackageJsonPaths}

          # Force Node to use Nix's playwright-test binary instead of node_modules/.bin
          export PATH="${pkgs.playwright-test}/bin:$PATH"

          # for the devshell to pick up the src
          export HERMES_PYTHON_SRC_ROOT=$(git rev-parse --show-toplevel)
          echo "Hermes Agent dev shell in $HERMES_PYTHON_SRC_ROOT"
          echo "Ready. Run 'hermes' or 'sandbox hermes' to start."
        '';
      };
    };
}

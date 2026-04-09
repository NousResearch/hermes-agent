# nix/tui.nix — Hermes TUI (Ink/React) compiled with tsc and bundled
{ pkgs, ... }:
let
  src = ../ui-tui;
  npmDeps = pkgs.fetchNpmDeps {
    inherit src;
    hash = "sha256-iz6TrWec4MpfDLZR48V6XHoKnZkEn9x2t97YOqWZt5k=";
  };

  packageJson = builtins.fromJSON (builtins.readFile (src + "/package.json"));
  version = packageJson.version;

  npmLockHash = builtins.hashString "sha256" (builtins.readFile ../ui-tui/package-lock.json);
in
pkgs.buildNpmPackage {
  pname = "hermes-tui";
  inherit src npmDeps version;

  doCheck = false;

  installPhase = ''
    runHook preInstall

    mkdir -p $out/lib/hermes-tui

    cp -r dist $out/lib/hermes-tui/dist

    # runtime node_modules
    cp -r node_modules $out/lib/hermes-tui/node_modules

    # package.json needed for "type": "module" resolution
    cp package.json $out/lib/hermes-tui/

    runHook postInstall
  '';

  passthru.devShellHook = ''
    STAMP=".nix-stamps/hermes-tui"
    STAMP_VALUE="${npmLockHash}"
    if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$STAMP_VALUE" ]; then
      echo "hermes-tui: installing npm dependencies..."
      cd ui-tui && npm install --silent --no-fund --no-audit 2>/dev/null && cd ..
      mkdir -p .nix-stamps
      echo "$STAMP_VALUE" > "$STAMP"
    fi
  '';
}

# nix/tui.nix — Hermes TUI (Ink/React) compiled with tsc and bundled
{ pkgs, npm-lockfile-fix, ... }:
let
  src = ../ui-tui;
  npmDeps = pkgs.fetchNpmDeps {
    inherit src;
    hash = "sha256-zsUPmbC6oMUO10EhS3ptvDjwlfpCSEmrkjyeORw7fac=";
  };

  packageJson = builtins.fromJSON (builtins.readFile (src + "/package.json"));
  version = packageJson.version;

  npmLockHash = builtins.hashString "sha256" (builtins.readFile ../ui-tui/package-lock.json);
in
pkgs.buildNpmPackage {
  pname = "hermes-tui";
  inherit src npmDeps version;

  doCheck = false;

  postPatch = ''
    # fetchNpmDeps strips the trailing newline; match it so the diff passes
    sed -i -z 's/\n$//' package-lock.json
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/lib/hermes-tui

    cp -r dist $out/lib/hermes-tui/dist

    # runtime node_modules
    cp -r node_modules $out/lib/hermes-tui/node_modules

    # @hermes/ink is a file: dependency, we need to copy it in fr
    rm -f $out/lib/hermes-tui/node_modules/@hermes/ink
    cp -r packages/hermes-ink $out/lib/hermes-tui/node_modules/@hermes/ink

    # package.json needed for "type": "module" resolution
    cp package.json $out/lib/hermes-tui/

    runHook postInstall
  '';

  nativeBuildInputs = [
    (pkgs.writeShellScriptBin "update_tui_lockfile" ''
      set -euox pipefail

      # get root of repo
      REPO_ROOT=$(git rev-parse --show-toplevel)

      # cd into ui-tui and reinstall
      cd "$REPO_ROOT/ui-tui"
      rm -rf node_modules/
      npm cache clean --force
      CI=true npm install # ci env var to suppress annoying unicode install banner lag
      ${pkgs.lib.getExe npm-lockfile-fix} ./package-lock.json

      NIX_FILE="$REPO_ROOT/nix/tui.nix"
      # compute the new hash
      sed -i "s/hash = \"[^\"]*\";/hash = \"\";/" $NIX_FILE
      NIX_OUTPUT=$(nix build .#tui 2>&1 || true)
      NEW_HASH=$(echo "$NIX_OUTPUT" | grep 'got:' | awk '{print $2}') 
      echo got new hash $NEW_HASH
      sed -i "s|hash = \"[^\"]*\";|hash = \"$NEW_HASH\";|" $NIX_FILE
      nix build .#tui
      echo "Updated npm hash in $NIX_FILE to $NEW_HASH"
    '')
  ];

  passthru.devShellHook = ''
    STAMP=".nix-stamps/hermes-tui"
    STAMP_VALUE="${npmLockHash}"
    if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$STAMP_VALUE" ]; then
      echo "hermes-tui: installing npm dependencies..."
      cd ui-tui && CI=true npm install --silent --no-fund --no-audit 2>/dev/null && cd ..
      mkdir -p .nix-stamps
      echo "$STAMP_VALUE" > "$STAMP"
    fi
  '';
}

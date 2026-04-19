# nix/web.nix — Hermes Web Dashboard (Vite/React) frontend build
{ pkgs, npm-lockfile-fix, ... }:
let
  src = ../web;
  npmDeps = pkgs.fetchNpmDeps {
    inherit src;
    hash = "sha256-Y0pOzdFG8BLjfvCLmsvqYpjxFjAQabXp1i7X9W/cCU4=";
  };

  npmLockHash = builtins.hashString "sha256" (builtins.readFile ../web/package-lock.json);
in
pkgs.buildNpmPackage {
  pname = "hermes-web";
  version = "0.0.0";
  inherit src npmDeps;

  doCheck = false;

  buildPhase = ''
    npx tsc -b
    npx vite build --outDir dist
  '';

  installPhase = ''
    runHook preInstall
    cp -r dist $out
    runHook postInstall
  '';

  nativeBuildInputs = [
    (pkgs.writeShellScriptBin "update_web_lockfile" ''
      set -euox pipefail

      REPO_ROOT=$(git rev-parse --show-toplevel)

      cd "$REPO_ROOT/web"
      rm -rf node_modules/
      npm cache clean --force
      CI=true npm install
      ${pkgs.lib.getExe npm-lockfile-fix} ./package-lock.json

      NIX_FILE="$REPO_ROOT/nix/web.nix"
      sed -i "s/hash = \"[^\"]*\";/hash = \"\";/" $NIX_FILE
      NIX_OUTPUT=$(nix build .#web 2>&1 || true)
      NEW_HASH=$(echo "$NIX_OUTPUT" | grep 'got:' | awk '{print $2}')
      echo got new hash $NEW_HASH
      sed -i "s|hash = \"[^\"]*\";|hash = \"$NEW_HASH\";|" $NIX_FILE
      nix build .#web
      echo "Updated npm hash in $NIX_FILE to $NEW_HASH"
    '')
  ];

  passthru.devShellHook = ''
    STAMP=".nix-stamps/hermes-web"
    STAMP_VALUE="${npmLockHash}"
    if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$STAMP_VALUE" ]; then
      echo "hermes-web: installing npm dependencies..."
      cd web && CI=true npm install --silent --no-fund --no-audit 2>/dev/null && cd ..
      mkdir -p .nix-stamps
      echo "$STAMP_VALUE" > "$STAMP"
    fi
  '';
}

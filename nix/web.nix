# nix/web.nix — Hermes Web Dashboard (Vite/React) frontend build
{ pkgs, hermesNpmLib, ... }:
let
  # @hermes/shared ships as a file: workspace dep of web, so its source
  # must be in the filtered src tree too.
  npm = hermesNpmLib.mkNpmPassthru {
    dirs = [
      "web"
      "apps/shared"
    ];
  };

  packageJson = builtins.fromJSON (builtins.readFile (npm.src + "/web/package.json"));
  version = packageJson.version;
in
pkgs.buildNpmPackage (npm // {
  pname = "hermes-web";
  inherit version;

  doCheck = false;

  buildPhase = ''
    # Build from web/ so vite.config.ts and tsconfig resolve correctly.
    # The workspace root's node_modules/ is at ../node_modules/.
    cd web
    node ../node_modules/typescript/bin/tsc -b
    # outDir in vite.config.ts points to ../hermes_cli/web_dist for the
    # monorepo layout.  Override with --outDir dist for the nix build.
    node ../node_modules/vite/bin/vite.js build --outDir dist

    # Return to source root so installPhase paths are correct.
    cd ..
  '';

  installPhase = ''
    runHook preInstall
    # vite writes to web/dist/ (we cd'd there, overrode outDir, then cd'd back).
    cp -r web/dist $out
    runHook postInstall
  '';
})

# nix/tui.nix — Hermes TUI (Ink/React) compiled with tsc and bundled
{ pkgs, hermesNpmLib, ... }:
let
  src = ../ui-tui;
  npmDeps = pkgs.fetchNpmDeps {
    inherit src;
    hash = "sha256-Chz+NW9NXqboXHOa6PKwf5bhAkkcFtKNhvKWwg2XSPc=";
  };

  npm = hermesNpmLib.mkNpmPassthru { folder = "ui-tui"; attr = "tui"; pname = "hermes-tui"; };

  packageJson = builtins.fromJSON (builtins.readFile (src + "/package.json"));
  version = packageJson.version;
in
pkgs.buildNpmPackage (npm // {
  pname = "hermes-tui";
  inherit src npmDeps version;

  doCheck = false;
  npmFlags = [ "--legacy-peer-deps" ];

  installPhase = ''
    runHook preInstall

    mkdir -p $out/lib/hermes-tui

    # Single self-contained bundle built by scripts/build.mjs (esbuild).
    # No runtime node_modules needed.
    cp -r dist $out/lib/hermes-tui/dist

    # package.json kept for "type": "module" resolution on `node dist/entry.js`.
    cp package.json $out/lib/hermes-tui/

    runHook postInstall
  '';
})

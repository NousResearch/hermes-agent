# nix/tui.nix — Hermes TUI (Ink/React) compiled with tsc and bundled
{ pkgs, hermesNpmLib, ... }:
let
  src = ../ui-tui;
  npmDeps = pkgs.fetchNpmDeps {
    inherit src;
    hash = "sha256-RU4qSHgJPMyfRSEJDzkG4+MReDZDc6QbTD2wisa5QE0=";
  };

  npm = hermesNpmLib.mkNpmPassthru { folder = "ui-tui"; attr = "tui"; pname = "hermes-tui"; };

  packageJson = builtins.fromJSON (builtins.readFile (src + "/package.json"));
  version = packageJson.version;
in
pkgs.buildNpmPackage (npm // {
  pname = "hermes-tui";
  inherit src npmDeps version;

  doCheck = false;

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
})

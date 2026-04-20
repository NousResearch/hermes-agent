# nix/tui.nix — Hermes TUI (Ink/React) compiled with tsc and bundled
{ pkgs, hermesNpmLib, ... }:
let
  src = ../ui-tui;
  npmDeps = pkgs.fetchNpmDeps {
    inherit src;
    hash = "sha256-BlxkTyn1x7ZQcj7pcMB5y5C2AyToT/CzxmtacTfEXmY=";
  };

  packageJson = builtins.fromJSON (builtins.readFile (src + "/package.json"));
  version = packageJson.version;
in
pkgs.buildNpmPackage {
  pname = "hermes-tui";
  inherit src npmDeps version;

  doCheck = false;

  patchPhase = ''
    runHook prePatch
    sed -i -z 's/\n$//' package-lock.json
    runHook postPatch
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
    (hermesNpmLib.mkUpdateLockfileScript {
      name = "update_tui_lockfile";
      folder = "ui-tui";
      nixFile = "nix/tui.nix";
      attr = "tui";
    })
  ];

  passthru = {
    devShellHook = hermesNpmLib.mkNpmDevShellHook {
      name = "hermes-tui";
      folder = "ui-tui";
    };
    npmLockfile = {
      attr = "tui";
      folder = "ui-tui";
      nixFile = "nix/tui.nix";
    };
  };
}

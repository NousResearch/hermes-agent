# Self-contained Hermes TUI, compiled by the same recipe as npm.
{ hermesNpmLib, ... }:
hermesNpmLib.buildNpmPackage {
  dirs = [
    "ui-tui"
    "apps/shared"
    "scripts/build/tui.mjs"
    "scripts/build/frontend-common.mjs"
  ];

  doCheck = false;

  buildPhase = ''
    runHook preBuild
    node scripts/build/tui.mjs --source "$PWD" --out "$TMPDIR/tui-product"
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/lib/hermes-tui
    cp -r "$TMPDIR/tui-product/." $out/lib/hermes-tui/
    runHook postInstall
  '';
}

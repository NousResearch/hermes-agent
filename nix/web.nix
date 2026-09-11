# Dashboard compilation consumes prepared icons; no npm lifecycle preparation.
{ hermesNpmLib, generatedIcons, ... }:
hermesNpmLib.buildNpmPackage {
  dirs = [
    "web"
    "apps/shared"
    "scripts/build/web.mjs"
    "scripts/build/frontend-common.mjs"
  ];

  doCheck = false;

  buildPhase = ''
    runHook preBuild
    node scripts/build/web.mjs --source "$PWD" --icons ${generatedIcons} --out "$TMPDIR/web-product"
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out
    cp -r "$TMPDIR/web-product/." $out/
    runHook postInstall
  '';
}

# Dashboard compilation consumes prepared icons; no npm lifecycle preparation.
{ hermesNpmLib, generatedIcons, lib, rev ? null, branch ? null, dirty ? false, ... }:
hermesNpmLib.buildNpmPackage {
  dirs = [
    "web"
    "apps/shared"
    "scripts/build/web.mjs"
    "scripts/build/freshness.mjs"
    "scripts/build/frontend-common.mjs"
  ];

  doCheck = false;

  buildPhase = ''
    runHook preBuild
    ${lib.optionalString (rev != null) ''
      # Filtered sources have no .git; preserve the flake's actual dirty state.
      export HERMES_REVISION=${lib.escapeShellArg rev}
      export BUILD_SOURCE_BRANCH=${lib.escapeShellArg (if branch != null then branch else "")}
      export BUILD_SOURCE_DIRTY=${if dirty then "true" else "false"}
    ''}
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

# The Webapp: the Desktop renderer built for a browser. The npm hook has
# already installed the locked workspace, so it compiles there, with no
# private workspace and no Electron.
{ hermesNpmLib, ... }:
hermesNpmLib.buildNpmPackage {
  dirs = [
    "apps/desktop"
    "apps/shared"
    "scripts/build/webapp.mjs"
    "scripts/build/node-deps.mjs"
    "scripts/build/freshness.mjs"
    "scripts/build/frontend-common.mjs"
  ];
  pname = "hermes-webapp";

  # The browser bundle loads no native module (node-pty belongs to Desktop),
  # so none is compiled.
  npmRebuildFlags = [ "--ignore-scripts" ];

  doCheck = false;

  buildPhase = ''
    runHook preBuild
    node --input-type=module -e '
      import { buildWebapp } from "./scripts/build/webapp.mjs"
      await buildWebapp({ source: process.cwd(), workspace: process.cwd(), out: `''${process.env.TMPDIR}/webapp-product` })
    '
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out
    cp -r "$TMPDIR/webapp-product/." $out/
    runHook postInstall
  '';
}

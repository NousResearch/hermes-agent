# The Webapp: the Desktop renderer built for a browser. pnpmConfigHook has
# already installed the locked workspace (without lifecycle scripts, so no
# native module and no Electron binary), so it compiles there, with no
# private workspace.
{ hermesNpmLib, ... }:
hermesNpmLib.buildPnpmPackage {
  dirs = [
    "apps/desktop"
    "apps/shared"
    "scripts/build/webapp.mjs"
    "scripts/build/node-deps.mjs"
    "scripts/build/freshness.mjs"
    "scripts/build/frontend-common.mjs"
  ];
  pname = "hermes-webapp";

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

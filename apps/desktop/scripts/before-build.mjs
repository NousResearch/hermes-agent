/**
 * Desktop bundles ship precompiled renderer assets. Returning false here tells
 * electron-builder to skip the node_modules collector/install step, which
 * avoids workspace dependency graph explosions and keeps packaging
 * deterministic across environments. Windows ships its own Python runtime and backend in extraResources.
 * Stage it with stage-windows-runtime.mjs before packaging.
 */
export default async function beforeBuild() {
  return false
}

/**
 * Desktop bundles ship precompiled renderer assets. Returning false here tells
 * electron-builder to skip the node_modules collector/install step, which
 * avoids workspace dependency graph explosions and keeps packaging
 * deterministic across environments. Reuben Desktop is a remote frontend, so
 * no Hermes/Reuben backend payload is bundled or fetched by this app.
 */
module.exports = async function beforeBuild() {
  return false
}

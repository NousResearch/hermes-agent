"use strict"

/**
 * Stages the canonical repo-root installers (scripts/install.sh +
 * scripts/install.ps1) into apps/desktop/build/ so electron-builder can ship
 * them inside the packaged app via the `extraResources` entries
 * ({ from: build/install.sh, to: install.sh } and likewise install.ps1) ->
 * process.resourcesPath/install.{sh,ps1}.
 *
 * Why bundle them at all:
 *   electron/bootstrap-runner.cjs otherwise fetches the installer from
 *   raw.githubusercontent.com/NousResearch/hermes-agent/<commit> at first
 *   launch. That host is blocked in mainland China, so a fresh ApexNodes
 *   install there dies before the first stage runs. Shipping the installer
 *   inside the app (bundledInstallScript) removes that network round-trip
 *   entirely — the China mirror logic lives in the installers themselves (the
 *   "ApexNodes China mirror mode" blocks), gated by HERMES_CN_MIRRORS.
 *
 * We deliberately copy the ONE canonical installer per platform rather than
 * maintaining separate China forks: the bundled copies stay byte-identical to
 * the tested install.sh / install.ps1, and their CN behavior is a no-op unless
 * the desktop turns it on. Mirrors write-build-stamp.cjs / stage-native-deps.cjs
 * (run from the `build` npm script before electron-builder packs).
 */

const fs = require("fs")
const path = require("path")

const DESKTOP_ROOT = path.resolve(__dirname, "..")
const REPO_ROOT = path.resolve(DESKTOP_ROOT, "..", "..")
const OUT_DIR = path.join(DESKTOP_ROOT, "build")

// install.sh (posix) + install.ps1 (Windows). Both ship so a packaged build can
// bootstrap offline on either platform — bootstrap-runner.cjs picks one by OS
// (installScriptName) and prefers the bundled copy over the GitHub fetch.
const INSTALLERS = ["install.sh", "install.ps1"]

function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true })
  for (const name of INSTALLERS) {
    const src = path.join(REPO_ROOT, "scripts", name)
    const out = path.join(OUT_DIR, name)
    if (!fs.existsSync(src)) {
      console.error(
        "[stage-install-script] ERROR: installer not found at " +
          src +
          "\n  The desktop bootstrap ships this file inside the app; a packaged" +
          "\n  build without it cannot install on a network-restricted machine."
      )
      process.exit(1)
    }

    fs.copyFileSync(src, out)
    // Executable bit for tidiness; bootstrap-runner spawns install.sh via
    // `bash <path>` and install.ps1 via PowerShell -File, so the mode is not
    // strictly required, but keeps the staged copy faithful.
    fs.chmodSync(out, 0o755)

    const bytes = fs.statSync(out).size
    console.log(
      "[stage-install-script] staged " +
        path.relative(REPO_ROOT, src) +
        " -> " +
        path.relative(REPO_ROOT, out) +
        " (" +
        bytes +
        " bytes)"
    )
  }
}

main()

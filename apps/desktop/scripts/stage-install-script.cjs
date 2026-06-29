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

// ApexNodes overlay seam libs (scripts/lib/apexnodes-region-detect.{sh,ps1}).
// install.sh / install.ps1 source these from a `lib/` dir RELATIVE to their own
// on-disk location (see the "ApexNodes overlay seam" block in each installer).
// The bundled installer runs from process.resourcesPath/install.{sh,ps1}, so the
// lib MUST sit at process.resourcesPath/lib/ for the China mirror / COS source
// downgrade to activate. We stage it into build/lib/ and map it through
// electron-builder's extraResources ({ from: build/lib, to: lib }). Without this
// the installer still runs, but a mainland-China desktop would lose its mirror
// downgrade and fall back to (blocked) github.com / pypi.org / npmjs.org.
const SEAM_LIBS = ["apexnodes-region-detect.sh", "apexnodes-region-detect.ps1"]

function stageFile(srcAbs, outAbs, label, mode) {
  if (!fs.existsSync(srcAbs)) {
    console.error(
      "[stage-install-script] ERROR: " +
        label +
        " not found at " +
        srcAbs +
        "\n  The desktop bootstrap ships this file inside the app; a packaged" +
        "\n  build without it cannot install on a network-restricted machine."
    )
    process.exit(1)
  }
  fs.mkdirSync(path.dirname(outAbs), { recursive: true })
  fs.copyFileSync(srcAbs, outAbs)
  // Executable bit for tidiness; bootstrap-runner spawns install.sh via
  // `bash <path>` and install.ps1 via PowerShell -File, so the mode is not
  // strictly required, but keeps the staged copy faithful.
  fs.chmodSync(outAbs, mode)
  const bytes = fs.statSync(outAbs).size
  console.log(
    "[stage-install-script] staged " +
      path.relative(REPO_ROOT, srcAbs) +
      " -> " +
      path.relative(REPO_ROOT, outAbs) +
      " (" +
      bytes +
      " bytes)"
  )
}

function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true })
  for (const name of INSTALLERS) {
    stageFile(path.join(REPO_ROOT, "scripts", name), path.join(OUT_DIR, name), "installer", 0o755)
  }
  for (const name of SEAM_LIBS) {
    stageFile(
      path.join(REPO_ROOT, "scripts", "lib", name),
      path.join(OUT_DIR, "lib", name),
      "overlay seam lib",
      0o644
    )
  }
}

main()

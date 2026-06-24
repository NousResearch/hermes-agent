"use strict"

/**
 * Stages the canonical repo-root installer (scripts/install.sh) into
 * apps/desktop/build/install.sh so electron-builder can ship it inside the
 * packaged app via the `extraResources` entry ({ from: build/install.sh, to:
 * install.sh } -> process.resourcesPath/install.sh).
 *
 * Why bundle it at all:
 *   electron/bootstrap-runner.cjs otherwise fetches install.sh from
 *   raw.githubusercontent.com/NousResearch/hermes-agent/<commit> at first
 *   launch. That host is blocked in mainland China, so a fresh ApexNodes
 *   install there dies before the first stage runs. Shipping the installer
 *   inside the app (bundledInstallScript) removes that network round-trip
 *   entirely — the China mirror logic lives in install.sh itself (its
 *   "ApexNodes China mirror mode" block), gated by HERMES_CN_MIRRORS.
 *
 * We deliberately copy the ONE canonical installer rather than maintaining a
 * separate China fork of it: the bundled copy stays byte-identical to the
 * tested install.sh, and its CN behavior is a no-op unless the desktop turns
 * it on. Mirrors write-build-stamp.cjs / stage-native-deps.cjs (run from the
 * `build` npm script before electron-builder packs).
 *
 * install.ps1 (Windows) is intentionally NOT staged yet — Windows + the CN
 * source/mirror path are a V0.2 concern; Windows packaged builds keep fetching
 * install.ps1 from GitHub (bootstrap-runner falls through when no bundled copy
 * is present).
 */

const fs = require("fs")
const path = require("path")

const DESKTOP_ROOT = path.resolve(__dirname, "..")
const REPO_ROOT = path.resolve(DESKTOP_ROOT, "..", "..")
const SRC = path.join(REPO_ROOT, "scripts", "install.sh")
const OUT_DIR = path.join(DESKTOP_ROOT, "build")
const OUT_FILE = path.join(OUT_DIR, "install.sh")

function main() {
  if (!fs.existsSync(SRC)) {
    console.error(
      "[stage-install-script] ERROR: installer not found at " +
        SRC +
        "\n  The desktop bootstrap ships this file inside the app; a packaged" +
        "\n  build without it cannot install on a network-restricted machine."
    )
    process.exit(1)
  }

  fs.mkdirSync(OUT_DIR, { recursive: true })
  fs.copyFileSync(SRC, OUT_FILE)
  // Executable bit for tidiness; bootstrap-runner spawns it via `bash <path>`,
  // so this is not strictly required, but keeps the staged copy faithful.
  fs.chmodSync(OUT_FILE, 0o755)

  const bytes = fs.statSync(OUT_FILE).size
  console.log(
    "[stage-install-script] staged " +
      path.relative(REPO_ROOT, SRC) +
      " -> " +
      path.relative(REPO_ROOT, OUT_FILE) +
      " (" +
      bytes +
      " bytes)"
  )
}

main()

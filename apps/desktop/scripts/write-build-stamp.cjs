"use strict"

/**
 * Writes apps/desktop/build/install-stamp.json with the git ref the desktop
 * .exe should pin to at first-launch bootstrap time.  This file ships inside
 * the packaged app via electron-builder's extraResources entry and is read
 * by electron/main.cjs to drive the install.ps1 stage bootstrap flow.
 *
 * Schema (subject to bump via STAMP_SCHEMA_VERSION):
 *   {
 *     "schemaVersion": 1,
 *     "commit":        "<40-char SHA>",
 *     "branch":        "<branch name>",
 *     "builtAt":       "<ISO 8601 UTC timestamp>",
 *     "dirty":         true|false,
 *     "source":        "ci" | "local"
 *   }
 *
 * Source preference order:
 *   1. CI env vars ($GITHUB_SHA / $GITHUB_REF_NAME) -- avoid edge cases with
 *      shallow clones, detached HEADs, etc. in CI.
 *   2. Local `git rev-parse` against the parent repo (../..).
 *
 * Dev / out-of-repo builds without git produce an explicit error rather than
 * silently writing an unstamped manifest -- the packaged app refuses to
 * bootstrap without a stamp.
 */

const fs = require("fs")
const path = require("path")
const { execSync } = require("child_process")

const STAMP_SCHEMA_VERSION = 1

const DESKTOP_ROOT = path.resolve(__dirname, "..")
const REPO_ROOT = path.resolve(DESKTOP_ROOT, "..", "..")
const OUT_DIR = path.join(DESKTOP_ROOT, "build")
const OUT_FILE = path.join(OUT_DIR, "install-stamp.json")

function tryExec(cmd, opts) {
  try {
    return execSync(cmd, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], ...opts }).trim()
  } catch {
    return null
  }
}

function fromCI() {
  const sha = process.env.GITHUB_SHA
  if (!sha) return null
  const branch = process.env.GITHUB_REF_NAME || process.env.GITHUB_HEAD_REF || null
  return {
    commit: sha,
    branch: branch,
    dirty: false, // CI builds from a checkout-of-ref by definition
    source: "ci"
  }
}

function fromLocalGit() {
  const sha = tryExec("git rev-parse HEAD", { cwd: REPO_ROOT })
  if (!sha) return null
  const branch = tryExec("git rev-parse --abbrev-ref HEAD", { cwd: REPO_ROOT })
  // `git status --porcelain -uno` is empty iff tracked files match HEAD.
  // We exclude untracked files (-uno) intentionally: a developer who's
  // checked out an installer scratch dir alongside the repo shouldn't
  // poison every local build with a [DIRTY] stamp.  We DO care about
  // tracked-but-modified files because those mean the .exe content
  // differs from the commit being pinned.
  const status = tryExec("git status --porcelain -uno", { cwd: REPO_ROOT })
  const dirty = status !== null && status.length > 0
  return {
    commit: sha,
    branch: branch === "HEAD" ? null : branch, // detached HEAD -> null
    dirty: dirty,
    source: "local"
  }
}

// ── ApexNodes V0.1 runtime pin ─────────────────────────────────────────────
// Path ②: the desktop ships our fork's Electron SHELL but installs the UPSTREAM
// Hermes Agent runtime at first launch (we deliberately do NOT fork the
// runtime). Our fork's build HEAD is not on NousResearch, so we cannot pin the
// first-launch runtime clone to it: bootstrap-runner fetches install.sh from
// raw.githubusercontent.com/NousResearch/hermes-agent/<commit> and install.sh
// then `git clone --branch <branch>` from NousResearch + checks out <commit>.
// We therefore pin to the upstream v0.17 release TAG — `git clone --depth 1
// --branch v2026.6.19` lands exactly on that commit, so no by-SHA fetch is
// needed (which the parent repo may reject). Bump both values when the bundle
// adopts a newer upstream runtime; set APEXNODES_RUNTIME_PIN=0 to fall back to
// the git HEAD (only correct when building from a ref pushed to NousResearch).
function fromApexNodesPin() {
  if (process.env.APEXNODES_RUNTIME_PIN === "0") return null
  return {
    commit: process.env.APEXNODES_RUNTIME_COMMIT || "87740e8021390455962caa3ad2c16d522c0d306a",
    branch: process.env.APEXNODES_RUNTIME_REF || "v2026.6.19",
    dirty: false,
    source: "apexnodes-pin"
  }
}

function main() {
  const stamp = fromApexNodesPin() || fromCI() || fromLocalGit()
  if (!stamp || !stamp.commit) {
    console.error(
      "[write-build-stamp] ERROR: could not determine git commit.\n" +
        "  - $GITHUB_SHA not set\n" +
        "  - `git rev-parse HEAD` failed at " +
        REPO_ROOT +
        "\n" +
        "Packaged builds require a git ref to pin first-launch install.ps1\n" +
        "against. Run from a git checkout or set $GITHUB_SHA explicitly."
    )
    process.exit(1)
  }

  if (stamp.dirty) {
    console.warn(
      "[write-build-stamp] WARNING: working tree is dirty.\n" +
        "  Pinning to " +
        stamp.commit.slice(0, 12) +
        " but the packaged code may differ from that commit.\n" +
        "  Commit your changes before publishing this build."
    )
  }

  const payload = {
    schemaVersion: STAMP_SCHEMA_VERSION,
    commit: stamp.commit,
    branch: stamp.branch,
    builtAt: new Date().toISOString(),
    dirty: stamp.dirty,
    source: stamp.source
  }

  fs.mkdirSync(OUT_DIR, { recursive: true })
  fs.writeFileSync(OUT_FILE, JSON.stringify(payload, null, 2) + "\n", "utf8")
  console.log(
    "[write-build-stamp] wrote " +
      path.relative(REPO_ROOT, OUT_FILE) +
      " -> " +
      stamp.commit.slice(0, 12) +
      (stamp.branch ? " (" + stamp.branch + ")" : "") +
      (stamp.dirty ? " [DIRTY]" : "")
  )
}

main()

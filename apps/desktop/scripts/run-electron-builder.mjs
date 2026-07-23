// Resolve electronDist at runtime (#38673, #47917): electron-builder 26.8.x can
// re-unpack a broken Electron.app; reusing the installed dist dodges that.
// npm workspace hoisting is non-deterministic — require.resolve finds electron
// wherever it landed. Dist present → -c.electronDist=<abs>/dist; absent → let
// electron-builder fetch via @electron/get (electronVersion + ELECTRON_MIRROR).

import fs from "node:fs"
import path from "node:path"
import { spawnSync } from "node:child_process"
import { createRequire } from "node:module"
import { fileURLToPath, pathToFileURL } from "node:url"

const require = createRequire(import.meta.url)

// WORKAROUND (desktop monorepo OOM): electron-builder 26's node-module collector
// walks the ENTIRE hoisted monorepo node_modules when it detects an npm workspace
// root, which explodes into millions of async proxy allocations and OOMs the
// "searching for node modules" phase on this repo. The packaged desktop app does
// not need the monorepo tree (its runtime deps are bundled by Vite/esbuild or
// staged into dist/ by stage-native-deps.mjs), so we neutralize the workspace root
// for the electron-builder child only: strip `workspaces` from the root
// package.json, spawn, then always restore it in `finally`.

// Resolve the MONOREPO ROOT package.json (the one carrying `workspaces`) by
// walking UP from this script's directory. electron-builder detects the
// workspace root from there, so that is what we must neutralize during the pack.
// (Walking up from electron's install location lands in apps/desktop, which is
// NOT the root — the root is one level higher.)
export function repoRootPkg(startDir) {
  const scriptDir = startDir ?? path.dirname(fileURLToPath(import.meta.url)) // apps/desktop/scripts
  let dir = scriptDir
  for (let i = 0; i < 8; i++) {
    const cand = path.join(dir, "package.json")
    if (fs.existsSync(cand)) {
      try {
        const json = JSON.parse(fs.readFileSync(cand, "utf8"))
        if (json.workspaces) return cand
      } catch { /* ignore, keep walking */ }
    }
    const parent = path.dirname(dir)
    if (parent === dir) break
    dir = parent
  }
  // fallback: the monorepo root is three levels up from
  // apps/desktop/scripts (scripts -> desktop -> apps -> root).
  return path.resolve(scriptDir, "..", "..", "..", "package.json")
}

// electron-builder 26 detects the npm workspace root via BOTH the root
// package.json `workspaces` field AND the root package-lock.json (it walks up
// from the app dir and reads the lock file to find the workspace root). Stripping
// only `workspaces` is not enough — we must also hide the root package-lock.json
// during the pack so electron-builder can't see the monorepo structure. Both are
// restored in `finally`.
// IMPORTANT: replacing the lock with a *minimal stub* (not just moving it aside)
// is what makes detection fail. If we only move it aside, electron-builder falls
// back to apps/desktop/pnpm-lock.yaml (still a workspace) and OOMs. The stub is a
// valid non-workspace npm lock, so electron-builder treats the project as a plain
// package and uses the app's own (pnpm) dependency tree only — no monorepo walk.
const MINIMAL_LOCK = JSON.stringify({
  name: "hermes-agent",
  version: "1.0.0",
  lockfileVersion: 3,
  requires: true,
  packages: { "": { name: "hermes-agent", version: "1.0.0" } },
}, null, 2)

function neutralizeWorkspaces() {
  const pkgPath = repoRootPkg()
  const lockPath = path.join(path.dirname(pkgPath), "package-lock.json")
  let pkgBackup = null
  let lockBackup = null
  try {
    const raw = fs.readFileSync(pkgPath, "utf8")
    const json = JSON.parse(raw)
    if (json.workspaces) {
      pkgBackup = pkgPath + ".owlbak"
      fs.writeFileSync(pkgBackup, raw)
      delete json.workspaces
      fs.writeFileSync(pkgPath, JSON.stringify(json, null, 2))
    }
    if (fs.existsSync(lockPath)) {
      lockBackup = lockPath + ".owlbak"
      fs.renameSync(lockPath, lockBackup)
      fs.writeFileSync(lockPath, MINIMAL_LOCK)
    }
    console.warn("[run-electron-builder] neutralized workspace root for this pack (restored after).")
  } catch (e) {
    console.warn("[run-electron-builder] workspace neutralize skipped: " + e.message)
  }
  return { pkgPath, pkgBackup, lockPath, lockBackup }
}

function restoreWorkspaces(state) {
  if (!state) return
  try {
    if (state.pkgBackup) {
      fs.writeFileSync(state.pkgPath, fs.readFileSync(state.pkgBackup))
      fs.unlinkSync(state.pkgBackup)
    }
    if (state.lockBackup && fs.existsSync(state.lockBackup)) {
      fs.renameSync(state.lockBackup, state.lockPath)
    }
    console.warn("[run-electron-builder] restored workspace root.")
  } catch (e) {
    console.warn("[run-electron-builder] could not auto-restore workspace root; backups at " + state.pkgBackup + " / " + state.lockBackup)
  }
}

function electronDistDir() {
  try {
    return path.join(path.dirname(require.resolve("electron/package.json")), "dist")
  } catch {
    return null
  }
}

function distBinary(dist) {
  if (process.platform === "darwin") {
    return path.join(dist, "Electron.app", "Contents", "MacOS", "Electron")
  }
  if (process.platform === "win32") {
    return path.join(dist, "electron.exe")
  }
  return path.join(dist, "electron")
}

function electronBuilderCli() {
  const pkgJson = require.resolve("electron-builder/package.json")
  const bin = require(pkgJson).bin
  const rel = typeof bin === "string" ? bin : bin["electron-builder"]
  return path.join(path.dirname(pkgJson), rel)
}

const dist = electronDistDir()
const args = []
if (dist && fs.existsSync(distBinary(dist))) {
  args.push(`-c.electronDist=${dist}`)
} else {
  console.warn(
    "[run-electron-builder] no local electron dist; electron-builder will fetch " +
      "via @electron/get (electronVersion + ELECTRON_MIRROR)."
  )
}
args.push(...process.argv.slice(2))

// Only run the actual pack when invoked directly (e.g. `node
// run-electron-builder.mjs`). When imported by a test, skip the
// spawnSync/process.exit so the module can be loaded safely.
const isMain =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href
if (isMain) {
// WORKAROND: strip `workspaces` from the root package.json and replace the root
// package-lock.json with a minimal stub so electron-builder's node-module
// collector only sees this app's context (not the whole monorepo), avoiding the
// unbounded walk / OOM. Always restore, even on failure or if this process is
// killed (SIGINT/SIGTERM) mid-pack.
const wsState = neutralizeWorkspaces()
function safeRestore() { try { restoreWorkspaces(wsState) } catch { /* best effort */ } }
process.on("exit", safeRestore)
process.on("SIGINT", () => { safeRestore(); process.exit(130) })
process.on("SIGTERM", () => { safeRestore(); process.exit(143) })
try {
  const result = spawnSync(process.execPath, [electronBuilderCli(), ...args], {
    stdio: "inherit",
  })
  if (result.error) {
    console.error(`[run-electron-builder] spawn failed: ${result.error.message}`)
    process.exit(1)
  }
  process.exit(result.status == null ? 1 : result.status)
} finally {
  safeRestore()
}
}
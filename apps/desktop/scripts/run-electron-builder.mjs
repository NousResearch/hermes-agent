// Resolve electronDist at runtime (#38673, #47917): electron-builder 26.8.x can
// re-unpack a broken Electron.app; reusing the installed dist dodges that.
// npm workspace hoisting is non-deterministic — require.resolve finds electron
// wherever it landed. Dist present → -c.electronDist=<abs>/dist; absent → let
// electron-builder fetch via @electron/get (electronVersion + ELECTRON_MIRROR).

import fs from "node:fs"
import path from "node:path"
import { spawnSync } from "node:child_process"
import { createRequire } from "node:module"

import { isMain } from "./utils.mjs"

const require = createRequire(import.meta.url)

// electron-builder's asar/blockmap pass runs out of the default V8 heap on the
// desktop bundle. The flag lives HERE, not in a `cross-env NODE_OPTIONS=…` npm
// script prefix: the `builder` script used to depend on the cross-env bin, and
// a devDependency the updater's `npm ci` did not stage (omit=dev inherited, a
// half-rolled-back install, a pruned workspace bin dir) turned every
// `hermes update` / `hermes desktop` pack into `cross-env: not found` /
// `'cross-env' is not recognized`, which the Python recovery then misread as a
// blocked Electron download and retried via the mirror (#110121). This script
// already spawns `process.execPath` itself, so it can set the heap flag on the
// child without any shell-portability helper.
export const HEAP_FLAG = "--max-old-space-size=16384"

export function builderNodeOptions(inherited = process.env.NODE_OPTIONS) {
  const parts = (inherited ?? "").split(/\s+/).filter(Boolean)
  if (!parts.some((p) => p.startsWith("--max-old-space-size"))) parts.push(HEAP_FLAG)
  return parts.join(" ")
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

if (isMain(import.meta.url)) {
  const dist = electronDistDir()
  // Local `hermes desktop` builds only ever package (--dir or dist), never
  // publish a GitHub release — no CI workflow drives this script. But the npm
  // lifecycle env sets CI=1 (so esbuild's postinstall doesn't try interactive
  // animations), and electron-builder treats CI=1 as a signal to implicitly
  // resolve a publish target. That resolution reads <projectDir>/.git/config
  // directly — projectDir here is apps/desktop, which has no .git of its own
  // (only the repo root does) and no "repository" field in its package.json —
  // so it fails with "Cannot detect repository by .git/config". Pin publish to
  // "never" so electron-builder skips that lookup entirely.
  const args = ["--publish", "never"]
  if (dist && fs.existsSync(distBinary(dist))) {
    args.push(`-c.electronDist=${dist}`)
  } else {
    console.warn(
      "[run-electron-builder] no local electron dist; electron-builder will fetch " +
        "via @electron/get (electronVersion + ELECTRON_MIRROR)."
    )
  }
  args.push(...process.argv.slice(2))

  const result = spawnSync(process.execPath, [electronBuilderCli(), ...args], {
    stdio: "inherit",
    env: { ...process.env, NODE_OPTIONS: builderNodeOptions() },
  })
  if (result.error) {
    console.error(`[run-electron-builder] spawn failed: ${result.error.message}`)
    process.exit(1)
  }
  process.exit(result.status == null ? 1 : result.status)
}

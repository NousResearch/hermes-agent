// Resolve electronDist at runtime (#38673, #47917): electron-builder 26.8.x can
// re-unpack a broken Electron.app; reusing the installed dist dodges that.
// npm workspace hoisting is non-deterministic — require.resolve finds electron
// wherever it landed. Dist present → -c.electronDist=<abs>/dist; absent → let
// electron-builder fetch via @electron/get (electronVersion + ELECTRON_MIRROR).

import fs from "node:fs"
import path from "node:path"
import { randomUUID } from "node:crypto"
import { spawnSync } from "node:child_process"
import { fileURLToPath } from "node:url"
import { createRequire } from "node:module"

import {
  BUILDER_REEXEC_GUARD_ENV,
  desktopBuilderRuntimeProblem,
  selectNpmNodeRuntime
} from "./desktop-builder-runtime.mjs"
import { PACK_SESSION_ENV, settleDesktopPack } from "./desktop-pack-transaction.mjs"

const require = createRequire(import.meta.url)
const wrapperPath = fileURLToPath(import.meta.url)
const scriptDir = path.dirname(wrapperPath)
const releaseDir = path.join(path.dirname(scriptDir), "release")

const selectedNpmNode = selectNpmNodeRuntime({
  currentExecPath: process.execPath,
  npmNodeExecPath: process.env.npm_node_execpath,
  guardValue: process.env[BUILDER_REEXEC_GUARD_ENV],
  exists: fs.existsSync,
  realpath: fs.realpathSync.native
})
if (selectedNpmNode) {
  console.warn(
    `[run-electron-builder] PATH selected ${process.execPath}; re-executing with npm runtime ${selectedNpmNode}`
  )
  const reexec = spawnSync(selectedNpmNode, [wrapperPath, ...process.argv.slice(2)], {
    env: { ...process.env, [BUILDER_REEXEC_GUARD_ENV]: "1" },
    stdio: "inherit"
  })
  if (reexec.error) {
    console.error(`[run-electron-builder] Node runtime hand-off failed: ${reexec.error.message}`)
    process.exit(1)
  }
  process.exit(reexec.status == null ? 1 : reexec.status)
}

const runtimeProblem = desktopBuilderRuntimeProblem({
  version: process.versions.node,
  execPath: process.execPath,
  requireModuleSupported: process.features?.require_module
})
if (runtimeProblem) {
  console.error(`[run-electron-builder] ${runtimeProblem}`)
  console.error(
    "[run-electron-builder] Close stale version-manager shells or run the build through Hermes-managed Node."
  )
  process.exit(1)
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

const packSession = process.env[PACK_SESSION_ENV] || randomUUID()
const result = spawnSync(process.execPath, [electronBuilderCli(), ...args], {
  env: { ...process.env, [PACK_SESSION_ENV]: packSession },
  stdio: "inherit",
})
const builderSucceeded = !result.error && result.status === 0
const settlement = settleDesktopPack({ releaseDir, builderSucceeded, sessionId: packSession })

for (const restoredDir of settlement.restored) {
  console.warn(`[run-electron-builder] restored previous packaged app: ${restoredDir}`)
}
for (const retainedDir of settlement.retained) {
  console.log(
    `[run-electron-builder] retained rollback for canonical launchability verification: ${retainedDir}`
  )
}
for (const discardedDir of settlement.discarded) {
  console.log(`[run-electron-builder] discarded verified rollback backup: ${discardedDir}`)
}
for (const failure of settlement.failures) {
  console.error(`[run-electron-builder] ${failure.reason}`)
}

if (result.error) {
  console.error(`[run-electron-builder] spawn failed: ${result.error.message}`)
  process.exit(1)
}
if (!settlement.ok) {
  process.exit(1)
}
process.exit(result.status == null ? 1 : result.status)

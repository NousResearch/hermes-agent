// Resolve electronDist at runtime (#38673, #47917): electron-builder 26.8.x can
// re-unpack a broken Electron.app; reusing the installed dist dodges that.
// npm workspace hoisting is non-deterministic — require.resolve finds electron
// wherever it landed. Dist present → -c.electronDist=<abs>/dist; absent → let
// electron-builder fetch via @electron/get (electronVersion + ELECTRON_MIRROR).
//
// #69179: also refuse a host-local dist whose PE Machine does not match the
// build target. `npm_config_arch` (or a poisoned cache) can leave an arm64
// electron.exe on an AMD64 host; packing that into win-unpacked yields the
// Windows modal 「此应用无法在你的电脑上运行」 / "This app can't run on your PC".

import path from "node:path"
import { spawnSync } from "node:child_process"
import { createRequire } from "node:module"

import {
  resolveBuilderTargetArch,
  shouldReuseElectronDist,
} from "./run-electron-builder-lib.mjs"

const require = createRequire(import.meta.url)

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
const targetArch = resolveBuilderTargetArch(process.argv.slice(2), process.arch)
const binary = dist ? distBinary(dist) : null
const decision = shouldReuseElectronDist({
  platform: process.platform,
  distDir: dist,
  binaryPath: binary,
  targetArch,
  hostArch: process.arch,
})

if (decision.reuse) {
  args.push(`-c.electronDist=${dist}`)
} else if (decision.reason === "arch-mismatch") {
  console.warn(
    `[run-electron-builder] refusing host electronDist (${decision.got}) for ` +
      `target ${decision.want}; electron-builder will fetch a matching build ` +
      `via @electron/get (#69179).`
  )
} else if (decision.reason === "unreadable-pe") {
  console.warn(
    `[run-electron-builder] host electronDist PE unreadable; electron-builder ` +
      `will fetch via @electron/get (#69179).`
  )
} else {
  console.warn(
    "[run-electron-builder] no local electron dist; electron-builder will fetch " +
      "via @electron/get (electronVersion + ELECTRON_MIRROR)."
  )
}
args.push(...process.argv.slice(2))

const result = spawnSync(process.execPath, [electronBuilderCli(), ...args], {
  stdio: "inherit",
})
if (result.error) {
  console.error(`[run-electron-builder] spawn failed: ${result.error.message}`)
  process.exit(1)
}
process.exit(result.status == null ? 1 : result.status)

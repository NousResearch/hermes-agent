// Resolve electronDist at runtime (#38673, #47917): electron-builder 26.8.x can
// re-unpack a broken Electron.app; reusing the installed dist dodges that.
// npm workspace hoisting is non-deterministic — require.resolve finds electron
// wherever it landed. Dist present → -c.electronDist=<abs>/dist; absent → let
// electron-builder fetch via @electron/get (electronVersion + ELECTRON_MIRROR).

import fs from "node:fs"
import path from "node:path"
import { spawnSync } from "node:child_process"
import { createRequire } from "node:module"

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

// Mach-O arch of `binary`: 'arm64' | 'x64' | 'universal' | null. Only the header is read.
function machoArch(binary) {
  let fd
  try {
    fd = fs.openSync(binary, "r")
    const head = Buffer.alloc(8)
    if (fs.readSync(fd, head, 0, 8, 0) < 8) return null
    const magic = head.readUInt32BE(0)
    if (magic === 0xcafebabe || magic === 0xbebafeca) return "universal"
    let cpu
    if (magic === 0xfeedface || magic === 0xfeedfacf) cpu = head.readUInt32BE(4)
    else if (magic === 0xcefaedfe || magic === 0xcffaedfe) cpu = head.readUInt32LE(4)
    else return null
    if (cpu === 0x0100000c) return "arm64"
    if (cpu === 0x01000007) return "x64"
    return null
  } catch {
    return null
  } finally {
    if (fd !== undefined) fs.closeSync(fd)
  }
}

function electronBuilderCli() {
  const pkgJson = require.resolve("electron-builder/package.json")
  const bin = require(pkgJson).bin
  const rel = typeof bin === "string" ? bin : bin["electron-builder"]
  return path.join(path.dirname(pkgJson), rel)
}

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
// Pin the macOS target arch to the Electron binary we are ACTUALLY packaging. electron-builder defaults
// to x64 on macOS, but `-c.electronDist` above hands it the local Electron — arm64 on Apple Silicon. The
// mismatch is silent and shipped a broken app three times: output lands in `release/mac` (the x64 dir),
// before-pack.mjs stages node-pty for the DECLARED x64 target, and the resulting arm64 app cannot load
// prebuilds/darwin-arm64/pty.node. `process.arch` is NOT a usable proxy — ~/.local/bin/node is an x86_64
// build on this arm64 Mac and reports "x64" — so the arch is read off the binary itself. An explicit
// --arm64/--x64/--universal from the caller always wins.
const forwarded = process.argv.slice(2)
const ARCH_FLAGS = ["--arm64", "--x64", "--ia32", "--armv7l", "--universal"]
if (process.platform === "darwin" && !forwarded.some((a) => ARCH_FLAGS.includes(a))) {
  const arch = dist ? machoArch(distBinary(dist)) : null
  if (arch === "arm64" || arch === "x64") {
    args.push(`--${arch}`)
    console.log(
      `[run-electron-builder] pinning target arch to ${arch} ` +
        `(read from the Electron binary; process.arch=${process.arch})`
    )
  } else {
    console.warn(
      "[run-electron-builder] could not read the packaged Electron binary's architecture; " +
        "leaving the target unpinned (electron-builder will default to x64 on macOS)."
    )
  }
}
args.push(...forwarded)

const result = spawnSync(process.execPath, [electronBuilderCli(), ...args], {
  stdio: "inherit",
})
if (result.error) {
  console.error(`[run-electron-builder] spawn failed: ${result.error.message}`)
  process.exit(1)
}
process.exit(result.status == null ? 1 : result.status)

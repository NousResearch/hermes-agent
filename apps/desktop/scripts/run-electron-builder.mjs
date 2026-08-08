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

// Never let electron-builder publish. On a CI tag build it auto-detects
// GitHub and demands GH_TOKEN after the artifacts are already built.
// The release workflow uploads artifacts in its own step.
if (!args.includes("--publish") && !args.some((a) => a.startsWith("-p"))) {
  args.push("--publish", "never")
}

// Windows signing config is composed HERE, from the AZURE_SIGN_* variables,
// not passed down as -c arguments. The publisherName contains spaces and
// commas, and no quoting survives the cmd.exe hops between the outer build
// script, npm's lifecycle spawn, and this script. This spawn is the first
// one with no shell in between, so values pass through verbatim.
// (azureSignOptions is the 26.x schema; the old win.sign.type=azure shape
// fails validation.)
if (
  args.includes("--win") &&
  process.env.AZURE_SIGN_ENDPOINT &&
  process.env.AZURE_CLIENT_ID &&
  !args.some((a) => a.includes("azureSignOptions"))
) {
  console.log(`[run-electron-builder] Windows signing: Azure Trusted Signing at ${process.env.AZURE_SIGN_ENDPOINT}`)
  args.push(
    "-c.win.signAndEditExecutable=true",
    `-c.win.azureSignOptions.endpoint=${process.env.AZURE_SIGN_ENDPOINT}`,
    `-c.win.azureSignOptions.codeSigningAccountName=${process.env.AZURE_SIGN_ACCOUNT}`,
    `-c.win.azureSignOptions.certificateProfileName=${process.env.AZURE_SIGN_PROFILE}`,
    `-c.win.azureSignOptions.publisherName=${process.env.AZURE_SIGN_PUBLISHER}`
  )
}

const result = spawnSync(process.execPath, [electronBuilderCli(), ...args], {
  stdio: "inherit",
})
if (result.error) {
  console.error(`[run-electron-builder] spawn failed: ${result.error.message}`)
  process.exit(1)
}
process.exit(result.status == null ? 1 : result.status)

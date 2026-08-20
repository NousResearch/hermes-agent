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

// Dev-only helper, inert unless HERMES_MAC_TIMESTAMP=none is exported.
// electron-builder's argv passthrough (-c.mac.timestamp=none) can already
// skip the timestamp server; this helper adds what argv cannot: a preflight
// keychain probe that warns up front when CSC_NAME is missing, instead of
// discovering mid-build that osx-sign failed and fell back to ad-hoc.
// Ad-hoc designated requirements bind the bundle identifier to the binary
// cdhash, which changes on every pack — macOS then treats each repack as a
// new app and re-prompts for TCC permissions (Screen Recording,
// Accessibility) the user already granted. Release/CI flows that set
// CSC_NAME without this env are completely untouched (no probe, no args).
function macSelfSignedDevArgs() {
  if (process.platform !== "darwin") return []
  if ((process.env.HERMES_MAC_TIMESTAMP || "").toLowerCase() !== "none") return []

  const identity = process.env.CSC_NAME
  // electron-builder sentinels for "do not sign" — a timestamp flag is
  // meaningless for an unsigned build.
  if (identity === "none" || identity === "-") return []

  // Skipping Apple's timestamp server is equally possible via electron-
  // builder's argv passthrough (-c.mac.timestamp=none); this helper exists
  // for the preflight: fail loudly NOW when CSC_NAME is not in the keychain,
  // instead of discovering mid-build that osx-sign fell back to ad-hoc.
  const args = [`-c.mac.timestamp=none`]
  console.log(
    "[run-electron-builder] HERMES_MAC_TIMESTAMP=none → -c.mac.timestamp=none " +
      "(self-signed identity: skip Apple timestamp server)"
  )

  if (!identity) {
    console.warn(
      "[run-electron-builder] HERMES_MAC_TIMESTAMP=none set but CSC_NAME is " +
        "not: skipping the timestamp server WITHOUT selecting an identity — " +
        "electron-builder will auto-detect one, or sign ad-hoc. Set CSC_NAME " +
        "to pin the intended identity."
    )
    return args
  }

  const probe = spawnSync("security", ["find-identity", "-v", "-p", "codesigning"], {
    encoding: "utf8",
    timeout: 5000,
  })
  if (probe.status !== 0 || !keychainHasIdentity(probe, identity)) {
    console.warn(
      `[run-electron-builder] CSC_NAME="${identity}" not found among keychain ` +
        "code-signing identities; signing will fail or fall back to ad-hoc."
    )
  }
  return args
}

// `security find-identity -v` prints lines like:  1) ABCDEF0123… "Developer ID …"
// Match the exact quoted name, never substrings (CSC_NAME=Apple must not
// false-pass against "Developer ID Application: …"; a TEAMID fragment must
// not false-pass either). The 40-hex cert hash printed before the quoted
// name is also accepted, since CSC_NAME=<hash> is a valid way to pin one.
function keychainHasIdentity(probe, identity) {
  const text = `${probe.stdout || ""}\n${probe.stderr || ""}`
  const isHash = /^[0-9A-Fa-f]{40}$/.test(identity)
  for (const line of text.split("\n")) {
    const m = line.match(/"((?:[^"\\]|\\.)*)"/)
    if (m && m[1] === identity) return true
    if (isHash && line.includes(identity)) return true
  }
  return false
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
args.push(...macSelfSignedDevArgs())
args.push(...process.argv.slice(2))

const result = spawnSync(process.execPath, [electronBuilderCli(), ...args], {
  stdio: "inherit",
})
if (result.error) {
  console.error(`[run-electron-builder] spawn failed: ${result.error.message}`)
  process.exit(1)
}
process.exit(result.status == null ? 1 : result.status)

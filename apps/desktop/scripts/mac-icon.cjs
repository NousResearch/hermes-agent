// Which macOS icon resource electron-builder packages, decided per build host.
//
// `assets/icon.icon` is the Icon Composer package for macOS 26 (the system
// masks its layers itself, so the border ring follows the real outline).
// electron-builder compiles it to `Contents/Resources/Assets.car` with
// actool, which exists only in Xcode >= 26; on any other host it throws and
// the whole build fails. Choosing at config time keeps a dev Mac on Xcode
// 16 building the .icns-only app it always built, while release CI selects
// Xcode 26 (`DEVELOPER_DIR`) and asserts the version before packaging so a
// silent fallback can never ship from there.
//
// Whichever resource is chosen, after-pack.mjs restores `assets/icon.icns`
// as the bundle's legacy icon: electron-builder's own `.icon` path replaces
// it with actool's 256px fallback, and macOS <= 15 shows the .icns.
// @ts-check
'use strict'

const path = require('node:path')
const { spawnSync } = require('node:child_process')

const ICON_COMPOSER = 'assets/icon.icon'
const LEGACY_ICNS = 'assets/icon.icns'

/**
 * The actool short version from `actool --version` plist output, or null
 * when actool is missing or answers with something else.
 * @param {string} output
 * @returns {string | null}
 */
function parseActoolVersion(output) {
  const match = /<key>short-bundle-version<\/key>\s*<string>([^<]+)<\/string>/.exec(output)
  return match ? match[1].trim() : null
}

/**
 * @param {string | null} version
 * @returns {boolean}
 */
function actoolSupportsIconComposer(version) {
  if (!version) return false
  const major = Number.parseInt(version.split('.')[0], 10)
  return Number.isInteger(major) && major >= 26
}

/**
 * @param {(command: string, args: string[]) => { status: number | null, stdout?: string | Buffer | null, stderr?: string | Buffer | null }} run
 * @returns {string | null}
 */
function installedActoolVersion(run = (command, args) => spawnSync(command, args, { encoding: 'utf8' })) {
  try {
    const result = run('actool', ['--version'])
    if (result.status !== 0) return null
    return parseActoolVersion(`${result.stdout ?? ''}${result.stderr ?? ''}`)
  } catch {
    return null
  }
}

/**
 * The `mac.icon` value for this host: the Icon Composer package when actool
 * can compile it, otherwise the legacy .icns alone.
 * @param {string} appDir the apps/desktop directory
 * @param {{ platform?: string, actoolVersion?: string | null, log?: (message: string) => void }} [options]
 * @returns {string}
 */
function macIconResource(appDir, options = {}) {
  const platform = options.platform ?? process.platform
  if (platform !== 'darwin') return LEGACY_ICNS
  const version = options.actoolVersion === undefined ? installedActoolVersion() : options.actoolVersion
  if (actoolSupportsIconComposer(version)) return ICON_COMPOSER
  const log = options.log ?? (message => console.warn(message))
  log(
    `[mac-icon] actool ${version ?? 'not found'}: packaging ${LEGACY_ICNS} only; ` +
      `the macOS 26 layered icon (${path.join(appDir, ICON_COMPOSER)}) needs Xcode 26 or newer`
  )
  return LEGACY_ICNS
}

module.exports = {
  ICON_COMPOSER,
  LEGACY_ICNS,
  actoolSupportsIconComposer,
  installedActoolVersion,
  macIconResource,
  parseActoolVersion
}

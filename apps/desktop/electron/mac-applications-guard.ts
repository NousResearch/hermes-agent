import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import type { App, Dialog, MessageBoxOptions } from 'electron'

import type { InstallStamp } from './install-stamp'
import { ephemeralMacBundleLocation, type MacEphemeralBundleLocation, resolveMacAppBundle } from './mac-bundle-location'

export const MAC_APPLICATIONS_GUARD_FILENAME = 'mac-applications-guard.json'

export type MacUnsafeBundleLocation = MacEphemeralBundleLocation | 'desktop-folder' | 'downloads-folder'

// "Don't ask again" is all startup reads, so it's all the file holds.
export interface MacApplicationsGuardPreference {
  dismissed: true
}

export type MacApplicationsGuardResult =
  | { action: 'relaunching'; location: MacUnsafeBundleLocation }
  | {
      action: 'continue'
      reason:
        | 'admin-required'
        | 'already-in-applications'
        | 'api-unavailable'
        | 'automated-environment'
        | 'conflict-newer-installed'
        | 'conflict-running'
        | 'conflict-unverified-installed'
        | 'custom-writable-location'
        | 'dismissed'
        | 'error'
        | 'move-cancelled'
        | 'move-failed'
        | 'not-app-bundle'
        | 'not-mac-bundle'
        | 'not-macos'
        | 'not-packaged'
        | 'user-declined'
      location?: MacUnsafeBundleLocation
      remembered?: boolean
      error?: string
    }

export interface MacApplicationsGuardDeps {
  app: Pick<App, 'getPath' | 'getVersion' | 'isInApplicationsFolder' | 'moveToApplicationsFolder'>
  dialog: Pick<Dialog, 'showMessageBox'>
  appName: string
  isPackaged: boolean
  installStamp: Readonly<Pick<InstallStamp, 'payload' | 'updateMechanism'>> | null
  platform?: NodeJS.Platform
  env?: NodeJS.ProcessEnv
  argv?: readonly string[]
  homeDir?: string
  access?: (file: string, mode: number) => void
  exists?: (file: string) => boolean
  writable?: (file: string) => boolean
  readBundleVersion?: (bundlePath: string) => string | null
  log: (message: string) => void
}

// Only an artifact Squirrel.Mac updates in place gains automatic updates by moving; a commit build
// (updateMechanism external) doesn't update in-app, so its copy names only the location's own problem.
const LOCATION_DETAILS: Readonly<Record<MacUnsafeBundleLocation, (appName: string, inAppUpdates: boolean) => string>> =
  {
    'app-translocation': (appName, inAppUpdates) =>
      `${appName} is running from a temporary macOS App Translocation folder (usually because it was opened from Downloads). ` +
      (inAppUpdates
        ? `Automatic updates cannot install from here. Moving ${appName} to Applications fixes this.`
        : `macOS gives it a new read-only location on every launch. Moving ${appName} to Applications gives it a permanent one.`),
    'desktop-folder': (appName, inAppUpdates) =>
      `${appName} is running from your Desktop folder. ` +
      (inAppUpdates
        ? 'Moving it to Applications keeps automatic updates working reliably.'
        : 'Moving it to Applications gives it a permanent home.'),
    'downloads-folder': (appName, inAppUpdates) =>
      `${appName} is running from your Downloads folder. ` +
      (inAppUpdates
        ? 'Moving it to Applications keeps automatic updates working reliably.'
        : 'Moving it to Applications gives it a permanent home.'),
    'read-only-volume': (appName, inAppUpdates) =>
      `${appName} is running from a read-only volume (such as a disk image). ` +
      (inAppUpdates
        ? `Automatic updates cannot install from here, and ${appName} stops working once the volume is ejected. ` +
          `Moving ${appName} to Applications fixes both.`
        : `${appName} stops working once the volume is ejected. Moving ${appName} to Applications fixes this.`)
  }

// While the move prompt waits, app.isReady() is already true, so a second launch must not open the first
// window (and start the backend) behind it; startup opens that window once the guard lets it continue.
export function macGuardAllowsWindows(result: MacApplicationsGuardResult | null): boolean {
  return result?.action === 'continue'
}

function isTruthyEnv(value: string | undefined): boolean {
  const normalized = String(value ?? '')
    .trim()
    .toLowerCase()

  return (
    normalized !== '' && normalized !== '0' && normalized !== 'false' && normalized !== 'no' && normalized !== 'off'
  )
}

export function isAutomatedDesktopLaunch(env: NodeJS.ProcessEnv, argv: readonly string[]): boolean {
  if (
    isTruthyEnv(env.CI) ||
    isTruthyEnv(env.GITHUB_ACTIONS) ||
    isTruthyEnv(env.HERMES_DESKTOP_SKIP_QUIT_CONFIRM) ||
    Boolean(env.HERMES_DESKTOP_USER_DATA_DIR?.trim())
  ) {
    return true
  }

  return argv.some(arg => arg === '--remote-debugging-port' || arg.startsWith('--remote-debugging-port='))
}

export function isMacBundleArtifact(
  stamp: Readonly<Pick<InstallStamp, 'payload' | 'updateMechanism'>> | null | undefined
): boolean {
  return stamp?.payload === 'bundled' || stamp?.payload === 'light' || stamp?.updateMechanism === 'electron-updater'
}

function isPosixPathWithin(root: string, candidate: string): boolean {
  const rel = path.posix.relative(path.posix.resolve(root), path.posix.resolve(candidate))

  return rel !== '' && rel !== '..' && !rel.startsWith('../') && !path.posix.isAbsolute(rel)
}

export function classifyMacBundleLocation(
  bundlePath: string,
  opts?: { homeDir?: string; access?: (file: string, mode: number) => void }
): MacUnsafeBundleLocation | null {
  const ephemeral = ephemeralMacBundleLocation(bundlePath, opts?.access)

  if (ephemeral) {
    return ephemeral
  }

  // Only flag Downloads/Desktop; CI ShipIt relaunches run in writable runner temp dirs without CI env vars.
  const normalized = path.posix.resolve(bundlePath)
  const home = opts?.homeDir ? path.posix.resolve(opts.homeDir) : ''

  if (home) {
    if (isPosixPathWithin(path.posix.join(home, 'Downloads'), normalized)) {
      return 'downloads-folder'
    }

    if (isPosixPathWithin(path.posix.join(home, 'Desktop'), normalized)) {
      return 'desktop-folder'
    }
  }

  return null
}

export function macApplicationsGuardPreferencePath(userDataDir: string): string {
  return path.join(userDataDir, MAC_APPLICATIONS_GUARD_FILENAME)
}

export function readMacApplicationsGuardPreference(userDataDir: string): MacApplicationsGuardPreference | null {
  if (!userDataDir) {
    return null
  }

  try {
    const parsed = JSON.parse(fs.readFileSync(macApplicationsGuardPreferencePath(userDataDir), 'utf8')) as {
      dismissed?: unknown
    } | null

    return parsed?.dismissed === true ? { dismissed: true } : null
  } catch {
    return null
  }
}

export function writeMacApplicationsGuardPreference(userDataDir: string): void {
  const target = macApplicationsGuardPreferencePath(userDataDir)
  const tmp = `${target}.tmp-${process.pid}-${Date.now()}`
  const preference: MacApplicationsGuardPreference = { dismissed: true }

  fs.mkdirSync(userDataDir, { recursive: true })
  fs.writeFileSync(tmp, `${JSON.stringify(preference, null, 2)}\n`, 'utf8')

  try {
    fs.renameSync(tmp, target)
  } catch (error) {
    try {
      fs.unlinkSync(tmp)
    } catch {
      void 0
    }

    throw error
  }
}

interface ParsedBundleVersion {
  major: number
  minor: number
  patch: number
  prerelease: string[]
  build: string
}

// The SemVer 2.0.0 grammar (semver.org): anything else is a version this guard can't order.
const SEMVER_IDENTIFIER = '(?:0|[1-9]\\d*|\\d*[A-Za-z-][0-9A-Za-z-]*)'

const SEMVER_RE = new RegExp(
  `^(0|[1-9]\\d*)\\.(0|[1-9]\\d*)\\.(0|[1-9]\\d*)` +
    `(?:-(${SEMVER_IDENTIFIER}(?:\\.${SEMVER_IDENTIFIER})*))?` +
    `(?:\\+([0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*))?$`
)

function parseBundleVersion(raw: string): ParsedBundleVersion | null {
  const match = SEMVER_RE.exec(raw.trim())

  if (!match) {
    return null
  }

  const prerelease = match[4] ? match[4].split('.') : []
  const numbers = [match[1], match[2], match[3], ...prerelease.filter(part => /^\d+$/.test(part))].map(Number)

  // Past 2^53 two different numbers can read as one, so such a version can't be ordered exactly.
  if (!numbers.every(Number.isSafeInteger)) {
    return null
  }

  return { major: numbers[0], minor: numbers[1], patch: numbers[2], prerelease, build: match[5] ?? '' }
}

// Canary and commit builds keep their stable core and carry the build as X.Y.Z+<kind>.<UTC stamp>, which
// SemVer precedence ignores; the fixed-width stamps order by time. Any other build-metadata difference on
// the same core and prerelease can't be ordered.
const RELEASE_BUILD_RE = /^(canary|commit)\.(20\d{6}T\d{6}Z)$/

function compareBuildMetadata(a: string, b: string): -1 | 0 | 1 | null {
  if (a === b) {
    return 0
  }

  const left = RELEASE_BUILD_RE.exec(a)
  const right = RELEASE_BUILD_RE.exec(b)

  if (!left || !right || left[1] !== right[1]) {
    return null
  }

  return left[2] < right[2] ? -1 : 1
}

function comparePrerelease(a: readonly string[], b: readonly string[]): -1 | 0 | 1 {
  if (a.length === 0 && b.length === 0) {
    return 0
  }

  if (a.length === 0) {
    return 1
  }

  if (b.length === 0) {
    return -1
  }

  const len = Math.min(a.length, b.length)

  for (let i = 0; i < len; i++) {
    const left = a[i]
    const right = b[i]

    if (left === right) {
      continue
    }

    const leftNum = /^\d+$/.test(left)
    const rightNum = /^\d+$/.test(right)

    if (leftNum && rightNum) {
      return Number(left) < Number(right) ? -1 : 1
    }

    if (leftNum !== rightNum) {
      return leftNum ? -1 : 1
    }

    return left < right ? -1 : 1
  }

  if (a.length === b.length) {
    return 0
  }

  return a.length < b.length ? -1 : 1
}

export function compareBundleVersions(running: string, existing: string): -1 | 0 | 1 | null {
  const a = parseBundleVersion(running)
  const b = parseBundleVersion(existing)

  if (!a || !b) {
    return null
  }

  for (const [left, right] of [
    [a.major, b.major],
    [a.minor, b.minor],
    [a.patch, b.patch]
  ] as const) {
    if (left !== right) {
      return left < right ? -1 : 1
    }
  }

  return comparePrerelease(a.prerelease, b.prerelease) || compareBuildMetadata(a.build, b.build)
}

export function parseBundleShortVersionFromPlist(plistXml: string): string | null {
  const match = /<key>\s*CFBundleShortVersionString\s*<\/key>\s*<string>([^<]+)<\/string>/.exec(plistXml)

  return match?.[1]?.trim() || null
}

export function readBundleShortVersion(
  bundlePath: string,
  readFile: (file: string, encoding: BufferEncoding) => string = (file, encoding) => fs.readFileSync(file, encoding)
): string | null {
  try {
    const plistPath = path.posix.join(path.posix.resolve(bundlePath), 'Contents', 'Info.plist')

    return parseBundleShortVersionFromPlist(readFile(plistPath, 'utf8'))
  } catch {
    return null
  }
}

interface InstalledCopyBlock {
  existingVersion: string | null
  newer: boolean
}

// Electron's bundle mover only ever writes /Applications/<bundle name> (NSLocalDomainMask); a copy in
// ~/Applications is never touched.
function applicationsDestination(bundlePath: string): string {
  return path.posix.join('/Applications', path.posix.basename(path.posix.resolve(bundlePath)))
}

function isWritable(file: string): boolean {
  try {
    fs.accessSync(file, fs.constants.W_OK)

    return true
  } catch {
    return false
  }
}

// Electron takes its admin-password path when /Applications, or the copy it would replace, isn't writable
// (NSFileManager isWritableFileAtPath, an access(2) W_OK check like isWritable). That path deletes the copy
// after the prompt without calling conflictHandler, so a copy installed while the prompt is open goes
// unchecked; such moves go to Finder instead, which asks before replacing.
function moveNeedsAdmin(
  bundlePath: string,
  exists: (file: string) => boolean,
  writable: (file: string) => boolean
): boolean {
  const destination = applicationsDestination(bundlePath)

  return !writable('/Applications') || (exists(destination) && !writable(destination))
}

// The installed copy may be replaced only when its version reads as no newer than this one; a version that
// can't be read or compared refuses the move.
function installedCopyBlockingMove(
  bundlePath: string,
  runningVersion: string,
  readBundleVersion: (targetBundle: string) => string | null,
  exists: (file: string) => boolean
): InstalledCopyBlock | null {
  const destination = applicationsDestination(bundlePath)

  if (!exists(destination)) {
    return null
  }

  const existingVersion = readBundleVersion(destination)
  const comparison = existingVersion ? compareBundleVersions(runningVersion, existingVersion) : null

  return comparison === null || comparison < 0 ? { existingVersion, newer: comparison !== null } : null
}

function buildMovePromptOptions(
  appName: string,
  location: MacUnsafeBundleLocation,
  inAppUpdates: boolean
): MessageBoxOptions {
  return {
    type: 'warning',
    title: `Move ${appName} to Applications`,
    message: `Move ${appName} to the Applications folder?`,
    detail: LOCATION_DETAILS[location](appName, inAppUpdates),
    buttons: ['Move to Applications', 'Keep running here'],
    defaultId: 0,
    cancelId: 1,
    checkboxLabel: "Don't ask again",
    checkboxChecked: false,
    noLink: true
  }
}

// Electron throws this when the user cancels its admin password prompt (electron_bundle_mover.mm); the user
// chose not to move, so it is a decline, not a failure.
const AUTHORIZATION_CANCELLED = 'User rejected the authorization request'

async function reportInstalledCopyConflict(
  deps: Pick<MacApplicationsGuardDeps, 'appName' | 'dialog' | 'log'>,
  location: MacUnsafeBundleLocation,
  runningVersion: string,
  { existingVersion, newer }: InstalledCopyBlock
): Promise<MacApplicationsGuardResult> {
  const notice = newer
    ? {
        reason: 'conflict-newer-installed' as const,
        log: `running ${runningVersion} is older than installed ${existingVersion}`,
        title: `Newer ${deps.appName} already installed`,
        message: `A newer version of ${deps.appName} is already in Applications`,
        detail:
          `This copy is version ${runningVersion}, while Applications already has version ${existingVersion}. ` +
          `Open ${deps.appName} from Applications instead to avoid downgrading.`
      }
    : {
        reason: 'conflict-unverified-installed' as const,
        log: `could not confirm the installed copy (${existingVersion ?? 'version unreadable'}) is not newer than running ${runningVersion}`,
        title: `${deps.appName} is already in Applications`,
        message: `${deps.appName} is already in Applications`,
        detail:
          `This copy is version ${runningVersion}, but the version of the copy in Applications could not be confirmed, ` +
          `so it was left in place. Open ${deps.appName} from Applications, or replace it in Finder.`
      }

  deps.log(`[install] move to /Applications refused: ${notice.log}`)
  await deps.dialog.showMessageBox({
    type: 'info',
    title: notice.title,
    message: notice.message,
    detail: notice.detail,
    buttons: ['Continue'],
    defaultId: 0,
    cancelId: 0,
    noLink: true
  })

  return { action: 'continue', reason: notice.reason, location }
}

async function sendMoveToFinder(
  deps: Pick<MacApplicationsGuardDeps, 'appName' | 'dialog' | 'log'>,
  location: MacUnsafeBundleLocation
): Promise<MacApplicationsGuardResult> {
  deps.log('[install] move to /Applications needs an administrator password; sent to Finder')
  await deps.dialog.showMessageBox({
    type: 'info',
    title: `Move ${deps.appName} to Applications in Finder`,
    message: `Moving ${deps.appName} to Applications needs an administrator password`,
    detail:
      `Quit ${deps.appName}, drag it into the Applications folder in Finder, then open it from there. ` +
      'Finder asks for the password, and asks before replacing a copy that is already there.',
    buttons: ['Continue'],
    defaultId: 0,
    cancelId: 0,
    noLink: true
  })

  return { action: 'continue', reason: 'admin-required', location }
}

async function attemptMoveToApplications(
  deps: Pick<
    MacApplicationsGuardDeps,
    'app' | 'appName' | 'dialog' | 'exists' | 'log' | 'readBundleVersion' | 'writable'
  >,
  bundlePath: string,
  location: MacUnsafeBundleLocation
): Promise<MacApplicationsGuardResult> {
  const readBundleVersion = deps.readBundleVersion ?? (target => readBundleShortVersion(target))
  const exists = deps.exists ?? fs.existsSync
  const runningVersion = readBundleVersion(bundlePath) ?? deps.app.getVersion()

  const checkInstalled = () => installedCopyBlockingMove(bundlePath, runningVersion, readBundleVersion, exists)

  // The version answer comes first, so a newer copy gets "open that one" even when moving needs admin rights.
  const preflight = checkInstalled()

  if (preflight) {
    return await reportInstalledCopyConflict(deps, location, runningVersion, preflight)
  }

  if (moveNeedsAdmin(bundlePath, exists, deps.writable ?? isWritable)) {
    return await sendMoveToFinder(deps, location)
  }

  let conflict: 'exists' | 'existsAndRunning' | null = null
  let blocked: InstalledCopyBlock | null = null
  let moved: boolean

  // Only Electron's mover is caught here: a notice that fails to open is not a failed move.
  try {
    moved = deps.app.moveToApplicationsFolder({
      conflictHandler: conflictType => {
        conflict = conflictType

        if (conflictType === 'existsAndRunning') {
          return false
        }

        blocked = checkInstalled()

        return blocked === null
      }
    })
  } catch (error) {
    return await reportMoveError(deps, location, error)
  }

  if (moved) {
    deps.log(`[install] relocated ${deps.appName} into /Applications from ${location}; relaunching`)

    return { action: 'relaunching', location }
  }

  if (blocked) {
    return await reportInstalledCopyConflict(deps, location, runningVersion, blocked)
  }

  if (conflict === 'existsAndRunning') {
    deps.log(`[install] move to /Applications skipped: another copy of ${deps.appName} is already running`)
    await deps.dialog.showMessageBox({
      type: 'info',
      title: `Could not move ${deps.appName}`,
      message: `Another copy of ${deps.appName} is already running in Applications`,
      detail:
        `Quit the running copy in Applications, then move ${deps.appName} to Applications in Finder ` +
        '(or reopen it from there).',
      buttons: ['Continue'],
      defaultId: 0,
      cancelId: 0,
      noLink: true
    })

    return { action: 'continue', reason: 'conflict-running', location }
  }

  deps.log('[install] move to /Applications cancelled by user')

  return { action: 'continue', reason: 'move-cancelled', location }
}

async function reportMoveError(
  deps: Pick<MacApplicationsGuardDeps, 'appName' | 'dialog' | 'log'>,
  location: MacUnsafeBundleLocation,
  error: unknown
): Promise<MacApplicationsGuardResult> {
  const message = error instanceof Error ? error.message : String(error)

  if (message === AUTHORIZATION_CANCELLED) {
    deps.log('[install] move to /Applications cancelled at the admin password prompt')

    return { action: 'continue', reason: 'move-cancelled', location }
  }

  deps.log(`[install] move to /Applications failed: ${message}`)
  await deps.dialog.showMessageBox({
    type: 'warning',
    title: `Could not move ${deps.appName}`,
    message: `${deps.appName} could not move itself to Applications`,
    detail:
      `Quit ${deps.appName}, drag it into the Applications folder in Finder, then open it from there.\n\n` +
      `Details: ${message}`,
    buttons: ['Continue'],
    defaultId: 0,
    cancelId: 0,
    noLink: true
  })

  return { action: 'continue', reason: 'move-failed', location, error: message }
}

type MacGuardPrecheck =
  | { prompt: false; result: MacApplicationsGuardResult }
  | { prompt: true; bundlePath: string; location: MacUnsafeBundleLocation; userDataDir: string }

function precheckMacApplicationsGuard(deps: MacApplicationsGuardDeps): MacGuardPrecheck {
  const platform = deps.platform ?? process.platform

  if (platform !== 'darwin') {
    return { prompt: false, result: { action: 'continue', reason: 'not-macos' } }
  }

  if (!deps.isPackaged) {
    return { prompt: false, result: { action: 'continue', reason: 'not-packaged' } }
  }

  if (!isMacBundleArtifact(deps.installStamp)) {
    return { prompt: false, result: { action: 'continue', reason: 'not-mac-bundle' } }
  }

  if (isAutomatedDesktopLaunch(deps.env ?? process.env, deps.argv ?? process.argv)) {
    return { prompt: false, result: { action: 'continue', reason: 'automated-environment' } }
  }

  const bundlePath = resolveMacAppBundle(deps.app.getPath('exe'), platform)

  if (!bundlePath) {
    return { prompt: false, result: { action: 'continue', reason: 'not-app-bundle' } }
  }

  if (
    typeof deps.app.isInApplicationsFolder !== 'function' ||
    typeof deps.app.moveToApplicationsFolder !== 'function'
  ) {
    return { prompt: false, result: { action: 'continue', reason: 'api-unavailable' } }
  }

  // The same native check as the mover's early return: when it is true, moveToApplicationsFolder returns true
  // without moving, relaunching or quitting, so a prompt here could only stall startup. A translocated
  // bundle reports false, since its resolved path is the translocation mount, so it still gets the prompt.
  if (deps.app.isInApplicationsFolder()) {
    return { prompt: false, result: { action: 'continue', reason: 'already-in-applications' } }
  }

  const location = classifyMacBundleLocation(bundlePath, {
    homeDir: deps.homeDir ?? os.homedir(),
    access: deps.access
  })

  if (!location) {
    return {
      prompt: false,
      result: { action: 'continue', reason: 'custom-writable-location' }
    }
  }

  const userDataDir = deps.app.getPath('userData')

  if (readMacApplicationsGuardPreference(userDataDir)?.dismissed) {
    deps.log(`[install] /Applications move prompt suppressed by saved preference (${location}: ${bundlePath})`)

    return { prompt: false, result: { action: 'continue', reason: 'dismissed', location } }
  }

  return { prompt: true, bundlePath, location, userDataDir }
}

function recordDeclinedMove(
  deps: Pick<MacApplicationsGuardDeps, 'log'>,
  bundlePath: string,
  location: MacUnsafeBundleLocation,
  userDataDir: string,
  remember: boolean
): MacApplicationsGuardResult {
  let remembered = false

  if (remember) {
    try {
      writeMacApplicationsGuardPreference(userDataDir)
      remembered = true
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)

      deps.log(`[install] could not save /Applications move preference: ${message}`)
    }
  }

  deps.log(
    `[install] user chose to keep running from ${location} (${bundlePath})${remembered ? '; remembered dismissal' : ''}`
  )

  return { action: 'continue', reason: 'user-declined', location, remembered }
}

export async function guardMacApplicationsLaunch(deps: MacApplicationsGuardDeps): Promise<MacApplicationsGuardResult> {
  let detectedLocation: MacUnsafeBundleLocation | undefined

  try {
    const check = precheckMacApplicationsGuard(deps)

    if ('result' in check) {
      return check.result
    }

    detectedLocation = check.location

    const { response, checkboxChecked } = await deps.dialog.showMessageBox(
      buildMovePromptOptions(deps.appName, check.location, deps.installStamp?.updateMechanism === 'electron-updater')
    )

    if (response !== 0) {
      return recordDeclinedMove(deps, check.bundlePath, check.location, check.userDataDir, Boolean(checkboxChecked))
    }

    return await attemptMoveToApplications(deps, check.bundlePath, check.location)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)

    deps.log(`[install] /Applications launch guard skipped: ${message}`)

    return {
      action: 'continue',
      reason: 'error',
      ...(detectedLocation ? { location: detectedLocation } : {}),
      error: message
    }
  }
}

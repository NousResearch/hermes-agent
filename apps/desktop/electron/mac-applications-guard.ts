import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import type { App, Dialog, MessageBoxOptions } from 'electron'

import type { InstallStamp } from './install-stamp'

export const MAC_APPLICATIONS_GUARD_FILENAME = 'mac-applications-guard.json'

export type MacUnsafeBundleLocation = 'app-translocation' | 'desktop-folder' | 'downloads-folder' | 'read-only-volume'

export interface MacApplicationsGuardPreference {
  schemaVersion: 1
  dismissed: true
  reason: MacUnsafeBundleLocation
  bundlePath: string
  appVersion: string
  dismissedAt: string
}

export type MacApplicationsGuardResult =
  | { action: 'relaunching'; location: MacUnsafeBundleLocation }
  | {
      action: 'continue'
      reason:
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
  readBundleVersion?: (bundlePath: string) => string | null
  log: (message: string) => void
}

const LOCATION_DETAILS: Readonly<Record<MacUnsafeBundleLocation, (appName: string) => string>> = {
  'app-translocation': appName =>
    `${appName} is running from a temporary macOS App Translocation folder (usually because it was opened from Downloads). ` +
    'Automatic updates cannot install from here, and command-line links will stop working after restart. ' +
    `Moving ${appName} to Applications fixes both.`,
  'desktop-folder': appName =>
    `${appName} is running from your Desktop folder. ` +
    'Moving it to Applications keeps automatic updates and command-line links working reliably.',
  'downloads-folder': appName =>
    `${appName} is running from your Downloads folder. ` +
    'Moving it to Applications keeps automatic updates and command-line links working reliably.',
  'read-only-volume': appName =>
    `${appName} is running from a read-only volume (such as a disk image). ` +
    'Automatic updates cannot install from here, and command-line links will stop working once the volume is ejected. ' +
    `Moving ${appName} to Applications fixes both.`
}

export function resolveMacAppBundle(exePath: string, platform: NodeJS.Platform = process.platform): string | null {
  if (platform !== 'darwin' || !exePath) {
    return null
  }

  let dir = path.posix.dirname(exePath)

  for (let i = 0; i < 2; i++) {
    dir = path.posix.dirname(dir)
  }

  return dir.endsWith('.app') ? dir : null
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
  const normalized = path.posix.resolve(bundlePath)

  if (normalized.includes('/AppTranslocation/')) {
    return 'app-translocation'
  }

  const access = opts?.access ?? fs.accessSync

  // EACCES (root-owned folder) is left to Squirrel's admin prompt; only EROFS means a read-only mount.
  for (const target of [path.posix.dirname(normalized), normalized]) {
    try {
      access(target, fs.constants.W_OK)
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'EROFS') {
        return 'read-only-volume'
      }
    }
  }

  // Only flag Downloads/Desktop; CI ShipIt relaunches run in writable runner temp dirs without CI env vars.
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
    const raw = fs.readFileSync(macApplicationsGuardPreferencePath(userDataDir), 'utf8')
    const parsed = JSON.parse(raw) as Partial<MacApplicationsGuardPreference>

    if (
      parsed &&
      typeof parsed === 'object' &&
      parsed.schemaVersion === 1 &&
      parsed.dismissed === true &&
      typeof parsed.reason === 'string' &&
      typeof parsed.bundlePath === 'string' &&
      typeof parsed.appVersion === 'string' &&
      typeof parsed.dismissedAt === 'string'
    ) {
      return parsed as MacApplicationsGuardPreference
    }
  } catch {
    return null
  }

  return null
}

export function writeMacApplicationsGuardPreference(
  userDataDir: string,
  preference: MacApplicationsGuardPreference
): void {
  const target = macApplicationsGuardPreferencePath(userDataDir)
  const tmp = `${target}.tmp-${process.pid}-${Date.now()}`

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
}

function parseBundleVersion(raw: string): ParsedBundleVersion | null {
  const cleaned = raw.trim().replace(/^v/i, '')
  const match = /^(\d+)\.(\d+)(?:\.(\d+))?(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/.exec(cleaned)

  if (!match) {
    return null
  }

  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: match[3] === undefined ? 0 : Number(match[3]),
    prerelease: match[4] ? match[4].split('.') : []
  }
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

  return comparePrerelease(a.prerelease, b.prerelease)
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

// CLI links into a translocated or read-only bundle dangle once macOS drops the translocation or the volume
// ejects. This is judged on the payload the links point into, whatever the launch guard decided; without a
// home dir, classifyMacBundleLocation reports only those two locations.
export function shouldProvisionBundledCliLinks(
  payloadRoot: string,
  log: (message: string) => void,
  platform: NodeJS.Platform = process.platform,
  access?: (file: string, mode: number) => void
): boolean {
  if (platform === 'win32') {
    return false
  }

  const location = platform === 'darwin' ? classifyMacBundleLocation(payloadRoot, { access }) : null

  if (location) {
    log(`[install] skipping ~/.local/bin CLI symlink provisioning while running from ${location}`)
  }

  return !location
}

interface InstalledCopyBlock {
  existingVersion: string | null
  newer: boolean
}

// Electron replaces whichever of /Applications or ~/Applications it picks, so every installed copy must be
// confirmed no newer than this one. A copy whose version can't be read or compared is never replaced.
function installedCopyBlockingMove(
  bundlePath: string,
  runningVersion: string,
  homeDir: string,
  readBundleVersion: (targetBundle: string) => string | null,
  exists: (file: string) => boolean
): InstalledCopyBlock | null {
  const bundleName = path.posix.basename(path.posix.resolve(bundlePath))
  const candidates = [path.posix.join('/Applications', bundleName)]

  if (homeDir) {
    candidates.push(path.posix.join(path.posix.resolve(homeDir), 'Applications', bundleName))
  }

  const installed = candidates.filter(candidate => exists(candidate))

  if (installed.length === 0) {
    return { existingVersion: null, newer: false }
  }

  for (const candidate of installed) {
    const existingVersion = readBundleVersion(candidate)
    const comparison = existingVersion ? compareBundleVersions(runningVersion, existingVersion) : null

    if (comparison === null || comparison < 0) {
      return { existingVersion, newer: comparison !== null }
    }
  }

  return null
}

function buildMovePromptOptions(appName: string, location: MacUnsafeBundleLocation): MessageBoxOptions {
  return {
    type: 'warning',
    title: `Move ${appName} to Applications`,
    message: `Move ${appName} to the Applications folder?`,
    detail: LOCATION_DETAILS[location](appName),
    buttons: ['Move to Applications', 'Keep running here'],
    defaultId: 0,
    cancelId: 1,
    checkboxLabel: "Don't ask again",
    checkboxChecked: false,
    noLink: true
  }
}

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

async function attemptMoveToApplications(
  deps: Pick<
    MacApplicationsGuardDeps,
    'app' | 'appName' | 'dialog' | 'exists' | 'homeDir' | 'log' | 'readBundleVersion'
  >,
  bundlePath: string,
  location: MacUnsafeBundleLocation
): Promise<MacApplicationsGuardResult> {
  const readBundleVersion = deps.readBundleVersion ?? (target => readBundleShortVersion(target))
  const exists = deps.exists ?? fs.existsSync
  const homeDir = deps.homeDir ?? os.homedir()
  let conflict: 'exists' | 'existsAndRunning' | null = null
  let blocked: { runningVersion: string; block: InstalledCopyBlock } | null = null

  try {
    const moved = deps.app.moveToApplicationsFolder({
      conflictHandler: conflictType => {
        conflict = conflictType

        if (conflictType === 'existsAndRunning') {
          return false
        }

        const runningVersion = readBundleVersion(bundlePath) ?? deps.app.getVersion()
        const block = installedCopyBlockingMove(bundlePath, runningVersion, homeDir, readBundleVersion, exists)

        blocked = block ? { runningVersion, block } : null

        return block === null
      }
    })

    if (moved) {
      deps.log(`[install] relocated ${deps.appName} into /Applications from ${location}; relaunching`)

      return { action: 'relaunching', location }
    }

    if (blocked) {
      const { runningVersion, block } = blocked

      return await reportInstalledCopyConflict(deps, location, runningVersion, block)
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
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)

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

  const translocated = bundlePath.includes('/AppTranslocation/')

  if (!translocated && deps.app.isInApplicationsFolder()) {
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
  deps: Pick<MacApplicationsGuardDeps, 'app' | 'log'>,
  bundlePath: string,
  location: MacUnsafeBundleLocation,
  userDataDir: string,
  remember: boolean
): MacApplicationsGuardResult {
  if (remember) {
    try {
      writeMacApplicationsGuardPreference(userDataDir, {
        schemaVersion: 1,
        dismissed: true,
        reason: location,
        bundlePath,
        appVersion: deps.app.getVersion(),
        dismissedAt: new Date().toISOString()
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)

      deps.log(`[install] could not save /Applications move preference: ${message}`)
    }
  }

  deps.log(
    `[install] user chose to keep running from ${location} (${bundlePath})${remember ? '; remembered dismissal' : ''}`
  )

  return { action: 'continue', reason: 'user-declined', location, remembered: remember }
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
      buildMovePromptOptions(deps.appName, check.location)
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

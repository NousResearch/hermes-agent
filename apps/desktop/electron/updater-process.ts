import { randomBytes } from 'node:crypto'
import { windowsPowerShellExecutable } from './windows-powershell-path'
import { queryWindowsProcessCreatedAt } from './windows-process-identity'
export const WINDOWS_HANDOFF_ENV = {
  branch: 'HERMES_UPDATE_HANDOFF_BRANCH',
  channel: 'HERMES_UPDATE_HANDOFF_CHANNEL',
  desktopPid: 'HERMES_UPDATE_HANDOFF_DESKTOP_PID',
  installRoot: 'HERMES_UPDATE_HANDOFF_INSTALL_ROOT',
  nonce: 'HERMES_UPDATE_HANDOFF_NONCE',
  relaunchAppPath: 'HERMES_UPDATE_HANDOFF_RELAUNCH_APP_PATH',
  relaunchExe: 'HERMES_UPDATE_HANDOFF_RELAUNCH_EXE',
  script: 'HERMES_UPDATE_HANDOFF_SCRIPT',
  /** "1" when a remote-served Desktop hands off: the script must not (re)start a local gateway (#117529). */
  noGateway: 'HERMES_UPDATE_HANDOFF_NO_GATEWAY'
} as const

const WINDOWS_HANDOFF_LAUNCHER = String.raw`
$ErrorActionPreference = 'Stop'
$required = @('HERMES_UPDATE_HANDOFF_SCRIPT','HERMES_UPDATE_HANDOFF_INSTALL_ROOT','HERMES_UPDATE_HANDOFF_BRANCH','HERMES_UPDATE_HANDOFF_DESKTOP_PID','HERMES_UPDATE_HANDOFF_RELAUNCH_EXE','HERMES_UPDATE_HANDOFF_NONCE')
foreach ($name in $required) { if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, 'Process'))) { throw "Missing required update handoff value: $name" } }
$desktopPid = 0
if (-not [int]::TryParse($env:HERMES_UPDATE_HANDOFF_DESKTOP_PID, [ref]$desktopPid) -or $desktopPid -le 0) { throw 'Invalid update handoff desktop PID' }
if (-not (Test-Path -LiteralPath $env:HERMES_UPDATE_HANDOFF_SCRIPT -PathType Leaf)) { throw 'Update handoff script is missing' }
$scriptArgs = @{ InstallRoot=$env:HERMES_UPDATE_HANDOFF_INSTALL_ROOT; Branch=$env:HERMES_UPDATE_HANDOFF_BRANCH; DesktopPid=$desktopPid; RelaunchExe=$env:HERMES_UPDATE_HANDOFF_RELAUNCH_EXE; HandoffNonce=$env:HERMES_UPDATE_HANDOFF_NONCE }
if (-not [string]::IsNullOrWhiteSpace($env:HERMES_UPDATE_HANDOFF_RELAUNCH_APP_PATH)) { $scriptArgs.RelaunchAppPath=$env:HERMES_UPDATE_HANDOFF_RELAUNCH_APP_PATH }
if (-not [string]::IsNullOrWhiteSpace($env:HERMES_UPDATE_HANDOFF_CHANNEL)) { $scriptArgs.Remove('Branch'); $scriptArgs.Channel=$env:HERMES_UPDATE_HANDOFF_CHANNEL }
if ($env:HERMES_UPDATE_HANDOFF_NO_GATEWAY -eq '1') { $scriptArgs.NoGateway=$true }
& $env:HERMES_UPDATE_HANDOFF_SCRIPT @scriptArgs
if ($null -eq $LASTEXITCODE) { exit 1 }
exit $LASTEXITCODE
`.trim()

/**
 * Per-handoff shared secret proving a marker claim came from the updater we
 * actually spawned (#B4).
 *
 * Before this, "authentication" only required that SOME process the Desktop
 * had not itself excluded wrote a plausible `<pid>\n<createdAt>` body into the
 * marker inside a ten-second window. Any same-user process satisfied it and
 * the Desktop then exited with no update running. The nonce travels to the
 * updater through the private handoff env block and comes back in an ack
 * sidecar, binding the claim to the exact child we started.
 */
export function createUpdateHandoffNonce(): string {
  return randomBytes(24).toString('hex')
}

const WINDOWS_HANDOFF_ENCODED_COMMAND = Buffer.from(WINDOWS_HANDOFF_LAUNCHER, 'utf16le').toString('base64')

import { spawn, type SpawnOptions, spawnSync, type SpawnSyncReturns } from 'node:child_process'
import { existsSync, realpathSync, statSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

/** Historical layouts still matter when uninstalling or migrating a source install. */
export function resolveVenvDir(updateRoot: string): string {
  for (const name of ['venv', '.venv']) {
    const candidate: string = path.join(updateRoot, name)

    try {
      if (statSync(candidate).isDirectory()) {
        return candidate
      }
    } catch {
      // Try the other supported layout, then retain the legacy diagnostic path.
    }
  }

  return path.join(updateRoot, 'venv')
}

import { platformDefaultHermesHome } from './data-paths'
import { hiddenWindowsChildOptions } from './windows-child-options'

/** Exact installation identity; PATH may refer to another checkout. */
export function resolveInstallationLauncher(
  updateRoot: string,
  isWindows: boolean = process.platform === 'win32',
  hermesHome: string = process.env.HERMES_HOME ?? ''
): string | null {
  const names: string[] = isWindows ? ['hermes.exe', 'hermes.cmd'] : ['hermes']

  for (const name of names) {
    const candidate: string = path.join(updateRoot, '.hermes', 'bin', name)

    if (stagedFileExists(candidate)) {
      return candidate
    }
  }

  // Earlier PM installers published only to user-bin. Trust that historical
  // launcher only after its existing version surface proves exact source identity.
  if (stagedFileExists(path.join(updateRoot, 'hermes_cli', '_launchers.py'))) {
    const extraDirs: string[] = isWindows ? [path.join(path.dirname(updateRoot), 'bin')] : []

    for (const candidate of userBinLaunchers(isWindows, hermesHome, extraDirs)) {
      if (launcherTargetsInstallation(candidate, updateRoot)) {
        return candidate
      }
    }
  }

  // An old shim is a migration rung, never a damaged PM install fallback.
  if (!existsSync(path.join(updateRoot, 'pm'))) {
    const legacy: string = path.join(updateRoot, 'venv', isWindows ? 'Scripts' : 'bin', names[0])

    if (stagedFileExists(legacy)) {
      return legacy
    }
  }

  return null
}

// cmd.exe re-parses its command line, so a launcher path carrying any of
// these would change the command instead of naming a file.
const CMD_UNSAFE_PATH: RegExp = /["%&|<>^\r\n]/

/** Published user-bin launchers, at fixed locations: a GUI launch's PATH may omit them. */
function userBinLaunchers(isWindows: boolean, hermesHome: string, extraDirs: string[] = []): string[] {
  const names: string[] = isWindows ? ['hermes.exe', 'hermes.cmd'] : ['hermes']
  const defaultHome: string = platformDefaultHermesHome(os.homedir(), process.env, isWindows ? 'win32' : 'linux')

  const dirs: string[] = isWindows
    ? [path.join(hermesHome || defaultHome, 'bin'), path.join(defaultHome, 'bin'), ...extraDirs]
    : [path.join(os.homedir(), '.local', 'bin'), path.join(hermesHome || defaultHome, 'bin'), ...extraDirs]

  return [...new Set(dirs)]
    .flatMap((dir: string): string[] => names.map((name: string): string => path.join(dir, name)))
    .filter(stagedFileExists)
}

/**
 * A source install outside the canonical root (install.sh --dir, a
 * setup-hermes.sh clone) is reachable only through the user-bin launcher it
 * published. Return the install directory that launcher reports, when it is a
 * Hermes source tree; the caller still resolves and probes it by root.
 */
export function userLauncherInstallRoot(
  isWindows: boolean = process.platform === 'win32',
  hermesHome: string = process.env.HERMES_HOME ?? ''
): { launcher: string; root: string } | null {
  for (const launcher of userBinLaunchers(isWindows, hermesHome)) {
    const root: string | null = launcherInstallDirectory(launcher)

    if (root && existsSync(path.join(root, 'hermes_cli', 'main.py'))) {
      return { launcher, root }
    }
  }

  return null
}

export function launcherTargetsInstallation(launcher: string, root: string): boolean {
  try {
    const reported: string | null = launcherInstallDirectory(launcher, root)

    return reported !== null && reported === realpathSync(root)
  } catch {
    return false
  }
}

/** The real install directory a launcher's `--version` reports, or null. */
function launcherInstallDirectory(launcher: string, root?: string): string | null {
  try {
    // Node refuses to exec a .cmd directly (CVE-2024-27980), and `shell:true`
    // would hand an interpolated path to cmd.exe wholesale. Invoke cmd.exe
    // explicitly instead: a fixed argv, the path quoted verbatim and screened
    // for cmd metacharacters, and nothing else for the shell to interpret.
    const viaCmd: boolean = process.platform === 'win32' && /\.cmd$/i.test(launcher)

    if (viaCmd && CMD_UNSAFE_PATH.test(launcher)) {
      return null
    }

    const command: string = viaCmd ? (process.env.ComSpec ?? 'cmd.exe') : launcher
    const args: string[] = viaCmd ? ['/d', '/s', '/c', `""${launcher}" --version"`] : ['--version']

    const probe: SpawnSyncReturns<string> = spawnSync(command, args, {
      cwd: root,
      encoding: 'utf8',
      timeout: 15000,
      windowsHide: true,
      windowsVerbatimArguments: viaCmd,
      env: root ? { ...process.env, HERMES_INSTALL_ROOT: root } : process.env
    })

    if (probe.error || probe.status !== 0) {
      return null
    }

    const reported: string | undefined = /^Install directory: (.+)$/m.exec(probe.stdout)?.[1]?.trim()

    return reported ? realpathSync(reported) : null
  } catch {
    return null
  }
}

/** File prerequisites only: dependency recovery remains reachable through update. */
export function windowsUpdatePrerequisiteError(updateRoot: string, hermesHome?: string): string | null {
  if (!resolveInstallationLauncher(updateRoot, true, hermesHome)) {
    return `Update aborted: the installation launcher under ${updateRoot} is missing. Repair this installation before retrying.`
  }

  const maintainedDir: string = path.join(updateRoot, 'scripts', 'desktop-update')

  if (existsSync(maintainedDir)) {
    for (const name of ['windows.ps1']) {
      const candidate: string = path.join(maintainedDir, name)

      if (!stagedFileExists(candidate)) {
        return `Update aborted: ${candidate} is missing or unreadable. Repair the installation and review antivirus quarantine before retrying.`
      }
    }
  }

  return null
}

export interface UpdaterChild {
  pid?: number
  exitCode?: number | null
  signalCode?: NodeJS.Signals | null
  kill?: (signal?: NodeJS.Signals | number) => boolean
  unref: () => void
}

export interface ResolveUpdateScriptHandoffDeps {
  isWindows?: boolean
  fileExists?: (candidate: string) => boolean
}

export interface UpdateScriptHandoff {
  command: string
  args: string[]
  scriptPath: string
}

export type WindowsUpdateTransport = { kind: 'script'; handoff: UpdateScriptHandoff } | { kind: 'manual' }

export interface WindowsUpdateHandoffValues {
  channel?: string
  branch: string
  desktopPid: number
  installRoot: string
  /** Secret from createUpdateHandoffNonce; the script echoes it in the ack. */
  nonce: string
  relaunchAppPath?: string
  relaunchExe: string
  /**
   * Remote-served Desktop (#117529): the hand-off must not let the update
   * (re)start a local messaging gateway that would compete with the remote
   * host's channel polling. Same rule as the posix hand-off.
   */
  noGateway?: boolean
}

export interface DetachedWindowsHandoff {
  command: string
  args: string[]
  env: Record<string, string>
  /** Always false: see wrapHandoffForDetachedConsole (#116161). */
  detached: false
}

export type WindowsUpdateLaunchResult =
  | { kind: 'manual' }
  | { kind: 'spawned'; child: UpdaterChild; handoff: UpdateScriptHandoff }

/**
 * Repo-owned Windows update hand-off (frozen-binary escape hatch).
 *
 * The staged Tauri `hermes-setup.exe` has no self-update path, so every
 * updater-side fix only reaches users when a new binary is built, signed and
 * published — which historically lags main by months and strands users on
 * long-fixed bugs (cache resolver #67369, marker self-adopt #74782; the
 * 2026-08-09 incident chain). `scripts/desktop-update/windows.ps1` lives in the repo
 * checkout instead: every `hermes update` refreshes the code that drives the
 * NEXT update, and only PowerShell itself is frozen.
 *
 * Returns the spawn recipe when the script exists in the checkout, or null
 * (caller falls back to the staged binary — old checkouts that predate the
 * script keep working unchanged). Windows-only by the same policy as
 * resolveStagedUpdaterBinary: POSIX updates in place via
 * applyUpdatesPosixInApp and needs no hand-off at all.
 */
export function resolveUpdateScriptHandoff(
  updateRoot: string,
  deps: ResolveUpdateScriptHandoffDeps = {}
): UpdateScriptHandoff | null {
  const isWindows = deps.isWindows ?? process.platform === 'win32'

  if (!isWindows) {
    return null
  }

  const exists = deps.fileExists ?? stagedFileExists

  // Current layout first, then the pre-reorg flat path — an updated asar can
  // meet a checkout from either side of the move (the checkout also ships a
  // forwarder at the legacy path for the inverse skew).
  for (const candidate of [
    path.join(updateRoot, 'scripts', 'desktop-update', 'windows.ps1'),
    path.join(updateRoot, 'scripts', 'desktop-update.ps1')
  ]) {
    if (exists(candidate)) {
      return {
        command: 'powershell',
        args: ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', candidate],
        scriptPath: candidate
      }
    }
  }

  return null
}

/**
 * Ordinary Windows updates must use the updater logic shipped with the live
 * checkout. The staged installer is frozen at build time and remains reserved
 * for bootstrap recovery, where the checkout may not yet provide a script.
 */
export function resolveWindowsUpdateTransport(
  updateRoot: string,
  deps: ResolveUpdateScriptHandoffDeps = {}
): WindowsUpdateTransport {
  const handoff = resolveUpdateScriptHandoff(updateRoot, deps)

  return handoff ? { kind: 'script', handoff } : { kind: 'manual' }
}

/**
 * Repo-owned POSIX update hand-off (the mac/linux twin of the above).
 *
 * Replaces the in-app posix updater: the Desktop spawns the script detached
 * and QUITS, the script waits it out, runs `hermes update`, swaps/relaunches
 * the app, and writes .hermes-update-result.json. With the app gone before
 * the update starts, the HERMES_DESKTOP_CHILD_PID reaper-exclusion dance is
 * unnecessary — there are no live desktop backends to spare.
 *
 * Null when the checkout predates the script (caller surfaces the manual
 * `hermes update` card — old checkouts pull the script on their next update).
 */
export function resolvePosixScriptHandoff(
  updateRoot: string,
  deps: ResolveUpdateScriptHandoffDeps = {}
): UpdateScriptHandoff | null {
  const isWindows = deps.isWindows ?? process.platform === 'win32'

  if (isWindows) {
    return null
  }

  const scriptPath = path.join(updateRoot, 'scripts', 'desktop-update', 'posix.sh')
  const exists = deps.fileExists ?? stagedFileExists

  if (!exists(scriptPath)) {
    return null
  }

  return {
    command: '/bin/bash',
    args: [scriptPath],
    scriptPath
  }
}

/**
 * Wrap a PowerShell hand-off invocation so it survives a hidden spawn from
 * Electron without ever showing a console window (#116161).
 *
 * Verified empirically (2026-08-09, Windows 11): `spawn('powershell', [...,
 * '-File', script], { detached: true, stdio: 'ignore', windowsHide: true })`
 * exits 0 WITHOUT executing a single line of the script. powershell.exe is a
 * console-subsystem binary; libuv maps `detached: true` to DETACHED_PROCESS,
 * which gives the child NO console (and makes the OS ignore CREATE_NO_WINDOW),
 * and Windows PowerShell 5.1 dies during console init before -File processing
 * (the same class of failure as #54220's conhost work, on the launch side).
 *
 * The parent-console model that follows from that (and that every other
 * hidden spawn in this app relies on): a console child inherits its parent's
 * console; only a console-LESS parent forces the OS to allocate a new,
 * visible one. So the wrapper cmd.exe is spawned NON-detached — libuv then
 * honours `windowsHide` (CREATE_NO_WINDOW) and cmd.exe owns one hidden
 * console — and `start "" /b powershell ...` runs the script inside that
 * hidden console (`/min` would tell `start` to allocate a NEW console for the
 * child, which is what flashed a minimized PowerShell window on every
 * hand-off; `/b` shares the wrapper's). The console outlives cmd.exe for as
 * long as powershell is attached to it.
 *
 * Survival past our own exit does not need `detached`: libuv's per-process
 * job object has JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK, so grandchildren
 * (`start`'s powershell) are never members and Windows does not tie a
 * process's lifetime to its parent. `start` returns immediately, so the
 * spawned pid is the WRAPPER's — callers must not use it as a marker owner
 * (the script claims the marker itself with its own $PID).
 */
/* eslint-disable no-redeclare */
export function wrapHandoffForDetachedConsole(handoff: UpdateScriptHandoff, extraArgs: string[]): { command: string; args: string[]; detached: false }
export function wrapHandoffForDetachedConsole(handoff: UpdateScriptHandoff, values: WindowsUpdateHandoffValues): DetachedWindowsHandoff

export function wrapHandoffForDetachedConsole(
  handoff: UpdateScriptHandoff,
  extraArgsOrValues: string[] | WindowsUpdateHandoffValues
): { command: string; args: string[]; env?: Record<string, string>; detached?: false } {
  if (Array.isArray(extraArgsOrValues)) {
    return {
      command: 'cmd.exe',
      args: ['/d', '/s', '/c', 'start', '', '/b', handoff.command, ...handoff.args, ...extraArgsOrValues],
      detached: false
    }
  }

  if (!Number.isSafeInteger(extraArgsOrValues.desktopPid) || extraArgsOrValues.desktopPid <= 0) {
    throw new Error('Windows update handoff requires a positive desktop PID')
  }

  if ([handoff.scriptPath, extraArgsOrValues.installRoot, extraArgsOrValues.branch, extraArgsOrValues.relaunchExe,
    extraArgsOrValues.nonce]
    .some(value => typeof value !== 'string' || value.trim().length === 0)) {
    throw new Error('Windows update handoff requires every environment value')
  }

  return {
    command: 'cmd.exe',
    args: [
      '/d',
      '/s',
      '/c',
      'start',
      '',
      '/b',
      windowsPowerShellExecutable(),
      '-NoProfile',
      '-NonInteractive',
      '-ExecutionPolicy',
      'Bypass',
      '-EncodedCommand',
      WINDOWS_HANDOFF_ENCODED_COMMAND
    ],
    env: {
      [WINDOWS_HANDOFF_ENV.branch]: extraArgsOrValues.branch,
      [WINDOWS_HANDOFF_ENV.channel]: extraArgsOrValues.channel ?? '',
      [WINDOWS_HANDOFF_ENV.desktopPid]: String(extraArgsOrValues.desktopPid),
      [WINDOWS_HANDOFF_ENV.installRoot]: extraArgsOrValues.installRoot,
      [WINDOWS_HANDOFF_ENV.nonce]: extraArgsOrValues.nonce,
      [WINDOWS_HANDOFF_ENV.relaunchAppPath]: extraArgsOrValues.relaunchAppPath ?? '',
      [WINDOWS_HANDOFF_ENV.relaunchExe]: extraArgsOrValues.relaunchExe,
      [WINDOWS_HANDOFF_ENV.script]: handoff.scriptPath,
      ...(extraArgsOrValues.noGateway ? { [WINDOWS_HANDOFF_ENV.noGateway]: '1' } : {})
    },
    detached: false
  }
}
/* eslint-enable no-redeclare */

/**
 * Electron/Chromium internal switches that must NOT be replayed on re-exec:
 * runtime artifacts of THIS launch, not user intent (ported from the deleted
 * update-relaunch.ts; #45205). `--no-sandbox` is deliberately kept — it is
 * the user's sandbox opt-out and the signal that makes a relaunch safe when
 * chrome-sandbox isn't setuid.
 */
export const INTERNAL_ARG_PREFIXES = [
  '--type=',
  '--user-data-dir=',
  '--enable-features=',
  '--disable-features=',
  '--field-trial-handle=',
  '--enable-logging',
  '--log-file=',
  '--disable-gpu-sandbox',
  '--lang=',
  '--inspect',
  '--remote-debugging-port='
]

/** Filter Electron internals from process.argv.slice(1) so the relaunched
 * app replays only user/launcher intent (deep links, app flags). */
export function collectRelaunchArgs(argv: unknown): string[] {
  if (!Array.isArray(argv)) {
    return []
  }

  return argv.filter((arg): arg is string => {
    if (typeof arg !== 'string' || arg.length === 0) {
      return false
    }

    return !INTERNAL_ARG_PREFIXES.some(prefix =>
      prefix.endsWith('=') ? arg.startsWith(prefix) : arg === prefix || arg.startsWith(prefix + '=')
    )
  })
}

/** True when the user has opted out of the SUID sandbox — the relaunch is
 * safe even if chrome-sandbox fails preflight (ported from update-relaunch.ts). */
export function sandboxFallbackFromEnv(env: Record<string, string | undefined>, launchArgs: string[]): boolean {
  const disable = String(env?.ELECTRON_DISABLE_SANDBOX || '').trim()

  if (disable === '1' || disable.toLowerCase() === 'true') {
    return true
  }

  return Array.isArray(launchArgs) && launchArgs.includes('--no-sandbox')
}

export interface ResolveStagedUpdaterBinaryDeps {
  stagedMtimeMs?: (candidate: string) => number | null
  isWindows?: boolean
  fileExists?: (candidate: string) => boolean
}

function stagedFileExists(candidate: string): boolean {
  try {
    return statSync(candidate).isFile()
  } catch {
    return false
  }
}

/**
 * Decide which staged installer binary — if any — may be handed an update.
 *
 * The Tauri installer self-copies into HERMES_HOME on *every* platform
 * (`hermes-setup.exe` on Windows, `hermes-setup` elsewhere — see
 * apps/bootstrap-installer `paths::installer_dest` and
 * `bootstrap::copy_self_to_hermes_home`), so finding that binary on macOS or
 * Linux is expected, not leftover junk.
 *
 * Handing an update to it is nonetheless a Windows-only policy. Windows needs
 * the quit -> hand-off -> rebuild dance because a venv shim file lock keeps the
 * running desktop from rewriting its own bits; macOS and Linux have no such
 * lock and update in place through applyUpdatesPosixInApp(). Off Windows the
 * hand-off therefore buys nothing and costs a great deal: a staged binary older
 * than the hand-off protocol holds the update marker, spawns `hermes update`,
 * and that child refuses its own parent — wedging the in-app Update button for
 * good, with no route (update, re-download, reinstall) to a newer binary
 * (#74836). Returning null off Windows is what routes those platforms to the
 * in-app updater.
 *
 * Null on Windows too when nothing is staged (a dev/source run, or a CLI
 * install that never went through the installer); callers degrade gracefully.
 */
export function resolveStagedUpdaterBinary(
  hermesHome: string,
  deps: ResolveStagedUpdaterBinaryDeps = {}
): string | null {
  const isWindows = deps.isWindows ?? process.platform === 'win32'

  if (!isWindows) {
    return null
  }

  const fileExists = deps.fileExists ?? stagedFileExists
  const candidate = path.join(hermesHome, 'hermes-setup.exe')

  return fileExists(candidate) ? candidate : null
}

export interface SpawnUpdaterProcessDeps {
  isWindows?: boolean
  spawnProcess?: (command: string, args: string[], options: SpawnOptions) => UpdaterChild
}

/**
 * Spawn the detached installer used for update and bootstrap-recovery handoffs.
 * The helper owns both hidden-console selection and unref semantics so every
 * updater handoff follows the same behavior and can be tested without Electron.
 */
export function spawnUpdaterProcess(
  updater: string,
  updaterArgs: string[],
  options: SpawnOptions,
  deps: SpawnUpdaterProcessDeps = {}
): UpdaterChild {
  const isWindows = deps.isWindows ?? process.platform === 'win32'
  const spawnOptions = hiddenWindowsChildOptions(options, isWindows) as SpawnOptions

  const child = deps.spawnProcess
    ? deps.spawnProcess(updater, updaterArgs, spawnOptions)
    : spawn(updater, updaterArgs, spawnOptions)

  child.unref()

  return child
}

export function resolveWindowsDevRelaunchAppPath(defaultApp: boolean, appPath: string): string | undefined {
  return defaultApp && appPath ? path.win32.resolve(appPath) : undefined
}

export function formatPowerShellArgvForDisplay(argv: string[]): string {
  return argv.map(value => (/^[A-Za-z0-9._/:-]+$/.test(value) ? value : `'${value.replaceAll("'", "''")}'`)).join(' ')
}

export function launchWindowsUpdateTransport(
  transport: WindowsUpdateTransport,
  values: WindowsUpdateHandoffValues,
  options: SpawnOptions,
  deps: SpawnUpdaterProcessDeps = {}
): WindowsUpdateLaunchResult {
  if (transport.kind === 'manual') {
    return transport
  }

  const wrapped = wrapHandoffForDetachedConsole(transport.handoff, values)

  // The wrapper must be NON-detached whatever the caller asked for: a
  // DETACHED_PROCESS cmd.exe owns no console, so `start /b` would make
  // powershell allocate a new visible one (#116161) or die in console init.
  const child = spawnUpdaterProcess(
    wrapped.command,
    wrapped.args,
    { ...options, detached: wrapped.detached, env: { ...options.env, ...wrapped.env } },
    deps
  )

  return { kind: 'spawned', child, handoff: transport.handoff }
}


export interface WindowsProcessCreatedAtDeps {
  queryCreatedAt?: (pid: number) => Promise<number | null>
}

export async function captureSpawnedUpdaterCreatedAt(
  pid: number,
  { queryCreatedAt = queryWindowsProcessCreatedAt }: WindowsProcessCreatedAtDeps = {}
): Promise<number | null> {
  if (!Number.isSafeInteger(pid) || pid <= 0) {
    return null
  }

  try {
    const createdAt = await queryCreatedAt(pid)

    return typeof createdAt === 'number' && Number.isFinite(createdAt) && createdAt > 0 ? createdAt : null
  } catch {
    return null
  }
}

export function isSpawnedUpdaterGenerationActive(child: UpdaterChild): boolean {
  try {
    return Boolean(child.kill?.(0))
  } catch {
    return false
  }
}

export async function terminateSpawnedUpdaterIfExact(
  child: UpdaterChild,
  expectedCreatedAt: number,
  { queryCreatedAt = queryWindowsProcessCreatedAt }: WindowsProcessCreatedAtDeps = {}
): Promise<boolean> {
  if (!Number.isSafeInteger(child.pid) || !Number.isSafeInteger(expectedCreatedAt) || expectedCreatedAt <= 0) {
    return false
  }

  if ((await captureSpawnedUpdaterCreatedAt(Number(child.pid), { queryCreatedAt })) !== expectedCreatedAt) {
    return false
  }

  try {
    return Boolean(child.kill?.())
  } catch {
    return false
  }
}

export interface UpdaterHandoffOutcome {
  ok: boolean
  /** Set when ok is false. */
  reason?: 'spawn-error' | 'early-exit'
  /** Human-readable detail for logs (never contains argv secrets). */
  message?: string
  /** Exit code when the child exited inside the settle window. */
  code?: number | null
  /** Signal when the child was killed inside the settle window. */
  signal?: string | null
}

export interface ObserveUpdaterHandoffDeps {
  setTimeoutFn?: (callback: () => void, ms: number) => unknown
  clearTimeoutFn?: (timer: unknown) => void
}

/**
 * User-facing copy for a hand-off that did not take (spawn error or early exit).
 * The lead sentence is plain: nothing changed and Hermes keeps running. The raw
 * outcome message (exit code / signal / spawn error) stays on a trailing
 * "Details:" line for logs and support.
 */
export function describeUpdaterHandoffFailure(outcome: Pick<UpdaterHandoffOutcome, 'message'>): string {
  const lead =
    "The updater couldn't start, so nothing was changed and Hermes keeps running as before. " +
    'Try again; if it keeps failing, open the logs and send them to support.'

  return outcome.message ? `${lead}\n\nDetails: ${outcome.message}` : lead
}

/**
 * Watch a just-spawned detached updater for the duration of the quit dwell
 * and report whether the hand-off actually became viable (#66753).
 *
 * Before this, the Desktop called `unref()` and quit after a fixed dwell
 * without ever observing the child's async `error` event (ENOENT/EACCES —
 * Node reports exec failures asynchronously) or an early `exit`. A failed
 * spawn therefore looked identical to a successful one: the app vanished, no
 * updater appeared, and nothing relaunched. Worse, an unhandled `'error'`
 * event on the detached child would crash the Electron main process outright.
 *
 * Success is: no `error` event AND either the child survives the settle
 * window or it exits 0 inside it (the Windows `cmd start` wrapper exits 0
 * immediately by design — see wrapHandoffForDetachedConsole). Failure is a
 * spawn `error`, a non-zero exit, or a signal death inside the window.
 *
 * Children that expose no event interface (bare test doubles) settle as ok
 * after the window — the observation is a best-effort hardening, never a new
 * way to wedge an update.
 */
export function observeUpdaterHandoff(
  child: UpdaterChild,
  settleMs: number,
  deps: ObserveUpdaterHandoffDeps = {}
): Promise<UpdaterHandoffOutcome> {
  const setTimeoutFn = deps.setTimeoutFn ?? setTimeout

  const clearTimeoutFn =
    deps.clearTimeoutFn ?? ((timer: unknown) => clearTimeout(timer as ReturnType<typeof setTimeout>))

  const observable = child as UpdaterChild & {
    once?: (event: string, listener: (...args: unknown[]) => void) => unknown
    removeListener?: (event: string, listener: (...args: unknown[]) => void) => unknown
  }

  if (typeof observable.once !== 'function') {
    return new Promise(resolve => {
      setTimeoutFn(() => resolve({ ok: true }), settleMs)
    })
  }

  return new Promise(resolve => {
    let settled = false

    const finish = (outcome: UpdaterHandoffOutcome) => {
      if (settled) {
        return
      }

      settled = true
      clearTimeoutFn(timer)
      observable.removeListener?.('error', onError)
      observable.removeListener?.('exit', onExit)
      resolve(outcome)
    }

    const onError = (...args: unknown[]) => {
      const error = args[0] as (Error & { code?: string }) | undefined

      finish({
        ok: false,
        reason: 'spawn-error',
        message: `updater spawn failed: ${error?.code || error?.message || 'unknown error'}`
      })
    }

    const onExit = (...args: unknown[]) => {
      const code = args[0] as number | null
      const signal = args[1] as string | null

      if (signal || (typeof code === 'number' && code !== 0)) {
        finish({
          ok: false,
          reason: 'early-exit',
          message: signal
            ? `updater died from signal ${signal} before the settle window elapsed`
            : `updater exited ${code} before the settle window elapsed`,
          code: code ?? null,
          signal: signal ?? null
        })

        return
      }

      // Clean exit 0 inside the window is expected for wrapper shapes
      // (cmd.exe `start` on Windows exits immediately after launching the
      // real script in its own console).
      finish({ ok: true, code: code ?? 0, signal: null })
    }

    const timer = setTimeoutFn(() => finish({ ok: true }), settleMs)

    observable.once('error', onError)
    observable.once('exit', onExit)
  })
}

export function stagedUpdaterSupportsPrewrittenMarker(
  candidate: string,
  deps: ResolveStagedUpdaterBinaryDeps = {}
): boolean {
  const mtimeMs = (deps.stagedMtimeMs ?? stagedFileMtimeMs)(candidate)

  return typeof mtimeMs === 'number' && Number.isFinite(mtimeMs) && mtimeMs >= MARKER_SELF_ADOPT_EPOCH_MS
}


export const MARKER_SELF_ADOPT_EPOCH_MS = Date.UTC(2026, 6, 31)
function stagedFileMtimeMs(candidate: string): number | null {
  try { return statSync(candidate).mtimeMs } catch { return null }
}

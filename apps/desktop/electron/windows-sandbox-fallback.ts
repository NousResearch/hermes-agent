/**
 * Windows Chromium/Electron sandbox recovery for #38216.
 *
 * On some Windows hosts the GPU/renderer sandboxes die with STATUS_BREAKPOINT
 * (`0x80000003` / exit `-2147483645`). Chromium then FATAL-exits
 * ("GPU process isn't usable. Goodbye.") before the UI is usable.
 *
 * Recovery ladder, all scoped to win32:
 *
 * 1. ACL repair (first line): grant `S-1-15-2-2` (ALL APPLICATION PACKAGES)
 *    RX on the install tree. A missing ACE plus orphan AppContainer SIDs is a
 *    known Chromium CHECK failure (electron/electron#51761). Runs at install
 *    time, and again at launch ONLY when the marker shows a prior aborted
 *    boot — never on healthy launches (icacls /T recursion is not free).
 * 2. `--no-sandbox` (second line): enabled only on strong evidence —
 *    a signature-confirmed GPU/renderer breakpoint death, or TWO consecutive
 *    mid-boot aborts (a single abort can be a task-manager kill or power
 *    loss; the reported failure mode is a deterministic 100% crash loop).
 * 3. The fallback is sticky per app version, not forever: after an update
 *    the sandbox is re-probed once (a new Electron or an installer-applied
 *    ACL grant may have fixed the host). If the re-probe boot aborts, the
 *    next launch goes straight back to `--no-sandbox`.
 *
 * Pure helpers stay injectable so tests never boot Electron or touch real ACLs.
 */

import fs from 'node:fs'
import path from 'node:path'

export const WINDOWS_SANDBOX_MARKER_FILENAME = 'windows-sandbox-fallback.json'

/** Well-known SID for "ALL APPLICATION PACKAGES". */
export const ALL_APPLICATION_PACKAGES_SID = 'S-1-15-2-2'

/** STATUS_BREAKPOINT as a signed Win32 exit code (WER / Chromium). */
export const WINDOWS_SANDBOX_BREAKPOINT_EXIT = -2147483645

/** Consecutive mid-boot aborts required before enabling --no-sandbox. */
export const BOOT_ABORTS_BEFORE_FALLBACK = 2

export type SandboxMarkerState = 'booting' | 'fallback' | 'ok'

export type SandboxFallbackReason = 'gpu-breakpoint' | 'renderer-crash-loop' | 'boot-loop'

export interface SandboxMarker {
  state: SandboxMarkerState
  /** Why the fallback engaged (state === 'fallback'). */
  reason?: SandboxFallbackReason
  /** App version that entered fallback — a version change triggers a re-probe. */
  version?: string
  /** Consecutive aborted boots observed so far (state === 'booting'). */
  bootAborts?: number
  /** This boot is a sandbox re-probe after an app update; an abort returns
   *  straight to fallback instead of restarting the two-strike count. */
  reprobe?: boolean
}

export function sandboxMarkerPath(userDataDir: string): string {
  return path.join(String(userDataDir || ''), WINDOWS_SANDBOX_MARKER_FILENAME)
}

export function isWindowsSandboxBreakpointExit(exitCode: unknown): boolean {
  const n = Number(exitCode)

  if (!Number.isFinite(n)) {
    return false
  }

  // Signed STATUS_BREAKPOINT, or the same 32-bit pattern as unsigned.
  return n === WINDOWS_SANDBOX_BREAKPOINT_EXIT || n >>> 0 === 0x80000003
}

export function alreadyHasNoSandbox(argv: readonly string[] = [], env: NodeJS.ProcessEnv = process.env): boolean {
  if (Array.isArray(argv) && argv.some(arg => arg === '--no-sandbox')) {
    return true
  }

  const disable = String(env.ELECTRON_DISABLE_SANDBOX || '')
    .trim()
    .toLowerCase()

  return disable === '1' || disable === 'true' || disable === 'yes' || disable === 'on'
}

const FALLBACK_REASONS: readonly string[] = ['gpu-breakpoint', 'renderer-crash-loop', 'boot-loop']

export function parseSandboxMarker(raw: unknown): SandboxMarker | null {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const record = raw as Record<string, unknown>
  const state = record.state

  if (state !== 'booting' && state !== 'fallback' && state !== 'ok') {
    return null
  }

  const marker: SandboxMarker = { state }

  if (typeof record.reason === 'string' && FALLBACK_REASONS.includes(record.reason)) {
    marker.reason = record.reason as SandboxFallbackReason
  }

  if (typeof record.version === 'string' && record.version) {
    marker.version = record.version
  }

  const aborts = Number(record.bootAborts)

  if (Number.isInteger(aborts) && aborts > 0) {
    marker.bootAborts = aborts
  }

  if (record.reprobe === true) {
    marker.reprobe = true
  }

  return marker
}

export function readSandboxMarker(userDataDir: string, { readFileSync = fs.readFileSync } = {}): SandboxMarker | null {
  try {
    const raw = JSON.parse(readFileSync(sandboxMarkerPath(userDataDir), 'utf8'))

    return parseSandboxMarker(raw)
  } catch {
    return null
  }
}

export function writeSandboxMarker(
  userDataDir: string,
  marker: SandboxMarker,
  {
    mkdirSync = fs.mkdirSync,
    writeFileSync = fs.writeFileSync
  }: {
    mkdirSync?: typeof fs.mkdirSync
    writeFileSync?: typeof fs.writeFileSync
  } = {}
): void {
  const dir = String(userDataDir || '')

  if (!dir) {
    return
  }

  mkdirSync(dir, { recursive: true })
  writeFileSync(sandboxMarkerPath(dir), `${JSON.stringify(marker)}\n`, 'utf8')
}

export interface SandboxLaunchDecision {
  enable: boolean
  reason: string | null
  /** Marker to persist immediately, before GPU/sandbox children start. */
  nextMarker: SandboxMarker
}

/**
 * Single launch-time transition: decide whether this Windows launch disables
 * the Chromium sandbox AND what the marker becomes for crash-detection on the
 * next launch.
 *
 * - `booting` left behind → the prior launch aborted mid-boot. One abort is
 *   tolerated (could be a kill/power loss); the SECOND consecutive abort — or
 *   a single abort during a post-update re-probe — engages the fallback.
 * - `fallback` is sticky within one app version. A version change re-probes
 *   the sandbox once so a fixed host (new Electron, installer ACL repair)
 *   returns to full sandboxing instead of degrading forever.
 * - A manual `--no-sandbox` / ELECTRON_DISABLE_SANDBOX launch is honored but
 *   NOT made sticky: the marker keeps its normal lifecycle so the flag's
 *   removal restores the sandbox.
 */
export function decideWindowsSandboxLaunch(
  options: {
    platform?: NodeJS.Platform | string
    argv?: readonly string[]
    env?: NodeJS.ProcessEnv
    marker?: SandboxMarker | null
    appVersion?: string
  } = {}
): SandboxLaunchDecision {
  const appVersion = String(options.appVersion || '')

  if ((options.platform ?? process.platform) !== 'win32') {
    return { enable: false, reason: null, nextMarker: { state: 'booting' } }
  }

  const argv = options.argv ?? process.argv
  const env = options.env ?? process.env
  const marker = options.marker ?? null

  if (alreadyHasNoSandbox(argv, env)) {
    // Honor the explicit flag; keep the marker lifecycle unchanged. When the
    // relaunch path set the flag, the fallback marker it wrote is preserved.
    const nextMarker: SandboxMarker = marker?.state === 'fallback' ? marker : { state: 'booting' }

    return { enable: true, reason: 'already-enabled', nextMarker }
  }

  if (marker?.state === 'fallback') {
    if (marker.version && appVersion && marker.version !== appVersion) {
      // App updated since the fallback engaged — re-probe the sandbox once.
      return {
        enable: false,
        reason: null,
        nextMarker: { state: 'booting', reprobe: true, bootAborts: 0 }
      }
    }

    return {
      enable: true,
      reason: 'sticky-fallback',
      nextMarker: { ...marker, version: marker.version || appVersion || undefined }
    }
  }

  if (marker?.state === 'booting') {
    const abortsObserved = (marker.bootAborts ?? 0) + 1

    if (marker.reprobe) {
      // The one post-update sandboxed re-probe aborted → back to fallback.
      return {
        enable: true,
        reason: 'reprobe-failed',
        nextMarker: fallbackMarker('boot-loop', appVersion)
      }
    }

    if (abortsObserved >= BOOT_ABORTS_BEFORE_FALLBACK) {
      return {
        enable: true,
        reason: 'boot-loop',
        nextMarker: fallbackMarker('boot-loop', appVersion)
      }
    }

    return {
      enable: false,
      reason: null,
      nextMarker: { state: 'booting', bootAborts: abortsObserved }
    }
  }

  // No marker, or a clean `ok` from the previous run.
  return { enable: false, reason: null, nextMarker: { state: 'booting' } }
}

export function fallbackMarker(reason: SandboxFallbackReason, appVersion?: string): SandboxMarker {
  const marker: SandboxMarker = { state: 'fallback', reason }

  if (appVersion) {
    marker.version = appVersion
  }

  return marker
}

/**
 * After the main window reaches ready-to-show: keep the sticky fallback when
 * we launched with `--no-sandbox`, otherwise mark a clean boot so future
 * launches trust the sandbox again.
 */
export function markerAfterSuccessfulBoot(options: {
  fallbackActive: boolean
  reason?: SandboxFallbackReason
  appVersion?: string
}): SandboxMarker {
  if (!options.fallbackActive) {
    return { state: 'ok' }
  }

  return fallbackMarker(options.reason ?? 'boot-loop', options.appVersion)
}

/**
 * ACL repair is not free (`icacls /T` recurses the whole install tree), so it
 * only runs when there is evidence of trouble: a prior launch aborted
 * mid-boot, or the fallback already engaged. Healthy hosts never pay for it —
 * the installer already granted the ACE at install time.
 */
export function shouldAttemptAclRepair(marker: SandboxMarker | null | undefined): boolean {
  return marker?.state === 'booting' || marker?.state === 'fallback'
}

/**
 * Build `icacls` argv that grants ALL APPLICATION PACKAGES RX with inheritance.
 * `/T` applies to existing children (win-unpacked DLLs); `/C` continues on
 * errors; `/Q` stays quiet for installer logs.
 */
export function buildIcaclsGrantArgs(targetDir: string): string[] {
  return [String(targetDir), '/grant', `*${ALL_APPLICATION_PACKAGES_SID}:(OI)(CI)(RX)`, '/T', '/C', '/Q']
}

export function grantAllApplicationPackagesAcl(
  targetDir: string,
  {
    platform = process.platform,
    execFileSync
  }: {
    platform?: NodeJS.Platform | string
    execFileSync?: (file: string, args: readonly string[], options?: object) => Buffer | string
  } = {}
): { ok: boolean; error?: string } {
  if (platform !== 'win32') {
    return { ok: false }
  }

  const dir = String(targetDir || '').trim()

  if (!dir || typeof execFileSync !== 'function') {
    return { ok: false, error: 'missing-target-or-exec' }
  }

  try {
    execFileSync('icacls', buildIcaclsGrantArgs(dir), {
      windowsHide: true,
      timeout: 30_000,
      stdio: 'ignore'
    })

    return { ok: true }
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error)
    }
  }
}

/**
 * True when a GPU child died with the #38216 breakpoint signature and we
 * should one-shot relaunch with `--no-sandbox` before Chromium FATAL-exits.
 */
export function shouldRelaunchForGpuSandboxCrash(options: {
  platform?: NodeJS.Platform | string
  details?: { type?: string; exitCode?: number | string } | null
  alreadyNoSandbox?: boolean
  relaunchAttempted?: boolean
}): boolean {
  if ((options.platform ?? process.platform) !== 'win32') {
    return false
  }

  if (options.alreadyNoSandbox || options.relaunchAttempted) {
    return false
  }

  const type = String(options.details?.type || '').toLowerCase()

  if (type !== 'gpu') {
    return false
  }

  return isWindowsSandboxBreakpointExit(options.details?.exitCode)
}

/**
 * True when a renderer crash loop carries the sandbox breakpoint signature
 * and a one-shot `--no-sandbox` relaunch should replace the dead window
 * (#38216 renderer flavor; same recovery as #56726). Gated on the breakpoint
 * exit code so unrelated renderer crash loops (bad extension, OOM churn)
 * don't silently drop the sandbox.
 */
export function shouldRelaunchForRendererSandboxCrashLoop(options: {
  platform?: NodeJS.Platform | string
  reason?: string
  exitCode?: number | string
  alreadyNoSandbox?: boolean
  relaunchAttempted?: boolean
}): boolean {
  if ((options.platform ?? process.platform) !== 'win32') {
    return false
  }

  if (options.alreadyNoSandbox || options.relaunchAttempted) {
    return false
  }

  if (String(options.reason || '') !== 'crashed') {
    return false
  }

  return isWindowsSandboxBreakpointExit(options.exitCode)
}

export function buildNoSandboxRelaunchArgs(argv: readonly string[]): string[] {
  const args = (Array.isArray(argv) ? argv : []).filter(arg => arg !== '--no-sandbox')

  args.push('--no-sandbox')

  return args
}

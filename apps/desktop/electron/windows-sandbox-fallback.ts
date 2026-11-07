/**
 * Windows Chromium/Electron sandbox recovery for #38216.
 *
 * On some Windows hosts the GPU/renderer sandboxes die with STATUS_BREAKPOINT
 * (`0x80000003` / exit `-2147483645`). Chromium then FATAL-exits
 * ("GPU process isn't usable. Goodbye.") before the UI is usable.
 *
 * Recovery ladder:
 *
 * 1. ACL repair (Windows only): grant `S-1-15-2-2` (ALL APPLICATION PACKAGES)
 *    RX on the install tree. A missing ACE plus orphan AppContainer SIDs is a
 *    known Chromium CHECK failure (electron/electron#51761). Runs at install
 *    time, and again at launch ONLY when the marker shows a prior aborted
 *    boot — never on healthy launches (icacls /T recursion is not free).
 * 2. `--no-sandbox` (second line): enabled only on strong evidence —
 *    a signature-confirmed GPU/renderer breakpoint death, or TWO consecutive
 *    mid-boot aborts (a single abort can be a task-manager kill or power
 *    loss; the reported failure mode is a deterministic 100% crash loop).
 *    An abort only counts inside the machine boot that wrote the witness: a
 *    `booting` marker from an earlier boot is the remains of an unclean
 *    shutdown, not evidence about this boot's sandbox.
 *    On Linux (#121954) the evidence signal is the GPU child dying with
 *    SIGTERM — Chromium's own "GPU process isn't usable. Goodbye." shutdown
 *    for a GPU process that never came up — which the host matrix pinned to
 *    the sandboxed GPU child (only `--no-sandbox` survives; `--disable-gpu`
 *    crashes too, and the GPU child dies pre-main on an FD-ownership
 *    violation before any Chromium init).
 * 3. The fallback is sticky per app version and build, not forever: after an
 *    update or a rebuild the sandbox is re-probed once (a new Electron, an
 *    installer-applied ACL grant, or a kernel/driver update may have fixed the
 *    host). If the re-probe boot aborts, the next launch goes straight back to
 *    `--no-sandbox`.
 * 4. Windows-only extras (renderer crash-loop relaunch, ACL repair) stay
 *    win32-gated: Linux has no ACL model, and its renderer crash loops carry
 *    no sandbox-specific signature.
 *
 * Pure helpers stay injectable so tests never boot Electron or touch real ACLs.
 *
 * Pure helpers stay injectable so tests never boot Electron or touch real ACLs.
 */

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { launchMarkerNeedsReprobe } from './launch-build-identity'

export const WINDOWS_SANDBOX_MARKER_FILENAME = 'windows-sandbox-fallback.json'

/**
 * Exit status Chromium uses to SIGTERM its GPU process when it never became
 * usable (#121954: sandbox-blocked GPU child dies pre-main on an
 * FD-ownership violation, browser prints "GPU process isn't usable.
 * Goodbye." and SIGTERMs it). Node reports signal deaths as `null` exitCode
 * + `signalName: 'SIGTERM'`; Electron serializes that as exitCode 143.
 */
export const GPU_CHILD_SANDBOX_SIGTERM_EXIT = 143

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
  /** Machine boot that wrote this witness (`currentBootIdentity()`). A
   *  `booting` marker from another boot is stale: it records an unclean
   *  shutdown, not an aborted boot of this one. */
  boot?: string
  /** Build identity that wrote this marker (`launchBuildIdentity()`). A
   *  rebuild or update re-probes the fallback once, so a re-stamped source
   *  install that reports version `0.0.0` does not stay degraded forever. An
   *  unstamped dev bundle has no identity to compare and stays sticky. */
  build?: string
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

  if (typeof record.boot === 'string' && record.boot) {
    marker.boot = record.boot
  }

  if (typeof record.build === 'string' && record.build) {
    marker.build = record.build
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

/**
 * Identity of the machine boot a witness was written in.
 *
 * Linux exposes a stable boot id. Everywhere else the boot instant derived
 * from `os.uptime()` is constant apart from wall-clock steps, which
 * `isSameBootWitness()` tolerates up to BOOT_INSTANT_TOLERANCE_SECONDS; a
 * larger step is read as a new boot and costs one launch before the ladder
 * re-arms. The `at:` form is best-effort: it carries no host discriminator,
 * so a userData directory shared between machines is not supported, and a
 * host whose clock does not advance across boots (dead RTC, no time sync)
 * produces the same instant every boot, which reads as the same boot and
 * counts the abort. A readable /proc boot_id avoids all of that.
 */
export function currentBootIdentity(
  options: {
    platform?: NodeJS.Platform | string
    readFileSync?: (file: string, encoding: 'utf8') => string
    nowMs?: number
    uptimeSeconds?: number
  } = {}
): string {
  if ((options.platform ?? process.platform) === 'linux') {
    const readFileSync = options.readFileSync ?? ((file: string, encoding: 'utf8') => fs.readFileSync(file, encoding))

    try {
      const id = String(readFileSync('/proc/sys/kernel/random/boot_id', 'utf8')).trim()

      if (id) {
        return `linux:${id}`
      }
    } catch {
      // No boot id (container without /proc, hardened host): fall through.
    }
  }

  const nowMs = Number(options.nowMs ?? Date.now())
  const uptimeSeconds = Number(options.uptimeSeconds ?? os.uptime())

  if (!Number.isFinite(nowMs) || !Number.isFinite(uptimeSeconds)) {
    return ''
  }

  return `at:${Math.round(nowMs / 1000 - uptimeSeconds)}`
}

/** Seconds of slack allowed between two `at:` boot instants (wall-clock derived). */
export const BOOT_INSTANT_TOLERANCE_SECONDS = 2

/**
 * True when two boot identities describe the same machine boot.
 *
 * Linux ids are exact kernel values. The `at:` fallback is derived from the
 * wall clock, so an NTP correction, a manual clock change or uptime tick
 * quantisation can shift it without a reboot: those compare with a tolerance,
 * so a live same-boot witness is not mistaken for a stale one (which would
 * silently reset the two-strike ladder). A step larger than the tolerance is
 * read as a new boot, and mixed forms (one side `linux:`, the other `at:` -
 * a transient /proc failure) deliberately compare false: stale keeps the
 * sandbox on, which is the safe direction.
 */
export function isSameBootWitness(markerBoot: string, currentBoot: string): boolean {
  if (!markerBoot || !currentBoot) {
    return false
  }

  if (markerBoot === currentBoot) {
    return true
  }

  const instant = (value: string): number => (value.startsWith('at:') ? Number(value.slice(3)) : Number.NaN)
  const markerAt = instant(markerBoot)
  const currentAt = instant(currentBoot)

  if (!Number.isFinite(markerAt) || !Number.isFinite(currentAt)) {
    return false
  }

  return Math.abs(markerAt - currentAt) <= BOOT_INSTANT_TOLERANCE_SECONDS
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
 * - A `booting` marker whose `boot` isn't the current machine boot is stale:
 *   the launch that wrote it never reached a verdict, so it cannot accuse the
 *   sandbox. It is dropped instead of counted (a power cut must not arm the
 *   `--no-sandbox` step, which on Linux user-namespace hosts kills every
 *   renderer).
 * - `fallback` is sticky within one app version and build. A version or build
 *   change re-probes the sandbox once so a fixed host (new Electron, installer
 *   ACL repair, rebuild) returns to full sandboxing instead of degrading
 *   forever.
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
    /** `currentBootIdentity()` of this launch; enables the stale-witness rule. */
    bootIdentity?: string
    /** `launchBuildIdentity()` of this launch; enables the re-probe rule. */
    buildIdentity?: string
  } = {}
): SandboxLaunchDecision {
  const appVersion = String(options.appVersion || '')
  const bootIdentity = String(options.bootIdentity || '')
  const buildIdentity = String(options.buildIdentity || '')

  // The two-strike boot-abort ladder is platform-neutral (both win32 #38216
  // and linux #121954); the platform-specific helpers gate themselves below.
  const launchPlatform = options.platform ?? process.platform

  if (launchPlatform !== 'win32' && launchPlatform !== 'linux') {
    return { enable: false, reason: null, nextMarker: { state: 'booting' } }
  }

  const argv = options.argv ?? process.argv
  const env = options.env ?? process.env
  const marker = options.marker ?? null

  const booting = (bootAborts?: number, extra: Partial<SandboxMarker> = {}): SandboxMarker => {
    const next: SandboxMarker = { state: 'booting', ...extra }

    if (bootAborts !== undefined) {
      next.bootAborts = bootAborts
    }

    if (bootIdentity) {
      next.boot = bootIdentity
    }

    if (buildIdentity) {
      next.build = buildIdentity
    }

    return next
  }

  if (alreadyHasNoSandbox(argv, env)) {
    // Honor the explicit flag; keep the marker lifecycle unchanged. When the
    // relaunch path set the flag, the fallback marker it wrote is preserved.
    const nextMarker: SandboxMarker = marker?.state === 'fallback' ? marker : booting()

    return { enable: true, reason: 'already-enabled', nextMarker }
  }

  if (marker?.state === 'fallback') {
    // Re-probe when the app version OR the built artifact changed: source
    // installs report `0.0.0` for every build, so a version-only comparison
    // would pin a fallback forever.
    if (launchMarkerNeedsReprobe(marker, { appVersion, buildIdentity })) {
      return { enable: false, reason: null, nextMarker: booting(0, { reprobe: true }) }
    }

    return {
      enable: true,
      reason: 'sticky-fallback',
      nextMarker: {
        ...marker,
        version: marker.version || appVersion || undefined,
        ...(buildIdentity ? { build: buildIdentity } : {})
      }
    }
  }

  if (marker?.state === 'booting') {
    // A witness written in an earlier machine boot never reached a verdict:
    // the host was powered off, hard-reset or the kernel died mid-launch. It
    // says nothing about this boot's sandbox, so it must not arm the
    // `--no-sandbox` step (fatal for renderers on user-namespace hosts). A
    // pending post-update re-probe keeps its intent: if it was interrupted by
    // the shutdown rather than by the sandbox, the next same-boot abort has to
    // return to the fallback instead of restarting the ladder.
    if (marker.boot && bootIdentity && !isSameBootWitness(marker.boot, bootIdentity)) {
      return { enable: false, reason: null, nextMarker: booting(undefined, marker.reprobe ? { reprobe: true } : {}) }
    }

    const abortsObserved = (marker.bootAborts ?? 0) + 1

    if (marker.reprobe) {
      // The one post-update sandboxed re-probe aborted → back to fallback.
      return {
        enable: true,
        reason: 'reprobe-failed',
        nextMarker: fallbackMarker('boot-loop', appVersion, buildIdentity)
      }
    }

    if (abortsObserved >= BOOT_ABORTS_BEFORE_FALLBACK) {
      return {
        enable: true,
        reason: 'boot-loop',
        nextMarker: fallbackMarker('boot-loop', appVersion, buildIdentity)
      }
    }

    return { enable: false, reason: null, nextMarker: booting(abortsObserved) }
  }

  // No marker, or a clean `ok` from the previous run.
  return { enable: false, reason: null, nextMarker: booting() }
}

export function fallbackMarker(
  reason: SandboxFallbackReason,
  appVersion?: string,
  buildIdentity?: string
): SandboxMarker {
  const marker: SandboxMarker = { state: 'fallback', reason }

  if (appVersion) {
    marker.version = appVersion
  }

  if (buildIdentity) {
    marker.build = buildIdentity
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
  buildIdentity?: string
}): SandboxMarker {
  if (!options.fallbackActive) {
    return { state: 'ok' }
  }

  return fallbackMarker(options.reason ?? 'boot-loop', options.appVersion, options.buildIdentity)
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
  details?: { type?: string; exitCode?: number | string; signalName?: string } | null
  alreadyNoSandbox?: boolean
  relaunchAttempted?: boolean
}): boolean {
  const platform = options.platform ?? process.platform

  if (options.alreadyNoSandbox || options.relaunchAttempted) {
    return false
  }

  if (String(options.details?.type || '').toLowerCase() !== 'gpu') {
    return false
  }

  if (platform === 'win32') {
    return isWindowsSandboxBreakpointExit(options.details?.exitCode)
  }

  if (platform === 'linux') {
    // #121954: the sandbox-blocked GPU child never comes up; Chromium
    // SIGTERMs it (exit 143) right before its own FATAL "Goodbye." abort.
    // Deliberately narrow: GPU-only, needs the explicit signature.
    return (
      options.details?.exitCode === GPU_CHILD_SANDBOX_SIGTERM_EXIT &&
      String(options.details?.signalName || '').toUpperCase() === 'SIGTERM'
    )
  }

  return false
}

/**
 * True when a renderer crash loop carries the sandbox breakpoint signature
 * and a one-shot `--no-sandbox` relaunch should replace the dead window
 * (#38216 renderer flavor; same recovery as #56726). Windows only: Linux
 * renderer crash loops carry no sandbox-specific exit signature, so
 * dropping the sandbox on one would be a guess.
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

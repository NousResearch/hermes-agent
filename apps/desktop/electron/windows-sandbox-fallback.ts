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
 *    a signature-confirmed GPU/renderer breakpoint death, or repeated
 *    mid-boot aborts (a single abort can be a task-manager kill or power
 *    loss; the reported failure mode is a deterministic 100% crash loop).
 *    Two consecutive aborts are the threshold, except after an update, where
 *    the one-shot re-probe only needs to abort once.
 *    On Linux (#121954) the evidence signal is the GPU child dying with
 *    SIGTERM — Chromium's own "GPU process isn't usable. Goodbye." shutdown
 *    for a GPU process that never came up — which the host matrix pinned to
 *    the sandboxed GPU child (only `--no-sandbox` survives; `--disable-gpu`
 *    crashes too, and the GPU child dies pre-main on an FD-ownership
 *    violation before any Chromium init).
 * 3. The fallback is sticky per app version, not forever: after an update
 *    the sandbox is re-probed once (a new Electron, an installer-applied
 *    ACL grant, or a kernel/driver update may have fixed the host). If the
 *    re-probe boot aborts, the next launch goes straight back to
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
import path from 'node:path'

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

export type SandboxMarkerState = 'booting' | 'fallback' | 'ok' | 'running'

export type SandboxFallbackReason = 'gpu-breakpoint' | 'renderer-crash-loop' | 'boot-loop'

export interface SandboxMarker {
  state: SandboxMarkerState
  /** Why the fallback engaged (state === 'fallback'). */
  reason?: SandboxFallbackReason
  /** App version that entered fallback — a version change triggers a re-probe. */
  version?: string
  /** Consecutive aborted boots observed so far (state === 'booting'). Absent
   *  means zero: `parseSandboxMarker` drops a non-positive count, so an explicit
   *  `0` written by decideWindowsSandboxLaunch does not survive a round trip
   *  through disk — every read already normalizes with `?? 0`. */
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

  if (state !== 'booting' && state !== 'fallback' && state !== 'ok' && state !== 'running') {
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
  /** A `running` leftover: the prior run reached a usable window, then died
   *  without a clean quit (main-process abort, task-manager kill, power
   *  loss). Steady-state evidence, never boot evidence. */
  priorSteadyAbort?: boolean
}

/**
 * Does this launch still OWE the sandbox its one allowed post-update retry?
 *
 * Both halves are required. The version-change arm arms `reprobe` for the
 * sandboxed re-probe launch itself (`enable: false`, sandbox ON) — and that
 * launch spending the retry by reaching a window is the entire point of arming
 * it. So the retry only survives when this launch owed it AND could not spend
 * it, i.e. when we also ran with the sandbox off.
 *
 * The caller owns this answer today, which is how an earlier version of this
 * branch ended up keying the reveal on "did we run without the sandbox" alone:
 * a manual `hermes --no-sandbox` satisfies that too, fabricating a retry on a
 * healthy host and latching it into `--no-sandbox` after one ordinary abort.
 */
export function launchStillOwesReprobe(decision: SandboxLaunchDecision): boolean {
  return decision.nextMarker.reprobe === true && decision.enable === true
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

  // The two-strike boot-abort ladder is platform-neutral (both win32 #38216
  // and linux #121954); the platform-specific helpers gate themselves below.
  const launchPlatform = options.platform ?? process.platform

  if (launchPlatform !== 'win32' && launchPlatform !== 'linux') {
    return { enable: false, reason: null, nextMarker: { state: 'booting' } }
  }

  const argv = options.argv ?? process.argv
  const env = options.env ?? process.env
  const marker = options.marker ?? null

  if (alreadyHasNoSandbox(argv, env)) {
    // Honor the explicit flag; keep the marker lifecycle unchanged. When the
    // relaunch path set the flag, the fallback marker it wrote is preserved.
    //
    // A pending post-update `reprobe` also survives: this launch runs with the
    // sandbox already off, so it is not the process that gets to use the retry.
    // The process that actually exercises the sandbox comes later, and it must
    // still find the retry armed or a still-broken sandbox gets a free extra
    // attempt (two aborts to reach fallback instead of one). Carried from the
    // same two states recordDirectCleanExit accepts, so the three sites that
    // preserve a pending retry cannot drift apart.
    let nextMarker: SandboxMarker
    if (marker?.state === 'fallback') {
      nextMarker = marker
    } else if (
      (marker?.state === 'booting' || marker?.state === 'running') &&
      marker.reprobe === true
    ) {
      nextMarker = { state: 'booting', reprobe: true }
    } else {
      nextMarker = { state: 'booting' }
    }

    return { enable: true, reason: 'already-enabled', nextMarker }
  }

  if (marker?.state === 'fallback') {
    if (marker.version && appVersion && marker.version !== appVersion) {
      // App updated since the fallback engaged — re-probe the sandbox once. The
      // explicit zero restates the fresh budget this launch starts from; it is
      // dropped on the next disk round trip by parseSandboxMarker either way.
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

  if (marker?.state === 'running') {
    // The prior run reached a usable window and then died without before-quit
    // (#112961 main-process abort, task-manager kill, power loss). This is
    // steady-state evidence, not boot evidence: it must not count toward the
    // boot-loop fallback and must not trigger ACL repair.
    //
    // A `reprobe` here means the window came up on a launch that owed a retry
    // but could not exercise the sandbox (see markerAfterSuccessfulBoot's
    // `pendingReprobe`), so the retry is still owed and moves to this launch.
    // Whether that launch really was owed one is the caller's decision; this
    // arm only carries the flag forward.
    const nextMarker: SandboxMarker = { state: 'booting' }

    if (marker.reprobe === true) {
      nextMarker.reprobe = true
    }

    return { enable: false, reason: null, nextMarker, priorSteadyAbort: true }
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
 *
 * `steady` (window-reveal path) records `running` instead of `ok`: a later
 * main-process abort then leaves a `running` leftover behind, so the next
 * launch can tell a mid-session death (#112961) from a clean quit. Both
 * clean-exit handlers now go through `recordDirectCleanExit()` instead, which
 * decides between `ok` and a preserved `reprobe`.
 */
export function markerAfterSuccessfulBoot(options: {
  fallbackActive: boolean
  reason?: SandboxFallbackReason
  appVersion?: string
  steady?: boolean
  pendingReprobe?: boolean
}): SandboxMarker {
  if (!options.fallbackActive) {
    if (options.steady) {
      const marker: SandboxMarker = { state: 'running' }

      if (options.appVersion) {
        marker.version = options.appVersion
      }

      // A post-update `reprobe` survives only on a reveal that could not spend
      // it. A reveal that DID run sandboxed is the re-probe succeeding, which
      // legitimately consumes the retry, so `running` is written plain there.
      // The caller owns that distinction: pass `pendingReprobe` only when this
      // launch owed a retry AND ran with the sandbox off.
      if (options.pendingReprobe) {
        marker.reprobe = true
      }

      return marker
    }

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

/**
 * Record a direct clean exit into the sandbox marker file.
 *
 * `app.exit()` never emits `before-quit`, so the clean `ok` write in that
 * handler does not run for an in-app relaunch or a GPU/renderer fallback
 * relaunch — the `running` marker window reveal wrote would stay behind and the
 * next launch would read it as a steady abort (#112961 instrumentation).
 *
 * This owns BOTH the decision and the write, so the seam is exercisable: a test
 * of a pure decision alone cannot see whether main.ts routes its clean exits
 * here. `writeSandboxMarker` injects its own fs calls.
 *
 * Keyed on sticky (not active) to match the before-quit handler: an engaged
 * fallback keeps its sticky marker. The platform gate belongs to the callers,
 * exactly as the reveal and startup writes gate theirs — the marker itself is
 * not platform-specific (both #38216 and #121954 write and read it).
 *
 * A clean `app.exit()` is a deliberate relaunch, so it spends no evidence: the
 * startup `booting` marker is written before any window exists, so this run may
 * never have revealed one, and a clean exit must not advance the two-strike
 * `bootAborts` budget, so no strike is recorded at all.
 *
 * The one thing that IS carried forward is a pending post-update `reprobe`.
 * That flag does not describe THIS process — it describes the NEXT one: it is
 * the sandbox's one allowed retry after an app update, and if a clean relaunch
 * happened between arming it and the process that actually exercises the
 * sandbox, then discarding it would silently hand a still-broken sandbox an
 * extra free attempt (it would need a second abort to reach fallback instead of
 * one). So a `booting`+`reprobe` leftover stays armed and just stops counting
 * aborts; every other prior state becomes plain `ok`.
 */
export function recordDirectCleanExit(
  userDataDir: string,
  options: {
    stickyFallback: boolean
  }
): void {
  if (options.stickyFallback) {
    return
  }

  // Only `reprobe` describes the next process rather than this one, so it is
  // the single field a clean exit carries forward. It can be pending on either
  // `booting` (relaunch before a window ever appeared) or `running` (a window
  // appeared, but on a boot that ran with the sandbox already off, so it
  // proves nothing). See the docstring.
  const prior = readSandboxMarker(userDataDir)
  const pendingReprobe =
    (prior?.state === 'booting' || prior?.state === 'running') && prior.reprobe === true

  writeSandboxMarker(
    userDataDir,
    pendingReprobe
      ? { state: 'booting', reprobe: true }
      : { state: 'ok' }
  )
}

export function buildNoSandboxRelaunchArgs(argv: readonly string[]): string[] {
  const args = (Array.isArray(argv) ? argv : []).filter(arg => arg !== '--no-sandbox')

  args.push('--no-sandbox')

  return args
}

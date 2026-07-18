/**
 * Windows Chromium/Electron sandbox recovery for #38216.
 *
 * On some Windows hosts the GPU/renderer sandboxes die with STATUS_BREAKPOINT
 * (`0x80000003` / exit `-2147483645`). Chromium then FATAL-exits
 * ("GPU process isn't usable. Goodbye.") before the UI is usable.
 *
 * Two complementary recoveries, both scoped to win32:
 *
 * 1. Grant `S-1-15-2-2` (ALL APPLICATION PACKAGES) RX on the install /
 *    userData trees. Missing that ACE plus orphan AppContainer SIDs is a
 *    known Chromium CHECK failure (electron/electron#51761).
 * 2. Sticky boot marker: write `booting` before sandbox bring-up; clear it
 *    only on a clean ready. An uncleared marker on the next launch means the
 *    previous process aborted mid-boot → enable `--no-sandbox` (the only flag
 *    reporters verified as fully stable for AMD RX 6000 / hybrid GPU hosts).
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

export type SandboxMarkerState = 'booting' | 'fallback' | 'ok'

export interface SandboxMarker {
  state: SandboxMarkerState
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
  return n === WINDOWS_SANDBOX_BREAKPOINT_EXIT || (n >>> 0) === 0x80000003
}

export function alreadyHasNoSandbox(
  argv: readonly string[] = [],
  env: NodeJS.ProcessEnv = process.env
): boolean {
  if (Array.isArray(argv) && argv.some(arg => arg === '--no-sandbox')) {
    return true
  }

  const disable = String(env.ELECTRON_DISABLE_SANDBOX || '')
    .trim()
    .toLowerCase()

  return disable === '1' || disable === 'true' || disable === 'yes' || disable === 'on'
}

export function parseSandboxMarker(raw: unknown): SandboxMarker | null {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const state = (raw as { state?: unknown }).state

  if (state === 'booting' || state === 'fallback' || state === 'ok') {
    return { state }
  }

  return null
}

export function readSandboxMarker(
  userDataDir: string,
  { readFileSync = fs.readFileSync } = {}
): SandboxMarker | null {
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
 * Decide whether this Windows launch should disable the Chromium sandbox.
 *
 * `booting` left from a prior launch → previous process aborted before ready.
 * `fallback` → we already recovered once; keep the workaround sticky so the
 * Start Menu shortcut does not crash every other launch.
 */
export function shouldEnableWindowsNoSandbox(options: {
  platform?: NodeJS.Platform | string
  argv?: readonly string[]
  env?: NodeJS.ProcessEnv
  marker?: SandboxMarker | null
} = {}): { enable: boolean; reason: string | null } {
  if ((options.platform ?? process.platform) !== 'win32') {
    return { enable: false, reason: null }
  }

  const argv = options.argv ?? process.argv
  const env = options.env ?? process.env

  if (alreadyHasNoSandbox(argv, env)) {
    return { enable: true, reason: 'already-enabled' }
  }

  const state = options.marker?.state

  if (state === 'booting') {
    return { enable: true, reason: 'uncleared-boot-marker' }
  }

  if (state === 'fallback') {
    return { enable: true, reason: 'sticky-fallback' }
  }

  return { enable: false, reason: null }
}

/**
 * Marker to persist immediately after the launch decision, before GPU/sandbox
 * child processes start. Successful boots rewrite this later.
 */
export function nextSandboxMarkerAfterLaunchDecision(options: {
  enabledNoSandbox: boolean
}): SandboxMarker {
  if (options.enabledNoSandbox) {
    return { state: 'fallback' }
  }

  return { state: 'booting' }
}

/**
 * After the main window reaches ready-to-show: keep sticky fallback when we
 * launched with `--no-sandbox`, otherwise mark a clean boot so the next
 * launch can try the sandbox again (e.g. after an ACL grant fixed the host).
 */
export function markerAfterSuccessfulBoot(options: { fallbackActive: boolean }): SandboxMarker {
  return options.fallbackActive ? { state: 'fallback' } : { state: 'ok' }
}

/**
 * Build `icacls` argv that grants ALL APPLICATION PACKAGES RX with inheritance.
 * `/T` applies to existing children (win-unpacked DLLs); `/C` continues on
 * errors; `/Q` stays quiet for installer logs.
 */
export function buildIcaclsGrantArgs(targetDir: string): string[] {
  return [
    String(targetDir),
    '/grant',
    `*${ALL_APPLICATION_PACKAGES_SID}:(OI)(CI)(RX)`,
    '/T',
    '/C',
    '/Q'
  ]
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

export function buildNoSandboxRelaunchArgs(argv: readonly string[]): string[] {
  const args = (Array.isArray(argv) ? argv : []).filter(arg => arg !== '--no-sandbox')

  args.push('--no-sandbox')

  return args
}

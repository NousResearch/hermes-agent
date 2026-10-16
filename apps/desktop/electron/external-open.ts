/**
 * external-open.ts
 *
 * The single route every external URL open in the desktop app flows through.
 * Electron-free: every side effect (shell.openExternal, spawn, the file
 * opener, the failure channel) is injected, so the "open failed → notify"
 * behavior is unit-tested without loading electron — the same shape as
 * native-oauth-login.ts and connection-config.ts. main.ts wires the real deps.
 *
 * A URL is opened exactly once. An open failure is logged and reported
 * through notifyFailure (the renderer shows a fallback modal with the URL);
 * the caller reads the result to decide whether an open-failure must also
 * abort its own flow (the native-OAuth path fails fast, link paths don't).
 *
 * Every open is bounded by OPEN_TIMEOUT_MS. Electron settles openExternal and
 * openPath only once the xdg-open it spawns exits, so a desktop-portal
 * handshake that never completes leaves them pending forever. A caller that
 * awaits that — main.ts's openPreviewInBrowser handler does — then never sends
 * its IPC reply, and the renderer reports "reply was never sent" instead of the
 * fallback modal, which reads as a dead control.
 */

import type { ChildProcess, SpawnOptions } from 'node:child_process'

import { absolutizeProtocolRelativeUrl, looksLikeLocalFilesystemPath } from './local-filesystem-path'

export type ExternalOpenResult =
  { ok: true } | { ok: false; reason: 'invalid' } | { ok: false; reason: 'failed'; message: string }

export interface FileOpenGuardDeps {
  log: (line: string) => void
  reportMissing: (rawUrl: string, message: string) => void
}

/**
 * Decide what a pre-open stat failure means for a file open (#122027). Lives
 * here, not inline in main.ts, so the classification is unit-tested without
 * loading electron — same injected-deps shape as the rest of the module.
 *
 * Returns true when the failure was a MISS: it has been reported through
 * reportMissing and the caller must NOT hand the path to the OS (a reveal of
 * a non-existent path is silently a no-op on macOS; an open of one reads as
 * "No application found" on LaunchServices). Returns false for every other
 * stat failure (EACCES on a locked volume, ELOOP, Windows EPERM): those are
 * logged and the caller still proceeds to the OS, so an existing-but-locked
 * file keeps its real error and a stat failure never fabricates a miss.
 */
export function reportPreOpenStatFailure(error: unknown, rawUrl: string, deps: FileOpenGuardDeps): boolean {
  if (error && typeof error === 'object' && (error as { code?: string }).code === 'missing-file') {
    deps.reportMissing(rawUrl, externalOpenErrorMessage(error))

    return true
  }

  deps.log(`[file] pre-open stat failed: ${externalOpenErrorMessage(error)}`)

  return false
}

export interface ExternalOpenDeps {
  isWsl: boolean
  spawn: (cmd: string, args: readonly string[], opts: SpawnOptions) => ChildProcess
  openExternal: (url: string) => Promise<void>
  openFile: (rawUrl: string) => Promise<void>
  /**
   * Open a BARE local filesystem path (POSIX `/…`, `~/…`, Windows drive/UNC)
   * that is not a URL. The impl resolves it through the same audited
   * `resolveRequestedPathForIpc` the `file:` route uses, then `shell.openPath`,
   * with the same reveal-in-folder fallback and missing-file reporting as the
   * file route. Resolves false when the path could not be resolved at all
   * (rejected syntax, blocked device path) so the caller logs and reports it.
   */
  openLocalPath: (rawPath: string) => Promise<boolean>
  notifyFailure: (url: string, message: string) => void
  log: (line: string) => void
}

const SUPPORTED_WEB = ['http:', 'https:', 'mailto:']

/**
 * Generous: an ordinary open returns in well under a second. This only has to be
 * long enough that a slow-but-real browser start is not called a failure.
 */
export const OPEN_TIMEOUT_MS = 8000

/**
 * A distinct type, not a plain Error: the file: route swallows open *failures*
 * (main already reveals the file in the file manager) but a timeout has had no
 * fallback applied anywhere, so it must be surfaced.
 */
class OpenTimeoutError extends Error {
  constructor() {
    super(`the system opener did not respond within ${OPEN_TIMEOUT_MS / 1000}s`)
    this.name = 'OpenTimeoutError'
  }
}

const OPEN_TIMED_OUT = Symbol('open-timed-out')

/**
 * Resolve with `work`'s value, or reject with OpenTimeoutError once `ms` have
 * passed. Rejections from `work` propagate unchanged, so existing failure
 * handling is untouched — only a hang becomes an error.
 */
async function withDeadline<T>(work: Promise<T>, ms: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined

  const deadline = new Promise<typeof OPEN_TIMED_OUT>(resolve => {
    timer = setTimeout(() => resolve(OPEN_TIMED_OUT), ms)
  })

  try {
    const winner = await Promise.race([work, deadline])

    if (winner === OPEN_TIMED_OUT) {
      // The deadline won, so nothing is listening to `work` any more. Attach a
      // handler so a late rejection cannot surface as an unhandled rejection.
      void work.catch(() => {})

      throw new OpenTimeoutError()
    }

    return winner
  } finally {
    clearTimeout(timer)
  }
}

export function externalOpenErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/**
 * Open a URL in the system browser. Resolves `invalid` for a URL the route
 * does not open (empty, malformed, unsupported scheme). NEVER rejects: an open
 * failure is logged + reported through notifyFailure and resolves `failed`,
 * so fire-and-forget callers can `void` the result safely.
 */
export async function openExternalUrl(rawUrl: string, deps: ExternalOpenDeps): Promise<ExternalOpenResult> {
  const raw = absolutizeProtocolRelativeUrl(String(rawUrl || '').trim())

  if (!raw) {
    return { ok: false, reason: 'invalid' }
  }

  // Bare local paths (POSIX `/…`, `~/…`, Windows drive/UNC) are not valid
  // absolute URLs: `new URL()` either throws (POSIX, `~`, UNC) or mis-parses a
  // drive letter as a bogus `c:` scheme, which the web allowlist below would
  // reject as "Invalid external URL". Route them through the audited file
  // resolver instead (hermes-agent 80946, 84361).
  if (looksLikeLocalFilesystemPath(raw)) {
    let opened: boolean

    try {
      opened = await deps.openLocalPath(raw)
    } catch (error) {
      return failOpen(deps, raw, error)
    }

    if (!opened) {
      deps.log(`[file] openPath resolve rejected: path=${raw}`)

      return { ok: false, reason: 'invalid' }
    }

    return { ok: true }
  }

  let parsed: URL

  try {
    parsed = new URL(raw)
  } catch {
    return { ok: false, reason: 'invalid' }
  }

  if (parsed.protocol === 'file:') {
    try {
      await withDeadline(deps.openFile(raw), OPEN_TIMEOUT_MS)
    } catch (error) {
      // main's openFile handles its own fallback; a failure is never surfaced
      // here. A timeout is the exception: nothing else is holding the caller's
      // promise open, so it has to resolve.
      if (error instanceof OpenTimeoutError) {
        return failOpen(deps, raw, error)
      }
    }

    return { ok: true }
  }

  if (!SUPPORTED_WEB.includes(parsed.protocol)) {
    return { ok: false, reason: 'invalid' }
  }

  const url = parsed.toString()

  if (deps.isWsl) {
    return openViaWsl(url, deps)
  }

  try {
    await withDeadline(deps.openExternal(url), OPEN_TIMEOUT_MS)

    return { ok: true }
  } catch (error) {
    return failOpen(deps, url, error)
  }
}

async function openViaWsl(url: string, deps: ExternalOpenDeps): Promise<ExternalOpenResult> {
  deps.log(`[link] opening via WSL→Windows: ${url}`)

  const proc = deps.spawn('cmd.exe', ['/c', 'start', '""', url], {
    detached: true,
    stdio: 'ignore',
    windowsHide: true
  })

  // 'error' only fires when the process could not be spawned. In that case
  // fall back to xdg-open; if that also fails, surface it. The handler runs
  // asynchronously after this function has already resolved.
  proc.on('error', error => {
    deps.log(`[link] cmd.exe start failed: ${error.message}; falling back to xdg-open`)

    deps.openExternal(url).catch(openError => {
      failOpen(deps, url, openError)
    })
  })

  try {
    proc.unref()
  } catch {
    // unref can throw on an already-closed handle in some node versions
  }

  return { ok: true }
}

function failOpen(deps: ExternalOpenDeps, url: string, error: unknown): ExternalOpenResult {
  const message = externalOpenErrorMessage(error)
  deps.log(`[link] openExternal failed: ${message}`)
  deps.notifyFailure(url, message)

  return { ok: false, reason: 'failed', message }
}

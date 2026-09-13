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
 */

import type { ChildProcess, SpawnOptions } from 'node:child_process'

export type ExternalOpenResult =
  | { ok: true }
  | { ok: false; reason: 'invalid' }
  | { ok: false; reason: 'failed'; message: string }

export interface ExternalOpenDeps {
  isWsl: boolean
  spawn: (cmd: string, args: readonly string[], opts: SpawnOptions) => ChildProcess
  openExternal: (url: string) => Promise<void>
  openFile: (rawUrl: string) => Promise<void>
  notifyFailure: (url: string, message: string) => void
  log: (line: string) => void
}

const SUPPORTED_WEB = ['http:', 'https:', 'mailto:']

export function externalOpenErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/**
 * Open a URL in the system browser. Resolves `invalid` for a URL the route
 * does not open (empty, malformed, unsupported scheme). NEVER rejects: an open
 * failure is logged + reported through notifyFailure and resolves `failed`,
 * so fire-and-forget callers can `void` the result safely.
 */
export async function openExternalUrl(
  rawUrl: string,
  deps: ExternalOpenDeps
): Promise<ExternalOpenResult> {
  const raw = String(rawUrl || '').trim()

  if (!raw) {
    return { ok: false, reason: 'invalid' }
  }

  let parsed: URL

  try {
    parsed = new URL(raw)
  } catch {
    return { ok: false, reason: 'invalid' }
  }

  if (parsed.protocol === 'file:') {
    try {
      await deps.openFile(raw)
    } catch {
      // main's openFile handles its own fallback; never surfaced here
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
    await deps.openExternal(url)

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

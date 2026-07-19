import fs from 'node:fs'
import path from 'node:path'

export type WatchdogPrewarmedBackend = {
  baseUrl: string
  token: string
  port: number
  pid?: number
  hermesRoot?: string
  source: 'watchdog'
}

const DEFAULT_PROBE_TIMEOUT_MS = 3_000

function watchdogManifestPath() {
  const local = process.env.LOCALAPPDATA

  if (!local) {
    return null
  }

  return path.join(local, 'HermesWatchdog', 'desktop-backend.json')
}

function readManifest(): Record<string, unknown> | null {
  const manifestPath = watchdogManifestPath()

  if (!manifestPath) {
    return null
  }

  try {
    return JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
  } catch {
    return null
  }
}

async function probePublicStatus(baseUrl: string, timeoutMs: number) {
  const url = `${String(baseUrl || '').replace(/\/+$/, '')}/api/status`
  const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) })

  return res.ok
}

/**
 * Connect to the Go watchdog's pre-warmed local serve when the packaged shortcut
 * launches Hermes.exe without HERMES_DESKTOP_* env (see desktop-backend.json).
 */
async function resolveWatchdogPrewarmedBackend(
  options: {
    hermesRoot?: string | null
    timeoutMs?: number
  } = {}
): Promise<WatchdogPrewarmedBackend | null> {
  if (process.platform !== 'win32') {
    return null
  }

  const raw = readManifest()

  if (!raw) {
    return null
  }

  const port = Number(raw.port)
  const token = typeof raw.token === 'string' ? raw.token.trim() : ''
  const baseUrl =
    typeof raw.baseUrl === 'string' && raw.baseUrl.trim()
      ? raw.baseUrl.trim()
      : Number.isInteger(port) && port > 0
        ? `http://127.0.0.1:${port}`
        : ''

  if (!baseUrl || !token) {
    return null
  }

  const manifestRoot = typeof raw.hermesRoot === 'string' ? raw.hermesRoot.trim() : ''
  const expectedRoot = options.hermesRoot && String(options.hermesRoot).trim()

  if (expectedRoot && manifestRoot && path.resolve(manifestRoot) !== path.resolve(expectedRoot)) {
    return null
  }

  const timeoutMs = options.timeoutMs ?? DEFAULT_PROBE_TIMEOUT_MS

  try {
    if (!(await probePublicStatus(baseUrl, timeoutMs))) {
      return null
    }
  } catch {
    return null
  }

  return {
    baseUrl,
    token,
    port: Number.isInteger(port) && port > 0 ? port : Number(new URL(baseUrl).port) || 0,
    pid: typeof raw.pid === 'number' ? raw.pid : undefined,
    hermesRoot: manifestRoot || undefined,
    source: 'watchdog'
  }
}

export { DEFAULT_PROBE_TIMEOUT_MS, resolveWatchdogPrewarmedBackend, watchdogManifestPath }

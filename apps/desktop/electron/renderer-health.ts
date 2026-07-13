import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

export type RendererHealthStatus = 'starting' | 'ready' | 'failed'

export interface RendererHealthReporter {
  readonly enabled: boolean
  ready: () => void
  fail: (reason: string) => void
}

interface RendererHealthReporterOptions {
  markerPath?: string
  nonce?: string
  pid?: number
  executable?: string
  tempDir?: string
  now?: () => number
}

const MARKER_BASENAME = /^hermes-desktop-health-[A-Za-z0-9-]+\.json$/
const NONCE = /^[A-Za-z0-9-]{16,128}$/

function samePath(left: string, right: string): boolean {
  const normalize = (value: string) => {
    const resolved = path.resolve(value)
    return process.platform === 'win32' ? resolved.toLowerCase() : resolved
  }
  return normalize(left) === normalize(right)
}

function markerPathIsSafe(markerPath: string, tempDir: string): boolean {
  if (!MARKER_BASENAME.test(path.basename(markerPath))) {
    return false
  }
  if (!samePath(path.dirname(markerPath), tempDir)) {
    return false
  }

  try {
    if (!samePath(fs.realpathSync(path.dirname(markerPath)), fs.realpathSync(tempDir))) {
      return false
    }
  } catch {
    return false
  }

  let info: fs.Stats
  try {
    info = fs.lstatSync(markerPath)
  } catch (error) {
    return (error as NodeJS.ErrnoException)?.code === 'ENOENT'
  }
  if (!info.isFile() || info.isSymbolicLink()) {
    return false
  }
  try {
    return samePath(fs.realpathSync(markerPath), markerPath)
  } catch {
    return false
  }
}

export function createRendererHealthReporter({
  markerPath = process.env.HERMES_DESKTOP_HEALTH_MARKER || '',
  nonce = process.env.HERMES_DESKTOP_HEALTH_NONCE || '',
  pid = process.pid,
  executable = process.execPath,
  tempDir = process.env.HERMES_DESKTOP_HEALTH_TEMP_DIR || os.tmpdir(),
  now = Date.now
}: RendererHealthReporterOptions = {}): RendererHealthReporter {
  let active = Boolean(markerPath && NONCE.test(nonce) && markerPathIsSafe(markerPath, tempDir))
  let status: RendererHealthStatus = 'starting'
  let failureCount = 0
  let failureReason: string | null = null

  const write = () => {
    if (!active || !markerPathIsSafe(markerPath, tempDir)) {
      active = false
      return
    }
    try {
      fs.writeFileSync(
        markerPath,
        JSON.stringify({
          schema: 1,
          nonce,
          pid,
          executable,
          status,
          failure_count: failureCount,
          failure_reason: failureReason,
          updated_at_ms: now()
        }),
        { encoding: 'utf8', mode: 0o600 }
      )
    } catch {
      active = false
    }
  }

  write()

  return {
    get enabled() {
      return active
    },
    ready() {
      status = 'ready'
      write()
    },
    fail(reason: string) {
      failureCount += 1
      failureReason = String(reason || 'renderer-failure').slice(0, 256)
      status = 'failed'
      write()
    }
  }
}

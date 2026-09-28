// Consent-gated renderer-crash counter for shared metrics. A crash can take the
// renderer (and its backend socket) down before anything is sent, so the main
// process persists a bucketed reason per render-process-gone and the renderer
// drains the list once a backend is attached (take/ack, like update-metrics.ts).
//
// Only the bucketed reason is stored — never exit codes, paths or window
// titles. Nothing is recorded or kept on disk until the renderer reports the
// user's opt-in, and revoking it deletes the pending file at once.

import fs from 'node:fs'
import path from 'node:path'

export type RendererCrashReason = 'crash' | 'oom' | 'killed' | 'other'

export const PENDING_RENDERER_CRASH_FILE = 'renderer-crash-pending.json'
export const MAX_PENDING_RENDERER_CRASHES = 20

const REASON_BY_ELECTRON: Readonly<Record<string, RendererCrashReason>> = {
  crashed: 'crash',
  oom: 'oom',
  killed: 'killed'
}

const KNOWN_REASONS: ReadonlySet<string> = new Set<RendererCrashReason>(['crash', 'oom', 'killed', 'other'])

export function rendererCrashReason(electronReason: unknown): RendererCrashReason {
  return typeof electronReason === 'string' && Object.hasOwn(REASON_BY_ELECTRON, electronReason)
    ? REASON_BY_ELECTRON[electronReason]
    : 'other'
}

export type RendererCrashFs = Pick<typeof fs, 'mkdirSync' | 'readFileSync' | 'renameSync' | 'rmSync' | 'writeFileSync'>

export interface RendererCrashRecorderOptions {
  dir: string
  fs?: RendererCrashFs
}

export class RendererCrashRecorder {
  private enabled = false
  private inFlight = false
  private taken = 0
  private readonly fs: RendererCrashFs
  private readonly file: string

  constructor(opts: RendererCrashRecorderOptions) {
    this.fs = opts.fs ?? fs
    this.file = path.join(opts.dir, PENDING_RENDERER_CRASH_FILE)
  }

  setEnabled(on: boolean): void {
    this.enabled = on

    if (!on) {
      this.inFlight = false
      this.taken = 0
      this.remove()
    }
  }

  record(electronReason: unknown): void {
    // A clean exit is a renderer shutting down normally, not a crash.
    if (!this.enabled || electronReason === 'clean-exit') {
      return
    }

    const reasons = this.read()

    if (reasons.length >= MAX_PENDING_RENDERER_CRASHES) {
      return
    }

    reasons.push(rendererCrashReason(electronReason))
    this.write(reasons)
  }

  /** Claim the pending reasons for one send; null when disabled, empty, or a claim is unacked. */
  take(): { reasons: RendererCrashReason[] } | null {
    if (!this.enabled || this.inFlight) {
      return null
    }

    const reasons = this.read()

    if (reasons.length === 0) {
      return null
    }

    this.inFlight = true
    this.taken = reasons.length

    return { reasons }
  }

  /** `sent` drops the claimed reasons (keeping any recorded since); false keeps them for the next attach. */
  ack(sent: boolean): void {
    if (!this.inFlight) {
      return
    }

    if (sent) {
      const rest = this.read().slice(this.taken)

      if (rest.length > 0) {
        this.write(rest)
      } else {
        this.remove()
      }
    }

    this.inFlight = false
    this.taken = 0
  }

  private read(): RendererCrashReason[] {
    try {
      const parsed = JSON.parse(String(this.fs.readFileSync(this.file, 'utf8'))) as { v?: unknown; reasons?: unknown }

      if (parsed?.v !== 1 || !Array.isArray(parsed.reasons)) {
        return []
      }

      return parsed.reasons
        .filter((reason): reason is RendererCrashReason => typeof reason === 'string' && KNOWN_REASONS.has(reason))
        .slice(0, MAX_PENDING_RENDERER_CRASHES)
    } catch {
      // Missing or unreadable: nothing pending.
      return []
    }
  }

  private write(reasons: RendererCrashReason[]): void {
    try {
      const tmp = `${this.file}.tmp`
      this.fs.mkdirSync(path.dirname(tmp), { recursive: true })
      this.fs.writeFileSync(tmp, JSON.stringify({ v: 1, reasons }))
      this.fs.renameSync(tmp, this.file)
    } catch {
      // Telemetry never breaks the app.
    }
  }

  private remove(): void {
    try {
      this.fs.rmSync(this.file, { force: true })
    } catch {
      // Retried on the next ack or opt-out.
    }
  }
}

interface IpcHandleTarget {
  handle(channel: string, listener: (event: unknown, ...args: unknown[]) => unknown): void
}

export function registerRendererCrashIpc(ipc: IpcHandleTarget, recorder: RendererCrashRecorder): void {
  ipc.handle('hermes:desktop-metrics:set-enabled', (_event: unknown, on: unknown): void =>
    recorder.setEnabled(on === true)
  )
  ipc.handle('hermes:desktop-metrics:crash:take', (): { reasons: RendererCrashReason[] } | null => recorder.take())
  ipc.handle('hermes:desktop-metrics:crash:ack', (_event: unknown, sent: unknown): void => recorder.ack(sent === true))
}

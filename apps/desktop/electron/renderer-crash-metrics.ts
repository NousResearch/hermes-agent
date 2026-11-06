// Consent-gated renderer-crash counter for shared metrics. A crash can take the
// renderer (and its backend socket) down before anything is sent, so the main
// process persists a bucketed reason per render-process-gone and the renderer
// drains the list once a backend is attached (take/ack, like update-metrics.ts).
//
// Consent is per window: each window reports its focused profile's switch, a
// crash is kept only when the crashed window's own profile collects, and each
// entry is tagged with that profile so only a window of the same profile drains
// it (and that profile turning off drops only its entries). Stored: the bucketed
// reason plus a hash of the profile key — never the name, exit codes, paths or
// window titles.

import { createHash } from 'node:crypto'
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

interface PendingCrash {
  profile: string
  reason: RendererCrashReason
}

function profileKey(profile: unknown): string {
  return createHash('sha256')
    .update(String(profile ?? ''))
    .digest('hex')
    .slice(0, 16)
}

export class RendererCrashRecorder {
  /** webContents id → profile key, only for windows whose focused profile collects. */
  private readonly consent = new Map<number, string>()
  /** profile key → the window holding an unacked take() and how many entries it claimed. */
  private readonly claims = new Map<string, { count: number; windowId: number }>()
  private readonly fs: RendererCrashFs
  private readonly file: string

  constructor(opts: RendererCrashRecorderOptions) {
    this.fs = opts.fs ?? fs
    this.file = path.join(opts.dir, PENDING_RENDERER_CRASH_FILE)
  }

  /** One window's focused profile and its switch; off drops that profile's pending crashes. */
  setEnabled(windowId: number, profile: unknown, on: boolean): void {
    const key = profileKey(profile)

    if (on) {
      this.consent.set(windowId, key)

      return
    }

    this.consent.delete(windowId)
    this.claims.delete(key)
    this.write(this.read().filter(entry => entry.profile !== key))
  }

  record(windowId: number, electronReason: unknown): void {
    const profile = this.consent.get(windowId)

    // A clean exit is a renderer shutting down normally, not a crash.
    if (!profile || electronReason === 'clean-exit') {
      return
    }

    const entries = this.read()

    if (entries.length >= MAX_PENDING_RENDERER_CRASHES) {
      return
    }

    entries.push({ profile, reason: rendererCrashReason(electronReason) })
    this.write(entries)
  }

  /** Claim this window's profile's pending reasons; null when it does not collect, none, or another
   *  window's claim is unacked (the same window re-claims: its renderer reloaded before acking). */
  take(windowId: number): { reasons: RendererCrashReason[] } | null {
    const profile = this.consent.get(windowId)
    const claim = profile ? this.claims.get(profile) : undefined

    if (!profile || (claim && claim.windowId !== windowId)) {
      return null
    }

    const reasons = this.read()
      .filter(entry => entry.profile === profile)
      .map(entry => entry.reason)

    if (reasons.length === 0) {
      return null
    }

    this.claims.set(profile, { count: reasons.length, windowId })

    return { reasons }
  }

  /** `sent` drops the claimed reasons (keeping any recorded since); false keeps them for the next attach. */
  ack(windowId: number, sent: boolean): void {
    const [profile, claim] = [...this.claims].find(([, held]) => held.windowId === windowId) ?? []

    if (!profile || !claim) {
      return
    }

    this.claims.delete(profile)

    if (sent) {
      let drop = claim.count
      this.write(this.read().filter(entry => entry.profile !== profile || drop-- <= 0))
    }
  }

  private read(): PendingCrash[] {
    try {
      const parsed = JSON.parse(String(this.fs.readFileSync(this.file, 'utf8'))) as { v?: unknown; entries?: unknown }

      if (parsed?.v !== 2 || !Array.isArray(parsed.entries)) {
        return []
      }

      return parsed.entries
        .filter(
          (entry): entry is PendingCrash =>
            typeof entry?.profile === 'string' && typeof entry.reason === 'string' && KNOWN_REASONS.has(entry.reason)
        )
        .map(({ profile, reason }) => ({ profile, reason }))
        .slice(0, MAX_PENDING_RENDERER_CRASHES)
    } catch {
      // Missing or unreadable: nothing pending.
      return []
    }
  }

  private write(entries: PendingCrash[]): void {
    try {
      if (entries.length === 0) {
        this.fs.rmSync(this.file, { force: true })

        return
      }

      const tmp = `${this.file}.tmp`
      this.fs.mkdirSync(path.dirname(tmp), { recursive: true })
      this.fs.writeFileSync(tmp, JSON.stringify({ v: 2, entries }))
      this.fs.renameSync(tmp, this.file)
    } catch {
      // Telemetry never breaks the app; a failed removal is retried on the next ack or opt-out.
    }
  }
}

interface IpcHandleTarget {
  handle(channel: string, listener: (event: unknown, ...args: unknown[]) => unknown): void
}

/** The calling window's webContents id (-1 for a caller without one: never consents). */
function senderId(event: unknown): number {
  const id = (event as { sender?: { id?: unknown } } | null)?.sender?.id

  return typeof id === 'number' ? id : -1
}

export function registerRendererCrashIpc(ipc: IpcHandleTarget, recorder: RendererCrashRecorder): void {
  ipc.handle('hermes:desktop-metrics:set-enabled', (event: unknown, on: unknown, profile: unknown): void =>
    recorder.setEnabled(senderId(event), profile, on === true)
  )
  ipc.handle('hermes:desktop-metrics:crash:take', (event: unknown): { reasons: RendererCrashReason[] } | null =>
    recorder.take(senderId(event))
  )
  ipc.handle('hermes:desktop-metrics:crash:ack', (event: unknown, sent: unknown): void =>
    recorder.ack(senderId(event), sent === true)
  )
}

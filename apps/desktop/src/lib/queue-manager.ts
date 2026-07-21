import type { GatewayRequest } from '@/app/session/hooks/use-prompt-actions/utils'
import { notify } from '@/store/notifications'
import { $workingSessionIds } from '@/store/session-states'
import type { ComposerAttachment } from '@/store/composer'

export interface QueuedPromptEntry {
  id: string
  text: string
  attachments: ComposerAttachment[]
  queuedAt: number
  storedSessionId: string
}

type QueueState = Record<string, QueuedPromptEntry[]>

const EMPTY_QUEUE: QueuedPromptEntry[] = []
const STORAGE_KEY = 'hermes.desktop.composerQueue.v1'
const MAX_DRAIN_RETRIES = 3
const DRAIN_RETRY_MS = 2_000
const POLL_MS = 30_000

/**
 * Singleton queue manager that survives component mount/unmount cycles.
 *
 * Drains: triggered by $workingSessionIds (busy → idle), with a 30s polling
 * fallback for connection-gap scenarios.  Each drain call resolves the source
 * session via session.resume to avoid stale runtime ids (#61573).
 */
class QueueManager {
  private requestGateway: GatewayRequest | null = null
  private queues = new Map<string, QueuedPromptEntry[]>()
  private unsubWorking: (() => void) | null = null
  private pollTimer: ReturnType<typeof setInterval> | null = null
  private listeners = new Set<() => void>()
  private notifyRaf: number | null = null

  private constructor() {
    this.load()
  }

  private static _instance: QueueManager | null = null

  static getInstance(): QueueManager {
    if (!QueueManager._instance) {
      QueueManager._instance = new QueueManager()
    }
    return QueueManager._instance
  }

  /** React-friendly subscription (useSyncExternalStore). */
  subscribe(cb: () => void): () => void {
    this.listeners.add(cb)
    return () => this.listeners.delete(cb)
  }

  private notify(): void {
    // RAF-batched: skip redundant notifications in a single frame.
    if (this.notifyRaf !== null) return
    this.notifyRaf = requestAnimationFrame(() => {
      this.notifyRaf = null
      for (const cb of this.listeners) cb()
    })
  }

  /**
   * Initialize the drain subsystem.  Must be called once from wiring.tsx once
   * the gateway connection is open.
   */
  init(requestGateway: GatewayRequest): void {
    this.requestGateway = requestGateway

    // Event-driven drain: sessions exiting $workingSessionIds trigger drain.
    this.unsubWorking = $workingSessionIds.subscribe(ids => {
      for (const [sid, entries] of this.queues) {
        if (entries.length === 0) continue
        if (!ids.includes(sid)) {
          void this.tryDrain(sid, 0)
        }
      }
    })

    // Polling fallback for missed events (reconnect windows, etc.).
    this.pollTimer = setInterval(() => this.pollSessions(), POLL_MS)
  }

  /** Tear down subscriptions and timers.  Idempotent. */
  destroy(): void {
    this.unsubWorking?.()
    this.unsubWorking = null
    if (this.pollTimer !== null) {
      clearInterval(this.pollTimer)
      this.pollTimer = null
    }
  }

  // ── Public CRUD ──

  enqueue(storedSessionId: string, payload: { text: string; attachments: ComposerAttachment[] }): QueuedPromptEntry | null {
    const sid = storedSessionId.trim()
    if (!sid) return null

    const entry: QueuedPromptEntry = {
      id: `queued-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      text: payload.text,
      attachments: payload.attachments.map(a => ({ ...a })),
      queuedAt: Date.now(),
      storedSessionId: sid
    }

    const queue = this.queues.get(sid) ?? []
    this.queues.set(sid, [...queue, entry])
    this.save()
    return entry
  }

  dequeue(storedSessionId: string): QueuedPromptEntry | null {
    const sid = storedSessionId.trim()
    if (!sid) return null

    const queue = this.queues.get(sid)
    if (!queue || queue.length === 0) return null

    const [head, ...rest] = queue
    if (rest.length === 0) {
      this.queues.delete(sid)
    } else {
      this.queues.set(sid, rest)
    }
    this.save()
    return head
  }

  remove(storedSessionId: string, id: string): boolean {
    const sid = storedSessionId.trim()
    if (!sid) return false

    const queue = this.queues.get(sid)
    if (!queue) return false

    const next = queue.filter(e => e.id !== id)
    if (next.length === queue.length) return false

    if (next.length === 0) {
      this.queues.delete(sid)
    } else {
      this.queues.set(sid, next)
    }
    this.save()
    return true
  }

  clear(storedSessionId: string): void {
    const sid = storedSessionId.trim()
    if (!sid) return
    if (this.queues.delete(sid)) this.save()
  }

  clearAll(): void {
    this.queues.clear()
    this.save()
  }

  getAll(storedSessionId: string): QueuedPromptEntry[] {
    const sid = storedSessionId.trim()
    return sid ? (this.queues.get(sid) ?? EMPTY_QUEUE) : EMPTY_QUEUE
  }

  migrate(fromSid: string, toSid: string): boolean {
    const from = fromSid.trim()
    const to = toSid.trim()
    if (!from || !to || from === to) return false

    const pending = this.queues.get(from)
    if (!pending || pending.length === 0) return false

    this.queues.delete(from)
    const target = this.queues.get(to) ?? []
    this.queues.set(to, [...target, ...pending])
    this.save()
    return true
  }

  promote(storedSessionId: string, entryId: string): boolean {
    const sid = storedSessionId.trim()
    if (!sid) return false

    const queue = this.queues.get(sid)
    if (!queue) return false

    const index = queue.findIndex(e => e.id === entryId)
    if (index <= 0) return false

    // New array reference so React useSyncExternalStore detects the change.
    const next = [queue[index], ...queue.slice(0, index), ...queue.slice(index + 1)]
    this.queues.set(sid, next)
    this.save()
    return true
  }

  update(storedSessionId: string, entryId: string, text: string, attachments?: ComposerAttachment[]): boolean {
    const sid = storedSessionId.trim()
    if (!sid) return false

    const queue = this.queues.get(sid)
    if (!queue) return false

    let changed = false
    const next = queue.map(e => {
      if (e.id !== entryId) return e
      const newAtts = attachments ? attachments.map(a => ({ ...a })) : e.attachments
      if (e.text === text && !attachments) return e
      changed = true
      return { ...e, text, attachments: newAtts }
    })

    if (!changed) return false
    this.queues.set(sid, next)
    this.save()
    return true
  }

  // ── Drain ──

  private async tryDrain(storedSessionId: string, retryCount: number): Promise<void> {
    const gw = this.requestGateway
    if (!gw) return

    const queue = this.queues.get(storedSessionId)
    if (!queue || queue.length === 0) return

    const entry = queue[0]

    try {
      const resumed = await gw<{ session_id: string; running?: boolean }>('session.resume', {
        session_id: storedSessionId,
        source: 'desktop'
      })

      if (!resumed?.session_id) {
        if (retryCount < MAX_DRAIN_RETRIES) {
          setTimeout(() => this.tryDrain(storedSessionId, retryCount + 1), DRAIN_RETRY_MS)
        }
        return
      }

      if (resumed.running) return

      await gw('prompt.submit', { session_id: resumed.session_id, text: entry.text }, 1_800_000)

      this.dequeue(storedSessionId)
    } catch {
      if (retryCount < MAX_DRAIN_RETRIES) {
        setTimeout(() => this.tryDrain(storedSessionId, retryCount + 1), DRAIN_RETRY_MS)
      } else {
        notify({
          id: `composer-queue-stuck-${storedSessionId}`,
          kind: 'error',
          title: 'Queue item stuck',
          message: 'A queued message could not be sent. Please try again manually.'
        })
      }
    }
  }

  private pollSessions(): void {
    const working = $workingSessionIds.get()

    for (const [sid, entries] of this.queues) {
      if (entries.length === 0) continue
      if (!working.includes(sid)) {
        void this.tryDrain(sid, 0)
      }
    }
  }

  // ── Persistence ──

  private save(): void {
    if (typeof window === 'undefined') return

    const state: QueueState = {}
    for (const [sid, entries] of this.queues) state[sid] = entries

    try {
      if (Object.keys(state).length === 0) {
        window.localStorage.removeItem(STORAGE_KEY)
      } else {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
      }
    } catch {
      // best-effort: queue stays in memory
    }

    this.notify()
  }

  private load(): void {
    if (typeof window === 'undefined') return

    try {
      const raw = window.localStorage.getItem(STORAGE_KEY)
      if (!raw) return

      const parsed: QueueState = JSON.parse(raw)
      if (typeof parsed !== 'object' || Array.isArray(parsed)) return

      for (const [sid, entries] of Object.entries(parsed)) {
        if (!Array.isArray(entries) || entries.length === 0) continue
        // Ensure legacy entries without storedSessionId are backfilled.
        this.queues.set(sid, entries.map(e => ({ ...e, storedSessionId: e.storedSessionId ?? sid })))
      }
    } catch {
      // best-effort
    }
  }
}

export const queueManager = QueueManager.getInstance()
export { queueManager as QueueManager }

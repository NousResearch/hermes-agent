import { atom } from 'nanostores'

/**
 * Composer-scheduled messages (#111873).
 *
 * A scheduled message is a draft the user deferred: it is NOT sent, it is not a
 * cron job and it carries no provider/model/delivery configuration. The backend
 * owns the truth (it persists the item and fires it as a normal user turn in
 * this session); this store is the renderer's cache of that list, keyed by
 * session id — the same scope the backend binds the item to.
 *
 * `dueAt` is epoch MILLISECONDS for the UI (the wire carries epoch seconds, the
 * backend's own unit). Times are formatted in the user's local zone, which is
 * the only zone a `datetime-local` input can mean.
 */

export interface ScheduledMessage {
  /** Backend id — the cancel handle. */
  id: string
  text: string
  /** What the panel shows; long drafts are truncated server-side. */
  displayText: string
  dueAt: number
  createdAt: number
  /** Past its due time and still waiting (the session was busy or not open). */
  overdue: boolean
}

type ScheduledState = Record<string, ScheduledMessage[]>

export const $scheduledMessagesBySession = atom<ScheduledState>({})

interface WireMessage {
  id?: unknown
  text?: unknown
  display_text?: unknown
  due_at?: unknown
  created_at?: unknown
  overdue?: unknown
}

/** Wire → UI list. Epoch SECONDS on the wire (backend unit) → ms here. */
export function fromWire(raw: unknown): ScheduledMessage[] {
  if (!Array.isArray(raw)) {
    return []
  }

  return raw.flatMap(entry => {
    const item = (entry ?? {}) as WireMessage
    const id = typeof item.id === 'string' ? item.id : ''
    const text = typeof item.text === 'string' ? item.text : ''
    const dueSeconds = typeof item.due_at === 'number' ? item.due_at : Number(item.due_at)

    if (!id || !text || !Number.isFinite(dueSeconds)) {
      return []
    }

    const createdSeconds = typeof item.created_at === 'number' ? item.created_at : Number(item.created_at)

    return [
      {
        id,
        text,
        displayText: typeof item.display_text === 'string' && item.display_text ? item.display_text : text,
        dueAt: dueSeconds * 1000,
        createdAt: Number.isFinite(createdSeconds) ? createdSeconds * 1000 : 0,
        overdue: item.overdue === true
      }
    ]
  })
}

const pad = (value: number) => String(value).padStart(2, '0')

/**
 * `<input type="datetime-local">` value for a Date, in LOCAL time.
 * `toISOString()` would flip the wall clock to UTC and schedule a message hours
 * off for anyone east or west of Greenwich.
 */
export function toLocalInputValue(date: Date): string {
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    `T${pad(date.getHours())}:${pad(date.getMinutes())}`
  )
}

/** Default value for the picker: now + `minutes`, rounded up to the next whole minute. */
export function defaultDueInput(now: Date = new Date(), minutes = 5): string {
  const target = new Date(now.getTime() + minutes * 60_000)

  target.setSeconds(0, 0)

  return toLocalInputValue(target)
}

/**
 * A `datetime-local` value → epoch ms, or null when unparseable.
 *
 * `new Date('2026-09-16T21:30')` is parsed as LOCAL time by spec, which is
 * exactly what the input means; anything else (empty, garbage) is null rather
 * than an Invalid Date the RPC would reject with a confusing message.
 */
export function parseDueInput(value: string): null | number {
  const text = value.trim()

  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(text)) {
    return null
  }

  const parsed = new Date(text)

  return Number.isNaN(parsed.getTime()) ? null : parsed.getTime()
}

/** True when `value` is a usable future time (allow a minute of slack for a slow clock). */
export function isFutureDueInput(value: string, now: number = Date.now()): boolean {
  const due = parseDueInput(value)

  return due !== null && due > now - 60_000
}

/** Local wall-clock rendering for the panel: `Sep 16, 21:30`. */
export function formatLocalDue(dueAtMs: number): string {
  const date = new Date(dueAtMs)

  return Number.isNaN(date.getTime())
    ? ''
    : date.toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
}

export function setScheduledMessages(sessionId: string, messages: ScheduledMessage[]): void {
  const current = $scheduledMessagesBySession.get()

  if (sessionId && messages.length > 0) {
    $scheduledMessagesBySession.set({ ...current, [sessionId]: messages })
  } else {
    const next = { ...current }

    if (sessionId) {
      delete next[sessionId]
    }

    $scheduledMessagesBySession.set(next)
  }
}

export function clearScheduledMessages(sessionId: string): void {
  setScheduledMessages(sessionId, [])
}

export function getScheduledMessages(state: ScheduledState, sessionId: null | string): ScheduledMessage[] {
  return (sessionId && state[sessionId]) || []
}

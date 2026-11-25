import { atom } from 'nanostores'

export type DaybreakRequester = (method: string, params?: Record<string, unknown>) => Promise<unknown>

/** `agent.daybreak` per connection + profile, as that owner last confirmed it. Profile-wide like
 *  Codex's own switch: the backend sends it only on models the account's Codex catalog lists. */
export const $daybreakByProfile = atom<Record<string, boolean>>({})

const confirmed = new Map<string, boolean>()
const reads = new Map<string, number>()
const writes = new Map<string, number>()
const pendingWrites = new Map<string, number>()
const queues = new Map<string, Promise<unknown>>()

/** Same profile name on two connections (local + remote `default`) is two settings. */
export function daybreakKey(profile: string, connectionId?: string): string {
  return `${connectionId ?? ''}::${profile.trim() || 'default'}`
}

// `config.get` / `config.set` are profile-scoped; a blank name is the launch profile (approval-mode.ts).
function scopeToProfile(profile: string, params: Record<string, unknown>): Record<string, unknown> {
  const name = profile.trim()

  return name ? { ...params, profile: name } : params
}

function bump(counter: Map<string, number>, key: string): number {
  const next = (counter.get(key) ?? 0) + 1
  counter.set(key, next)

  return next
}

function show(key: string, on: boolean): void {
  $daybreakByProfile.set({ ...$daybreakByProfile.get(), [key]: on })
}

function confirm(key: string, on: boolean): void {
  confirmed.set(key, on)
  show(key, on)
}

/** Read the owner's value. A read never overrides a write: it applies only when no write started
 *  (or is still in flight) since the read began. */
export async function syncDaybreak(request: DaybreakRequester, profile: string, connectionId?: string): Promise<void> {
  const key = daybreakKey(profile, connectionId)
  const read = bump(reads, key)
  const writesAtStart = writes.get(key) ?? 0
  const result = (await request('config.get', scopeToProfile(profile, { key: 'daybreak' }))) as { value?: unknown }

  if (reads.get(key) === read && (writes.get(key) ?? 0) === writesAtStart && !pendingWrites.get(key)) {
    confirm(key, result?.value === true)
  }
}

/** Optimistic write. Writes to one key go out one at a time, so the owner applies them in click
 *  order and every reply is its saved value. Each success updates that saved value even while a
 *  newer flip is pending; a refusal shows the saved value (if it is the latest flip) and resolves false. */
export async function setDaybreak(
  request: DaybreakRequester,
  profile: string,
  on: boolean,
  connectionId?: string
): Promise<boolean> {
  const key = daybreakKey(profile, connectionId)
  const write = bump(writes, key)
  pendingWrites.set(key, (pendingWrites.get(key) ?? 0) + 1)
  show(key, on)

  const send = () => request('config.set', scopeToProfile(profile, { key: 'daybreak', value: on }))
  const sent = (queues.get(key) ?? Promise.resolve()).then(send)
  queues.set(
    key,
    sent.catch(() => undefined)
  )

  try {
    const saved = ((await sent) as { value?: unknown } | undefined)?.value === true
    confirmed.set(key, saved)

    if (writes.get(key) === write) {
      show(key, saved)
    }

    return true
  } catch {
    if (writes.get(key) === write) {
      show(key, confirmed.get(key) ?? false)
    }

    return false
  } finally {
    pendingWrites.set(key, (pendingWrites.get(key) ?? 1) - 1)
  }
}

import { atom, computed } from 'nanostores'

/**
 * Plugin lines for the pet's speech bubble — the store behind `ctx.pet.say`.
 *
 * A plugin hands the host a short plain-text line; the host owns where and
 * when it shows (the core `PetBubble`, in-window and in the pop-out overlay),
 * so no plugin ever has to locate the pet in the app DOM or float its own
 * overlay over it. Every line is:
 *
 *  - plain text: control characters stripped, whitespace collapsed, capped at
 *    `PET_MESSAGE_MAX_CHARS` (rendered as a React text node, never HTML);
 *  - attributed: it carries the plugin id and its display name, which the
 *    bubble prints as a small label;
 *  - short-lived: a TTL (default `PET_MESSAGE_DEFAULT_TTL_MS`, clamped) retires
 *    it, and the disposer `say` returns retires it early;
 *  - bounded: each plugin holds at most `PET_MESSAGE_MAX_PER_PLUGIN` live lines
 *    and gets `PET_MESSAGE_RATE_LIMIT` says per `PET_MESSAGE_RATE_WINDOW_MS`.
 *
 * The bubble shows the newest live line. The owning plugin's context clears
 * all of its lines when it unloads or is disabled.
 */

export type PetMessageTone = 'error' | 'info' | 'wait'

export interface PetSayOptions {
  /** Stable key within your plugin. Saying again with the same id replaces
   *  that line in place (a ticking counter, a refreshed balance). */
  id?: string
  /** Visual tone: `info` (default, no glyph), `wait` (clock), `error` (alert). */
  tone?: PetMessageTone
  /** How long the line stays up, in ms. Clamped to 1 s – 30 s; default 6 s. */
  ttlMs?: number
}

/** One live plugin line. Plain data so it can cross to the overlay window. */
export interface PetPluginMessage {
  /** `<pluginId>:<id>` — unique across plugins. */
  key: string
  pluginId: string
  /** Display label (the plugin's name, else its id). */
  pluginName: string
  text: string
  tone: PetMessageTone
  /** Monotonic; the bubble shows the highest. */
  seq: number
}

export const PET_MESSAGE_MAX_CHARS = 120
export const PET_MESSAGE_LABEL_MAX_CHARS = 32
export const PET_MESSAGE_DEFAULT_TTL_MS = 6_000
export const PET_MESSAGE_MIN_TTL_MS = 1_000
export const PET_MESSAGE_MAX_TTL_MS = 30_000
export const PET_MESSAGE_MAX_PER_PLUGIN = 3
export const PET_MESSAGE_RATE_LIMIT = 10
export const PET_MESSAGE_RATE_WINDOW_MS = 10_000

const TONES: ReadonlySet<PetMessageTone> = new Set(['error', 'info', 'wait'])

export const $petPluginMessages = atom<PetPluginMessage[]>([])

/** The line the bubble shows: the newest live one, or null. */
export const $petPluginMessage = computed($petPluginMessages, list =>
  list.reduce<null | PetPluginMessage>((top, m) => (!top || m.seq > top.seq ? m : top), null)
)

const timers = new Map<string, ReturnType<typeof setTimeout>>()
const sayLog = new Map<string, number[]>()
let seq = 0

// C0/C1 control characters and the bidi overrides/isolates, which could make
// a line render as something other than what the plugin passed.
// eslint-disable-next-line no-control-regex
const UNSAFE_CHARS = /[\u0000-\u001f\u007f-\u009f\u200e\u200f\u202a-\u202e\u2066-\u2069]/g

/** Normalise arbitrary input to one capped line of plain text ('' = nothing to say). */
export function sanitizePetText(value: unknown, max = PET_MESSAGE_MAX_CHARS): string {
  const text = String(value ?? '')
    .replace(UNSAFE_CHARS, ' ')
    .replace(/\s+/g, ' ')
    .trim()

  return text.length > max ? `${text.slice(0, max - 1).trimEnd()}…` : text
}

function clampTtl(ttlMs: unknown): number {
  const n = typeof ttlMs === 'number' && Number.isFinite(ttlMs) ? ttlMs : PET_MESSAGE_DEFAULT_TTL_MS

  return Math.min(PET_MESSAGE_MAX_TTL_MS, Math.max(PET_MESSAGE_MIN_TTL_MS, Math.round(n)))
}

function allowSay(pluginId: string, now: number): boolean {
  const recent = (sayLog.get(pluginId) ?? []).filter(at => now - at < PET_MESSAGE_RATE_WINDOW_MS)

  if (recent.length >= PET_MESSAGE_RATE_LIMIT) {
    sayLog.set(pluginId, recent)

    return false
  }

  recent.push(now)
  sayLog.set(pluginId, recent)

  return true
}

function remove(key: string, expectedSeq?: number): void {
  const list = $petPluginMessages.get()
  const hit = list.find(m => m.key === key)

  // A disposer from a replaced line must not retire its replacement.
  if (!hit || (expectedSeq !== undefined && hit.seq !== expectedSeq)) {
    return
  }

  clearTimeout(timers.get(key))
  timers.delete(key)
  $petPluginMessages.set(list.filter(m => m.key !== key))
}

const noop = () => {}

/**
 * Show `text` in the pet bubble on behalf of `pluginId`. Returns a disposer
 * that removes the line early (idempotent; a no-op once replaced or expired).
 * Empty text, or a say over the plugin's rate limit, shows nothing.
 */
export function sayPetMessage(
  pluginId: string,
  pluginName: string,
  text: unknown,
  options: PetSayOptions = {}
): () => void {
  const clean = sanitizePetText(text)

  if (!clean) {
    return noop
  }

  if (!allowSay(pluginId, Date.now())) {
    console.warn(`[plugins] ${pluginId}: pet.say rate limit hit — line dropped`)

    return noop
  }

  const localId = sanitizePetText(options.id ?? '', 64) || `m${seq + 1}`
  const key = `${pluginId}:${localId}`
  const tone: PetMessageTone = options.tone && TONES.has(options.tone) ? options.tone : 'info'

  const message: PetPluginMessage = {
    key,
    pluginId,
    pluginName: sanitizePetText(pluginName, PET_MESSAGE_LABEL_MAX_CHARS) || pluginId,
    seq: ++seq,
    text: clean,
    tone
  }

  clearTimeout(timers.get(key))

  let next = [...$petPluginMessages.get().filter(m => m.key !== key), message]
  const own = next.filter(m => m.pluginId === pluginId)

  // Keep each plugin to its newest few lines; evict the oldest.
  for (const stale of own.slice(0, Math.max(0, own.length - PET_MESSAGE_MAX_PER_PLUGIN))) {
    clearTimeout(timers.get(stale.key))
    timers.delete(stale.key)
    next = next.filter(m => m.key !== stale.key)
  }

  $petPluginMessages.set(next)
  timers.set(
    key,
    setTimeout(() => remove(key, message.seq), clampTtl(options.ttlMs))
  )

  return () => remove(key, message.seq)
}

/** Remove one of a plugin's lines by its `id`, or every line it owns. */
export function clearPetMessages(pluginId: string, id?: string): void {
  const prefix = `${pluginId}:`
  const key = id === undefined ? null : `${prefix}${sanitizePetText(id, 64)}`

  for (const m of $petPluginMessages.get()) {
    if (m.pluginId === pluginId && (key === null || m.key === key)) {
      remove(m.key)
    }
  }
}

/** Overlay window: adopt the main renderer's list verbatim (it owns expiry). */
export function mirrorPetPluginMessages(list: unknown): void {
  $petPluginMessages.set(Array.isArray(list) ? (list as PetPluginMessage[]) : [])
}

/** Test seam: drop every line, timer, and rate-limit window. */
export function resetPetPluginMessages(): void {
  timers.forEach(clearTimeout)
  timers.clear()
  sayLog.clear()
  $petPluginMessages.set([])
}

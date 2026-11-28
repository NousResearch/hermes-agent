/**
 * A message a visitor sent before signing up (build plan 1C): held in this browser, never sent, and
 * the first thing that runs once they are signed in (2A).
 *
 * - Held: the visitor's latest message replaces any earlier one; only one runs first.
 * - Survives sign up: sign up is a full navigation away and back (and maybe a checkout), so the
 *   message lives in this browser's storage, not in memory. It is not a machine preference: before
 *   sign up there is no machine to share it with, and it must run once, here.
 * - Runs exactly once: whoever runs it takes it first ({@link takeHeldPrompt}), so a reload, a second
 *   tab or a failed send can never run it twice.
 * - Expires: a message held longer than {@link HELD_PROMPT_TTL_MS} is dropped instead of running
 *   unannounced on some later visit.
 *
 * It is not a conversation: it never enters the session list, the transcript or the gateway, and the
 * chat shows it through its own view (app/discovery/held-exchange.tsx).
 */

import { computed } from 'nanostores'

import { type Codec, persistentAtom } from '@/lib/persisted'

export const HELD_PROMPT_KEY = 'hermes.desktop.heldPrompt.v1'

/** A day: long enough for sign up and checkout, short enough that it is still what they meant. */
export const HELD_PROMPT_TTL_MS = 24 * 60 * 60 * 1000

interface HeldPrompt {
  text: string
  heldAt: number
}

function sanitize(value: unknown): HeldPrompt | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const { heldAt, text } = value as Partial<HeldPrompt>

  return typeof text === 'string' && text.trim() && typeof heldAt === 'number' && Number.isFinite(heldAt)
    ? { text, heldAt }
    : null
}

const codec: Codec<HeldPrompt | null> = {
  decode: raw => sanitize(JSON.parse(raw)),
  encode: value => (value ? JSON.stringify(value) : null)
}

const $held = persistentAtom<HeldPrompt | null>(HELD_PROMPT_KEY, null, codec)

const live = (held: HeldPrompt | null, now: number): null | string =>
  held && now - held.heldAt < HELD_PROMPT_TTL_MS ? held.text : null

/** The held message's text, or null when none is held or it expired. */
export const $heldPrompt = computed($held, held => live(held, Date.now()))

export function holdPrompt(text: string, now = Date.now()): void {
  const trimmed = text.trim()

  if (trimmed) {
    $held.set({ text: trimmed, heldAt: now })
  }
}

/** The held message, removed: the caller runs it. Null when none is held or it expired. */
export function takeHeldPrompt(now = Date.now()): null | string {
  // Read storage, not this tab's atom: another tab may have already claimed and cleared the entry,
  // and a stale in-memory copy must not let this tab send the same prompt a second time.
  let raw: null | string = null

  try {
    raw = window.localStorage.getItem(HELD_PROMPT_KEY)
  } catch {
    raw = null
  }

  // Claim it synchronously before returning, so two tabs can never both read the same entry.
  try {
    window.localStorage.removeItem(HELD_PROMPT_KEY)
  } catch {
    // Storage is best-effort; the in-memory clear below still retires this tab's copy.
  }

  let text: null | string = null

  if (raw !== null) {
    try {
      text = live(codec.decode(raw), now)
    } catch {
      text = null
    }
  }

  $held.set(null)

  return text
}

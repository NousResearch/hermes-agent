/**
 * Chat bubbles — the transcript's optional left/right layout.
 *
 * The document layout (the default, and what the app has always done) puts
 * every message in one full-width column: your prompt is a full-width bubble,
 * Hermes's reply is full-width prose. Two full-width columns stacked on each
 * other read as a single document — you have to *read* to know who is speaking,
 * and a long transcript turns into an undifferentiated wall of text.
 *
 * Turning this on gives the transcript the two-sided shape every chat app has:
 * your prompt becomes a right-aligned bubble, the reply a left-aligned bubble,
 * each with a small "tail" corner pointing at its speaker. The turn is legible
 * before a word is read, and the eye gets a column of its own to scan.
 *
 * Everyone who has used a messaging app already knows this shape, so nothing
 * has to be learned. It is opt-in, because the document layout is a deliberate
 * choice for long, technical answers (wide tables and code read better full
 * bleed) — this is the setting, not a replacement.
 *
 * Presentation-only state, so the renderer owns it (desktop AGENTS.md, "decide
 * state by authority": the renderer owns what is purely about this window's
 * presentation).
 *
 * SCOPE — desktop-wide, not per window and not per session. Every renderer
 * window of this profile shares one localStorage partition, so a layout choice
 * follows the user across windows the way the theme and the Message Bubble
 * lever do; a per-window layout would surprise more than it scopes. What that
 * scope DOES require is reconciliation: another window's change arrives as a
 * `storage` event (bottom of this file), and this window repaints from it
 * instead of disagreeing until the next launch.
 *
 * It paints as ONE root attribute plus two root variables that `styles.css`
 * consumes. That keeps the layout in exactly one place: no component branches,
 * no per-message props, and switching it repaints the whole transcript without
 * a single React render — which matters because the thread virtualizes long
 * conversations and must not re-render to change its paint. Nothing painted
 * here reaches a message node: the state lands on <html>, and the stylesheet
 * decides what that means for the transcript.
 *
 * Nothing here touches content either: no clamp, no overflow, no line cap. The
 * bubble is a surface — a long message stays whole, and the user bubble keeps
 * every core affordance (click to edit, restore, the sticky pin). The stylesheet
 * scopes the bubble to the human-prompt class alone, so background-process
 * notifications and agent-to-agent deliveries — which arrive on
 * `data-role="user"` and share the user message root — keep their flat,
 * left-aligned shape.
 */

import { atom } from 'nanostores'

import { readJson, writeJson } from '@/lib/storage'

/**
 * Desktop-wide renderer preference; the scope is in the name. Deliberately the
 * same shape as `hermes.desktop.user-bubble-transparency.v1` — an appearance
 * choice belongs to the app, not to one window.
 */
const KEY = 'hermes.desktop.chat-bubbles.v1'

/** Set on <html> while the bubble layout is on; absent = the document layout,
 *  which is why the CSS block needs no "document" rules and users who never
 *  touch this get byte-identical paint. */
export const CHAT_BUBBLE_ROOT_ATTR = 'data-chat-bubbles'

/** Which token family fills the bubbles. Also on <html>, because the fills are
 *  chosen in CSS (theme tokens), never computed in JS. */
export const CHAT_BUBBLE_TONE_ATTR = 'data-chat-bubble-tone'

export type ChatBubbleCorners = 'pill' | 'round' | 'soft'
export type ChatBubbleLayout = 'bubbles' | 'document'
export type ChatBubbleTone = 'accent' | 'neutral' | 'theme'

export interface ChatBubbleState {
  corners: ChatBubbleCorners
  layout: ChatBubbleLayout
  tone: ChatBubbleTone
}

export const CHAT_BUBBLE_FILL_MIN = 0
export const CHAT_BUBBLE_FILL_MAX = 100
export const CHAT_BUBBLE_FILL_STEP = 5

/**
 * Corner sets, in rem: the bubble's own radius plus the small "tail" corner
 * that points at its speaker (bottom-right for your prompt, top-left for the
 * reply). Rounded, not literal pixels, so the shape scales with UI Scale.
 */
export const CHAT_BUBBLE_RADII: Record<ChatBubbleCorners, { radius: string; tail: string }> = {
  pill: { radius: '1.5rem', tail: '0.375rem' },
  round: { radius: '1.125rem', tail: '0.3125rem' },
  soft: { radius: '0.625rem', tail: '0.25rem' }
}

export const CHAT_BUBBLE_DEFAULTS: ChatBubbleState = {
  corners: 'round',
  layout: 'document',
  tone: 'theme'
}

const CORNERS = new Set<string>(Object.keys(CHAT_BUBBLE_RADII))
const TONES = new Set<string>(['accent', 'neutral', 'theme'])

/** Coerce anything (a stale key from an older build, a hand-edited value, a
 *  value another window wrote) into a usable state instead of trusting
 *  storage. */
export function normalizeChatBubbles(value: unknown): ChatBubbleState {
  const raw = (value ?? {}) as Partial<Record<keyof ChatBubbleState, unknown>>

  return {
    corners:
      typeof raw.corners === 'string' && CORNERS.has(raw.corners)
        ? (raw.corners as ChatBubbleCorners)
        : CHAT_BUBBLE_DEFAULTS.corners,
    layout: raw.layout === 'bubbles' ? 'bubbles' : 'document',
    tone: typeof raw.tone === 'string' && TONES.has(raw.tone) ? (raw.tone as ChatBubbleTone) : CHAT_BUBBLE_DEFAULTS.tone
  }
}

/** Field-wise equality — used to swallow no-op external events so two windows
 *  cannot ping-pong writes at each other. */
export function sameChatBubbles(a: ChatBubbleState, b: ChatBubbleState): boolean {
  return a.corners === b.corners && a.layout === b.layout && a.tone === b.tone
}

/** The root variables a state paints. Split out of the DOM write so the
 *  contract is unit-tested without a document. */
export function chatBubbleVars(state: ChatBubbleState): Record<string, string> {
  const { radius, tail } = CHAT_BUBBLE_RADII[state.corners]

  return {
    '--chat-bubble-radius': radius,
    '--chat-bubble-tail': tail
  }
}

function paint(state: ChatBubbleState): void {
  const root = document.documentElement

  for (const [name, value] of Object.entries(chatBubbleVars(state))) {
    root.style.setProperty(name, value)
  }

  if (state.layout === 'bubbles') {
    root.setAttribute(CHAT_BUBBLE_ROOT_ATTR, 'bubbles')
    root.setAttribute(CHAT_BUBBLE_TONE_ATTR, state.tone)
  } else {
    root.removeAttribute(CHAT_BUBBLE_ROOT_ATTR)
    root.removeAttribute(CHAT_BUBBLE_TONE_ATTR)
  }
}

export const $chatBubbles = atom<ChatBubbleState>(
  typeof window === 'undefined' ? CHAT_BUBBLE_DEFAULTS : normalizeChatBubbles(readJson(KEY))
)

/** Patch one concern at a time; the others keep their current value. */
export function setChatBubbles(patch: Partial<ChatBubbleState>): void {
  $chatBubbles.set(normalizeChatBubbles({ ...$chatBubbles.get(), ...patch }))
}

export function resetChatBubbles(): void {
  $chatBubbles.set(CHAT_BUBBLE_DEFAULTS)
}

if (typeof window !== 'undefined') {
  // Skip a write this store already made, so an external change applied below
  // is not written straight back to the window it came from (a `storage` event
  // fires for every setItem, changed value or not, so an unguarded write here
  // would bounce between two windows forever).
  let written = JSON.stringify($chatBubbles.get())

  $chatBubbles.subscribe(state => {
    paint(state)

    const serialized = JSON.stringify(state)

    if (serialized !== written) {
      written = serialized
      writeJson(KEY, state)
    }
  })

  // Another window of this profile is the only other writer, and `storage` is
  // the event the browser fires for exactly that — never for our own writes.
  // Repaint from it rather than leaving two windows disagreeing until launch.
  window.addEventListener('storage', event => {
    if (event.key !== null && event.key !== KEY) {
      return
    }

    const next = normalizeChatBubbles(readJson(KEY))

    if (sameChatBubbles(next, $chatBubbles.get())) {
      return
    }

    $chatBubbles.set(next)
  })
}

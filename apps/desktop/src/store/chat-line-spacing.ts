import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.chat-line-spacing.v1'

// 100% is the pre-setting rendering exactly: the conversation reads the app's
// (theme-painted) leading unchanged, so this control is opt-in by construction.
const DEFAULT_CHAT_LINE_SPACING = 100

// Even 25% steps across the 75%–175% span: the 50%, 225% and 250% ends were
// tried and read as unusable extremes. Five presets fit one line of the
// Appearance row's action column (`minmax(15rem, 22rem)` → 240–352px) — the Chat
// Text Size row beside it ships six `${v}%` labels unwrapped — so the track keeps
// its default single row and needs no explicit columns.
export const CHAT_LINE_SPACING_PRESETS = [75, 100, 125, 150, 175] as const
export type ChatLineSpacing = (typeof CHAT_LINE_SPACING_PRESETS)[number]

export const CHAT_LINE_SPACING_MIN = CHAT_LINE_SPACING_PRESETS[0]
export const CHAT_LINE_SPACING_MAX = CHAT_LINE_SPACING_PRESETS[CHAT_LINE_SPACING_PRESETS.length - 1]

function normalizeChatLineSpacing(value: unknown): ChatLineSpacing {
  return CHAT_LINE_SPACING_PRESETS.find(preset => preset === Number(value)) ?? DEFAULT_CHAT_LINE_SPACING
}

export const $chatLineSpacing = atom<ChatLineSpacing>(normalizeChatLineSpacing(storedString(KEY)))

export function setChatLineSpacing(value: number): void {
  $chatLineSpacing.set(normalizeChatLineSpacing(value))
}

// Desktop-local presentation, independent of the window zoom and active profile.
// The percentage is published as a multiplier; styles.css applies it to the
// conversation's leading only (`--conversation-message-line-height`), so app
// chrome and the theme's own `--dt-line-height` baseline stay untouched.
if (typeof window !== 'undefined') {
  $chatLineSpacing.subscribe(value => {
    document.documentElement.style.setProperty('--chat-line-spacing', String(value / 100))
    persistString(KEY, value === DEFAULT_CHAT_LINE_SPACING ? null : String(value))
  })
}

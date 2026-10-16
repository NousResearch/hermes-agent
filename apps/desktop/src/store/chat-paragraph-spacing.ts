import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.chat-paragraph-spacing.v1'

// 100% is the pre-setting rendering exactly: whatever gap the surface painted
// (`--paragraph-gap-base` — the app's own, or the HUD dock's tighter one) reads
// unchanged, so this control is opt-in by construction.
const DEFAULT_CHAT_PARAGRAPH_SPACING = 100

// Steps widen above the 100% no-op. Near the default a 25% nudge is a hair
// (0.7rem → 0.875rem of gap), while the top of the ladder has to buy a real
// paragraph break: with a 21.45px line box, 200% (1.4rem / 22.4px) is barely
// one line and still reads tight, so the climb continues in 50% steps to 250%
// (1.75rem / 28px ≈ 1.3 lines). Five presets fit the Appearance row's action
// column (`minmax(15rem, 22rem)` → 240–352px) on the default single-row track —
// the same track the Chat Line Spacing row beside it ships — see the row in
// `appearance-settings.tsx`.
export const CHAT_PARAGRAPH_SPACING_PRESETS = [75, 100, 150, 200, 250] as const
export type ChatParagraphSpacing = (typeof CHAT_PARAGRAPH_SPACING_PRESETS)[number]

export const CHAT_PARAGRAPH_SPACING_MIN = CHAT_PARAGRAPH_SPACING_PRESETS[0]
export const CHAT_PARAGRAPH_SPACING_MAX = CHAT_PARAGRAPH_SPACING_PRESETS[CHAT_PARAGRAPH_SPACING_PRESETS.length - 1]

function normalizeChatParagraphSpacing(value: unknown): ChatParagraphSpacing {
  return CHAT_PARAGRAPH_SPACING_PRESETS.find(preset => preset === Number(value)) ?? DEFAULT_CHAT_PARAGRAPH_SPACING
}

export const $chatParagraphSpacing = atom<ChatParagraphSpacing>(normalizeChatParagraphSpacing(storedString(KEY)))

export function setChatParagraphSpacing(value: number): void {
  $chatParagraphSpacing.set(normalizeChatParagraphSpacing(value))
}

// Desktop-local presentation, independent of the window zoom and active profile.
// The percentage is published as a multiplier; styles.css applies it inside the
// transcript to the gap that surface already painted (`--paragraph-gap-base`), so
// the app's own base and the HUD dock's tighter rhythm each keep their 100% and
// neither override clobbers the other.
if (typeof window !== 'undefined') {
  $chatParagraphSpacing.subscribe(value => {
    document.documentElement.style.setProperty('--chat-paragraph-spacing', String(value / 100))
    persistString(KEY, value === DEFAULT_CHAT_PARAGRAPH_SPACING ? null : String(value))
  })
}

/**
 * Chat text size — an accessibility lever for the surface users read for hours.
 *
 * The conversation transcript sizes are hardcoded rem values in styles.css
 * (`--conversation-text-font-size` 0.8125rem, `--conversation-caption-font-size`
 * 0.75rem, and the caption line-height 1rem that must scale with them). This
 * preference paints percentage overrides on :root; the presentation-only
 * renderer owns it (desktop AGENTS.md: state lives with its authority).
 *
 * Default (100) paints nothing, so a user who never moved the lever gets
 * byte-identical CSS to before. The caption derives from the same scale so
 * the two stay in their designed proportion at every step.
 */

import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.chatTextSize.v1'

/** Exposed for tests asserting persistence; the key itself is private. */
export const CHAT_TEXT_SIZE_STORAGE_KEY = KEY

export const CHAT_TEXT_SIZES = ['90', '100', '110', '125', '150'] as const
export type ChatTextSize = (typeof CHAT_TEXT_SIZES)[number]
export const DEFAULT_CHAT_TEXT_SIZE: ChatTextSize = '100'

/** Percent → the caption scale keeps its designed 12/13 proportion. */
const CAPTION_RATIO = 0.75 / 0.8125

export function normalizeChatTextSize(value: unknown): ChatTextSize {
  const raw = typeof value === 'string' ? value.trim() : ''

  return (CHAT_TEXT_SIZES as readonly string[]).includes(raw) ? (raw as ChatTextSize) : DEFAULT_CHAT_TEXT_SIZE
}

export const $chatTextSize = atom<ChatTextSize>(
  typeof window === 'undefined' ? DEFAULT_CHAT_TEXT_SIZE : normalizeChatTextSize(storedString(KEY))
)

export function setChatTextSize(size: ChatTextSize): void {
  $chatTextSize.set(normalizeChatTextSize(size))
}

if (typeof window !== 'undefined') {
  $chatTextSize.subscribe(size => {
    const root = document.documentElement

    // The default paints nothing: styles.css keeps the hardcoded rem values,
    // so a fresh install renders exactly as before.
    if (size === DEFAULT_CHAT_TEXT_SIZE) {
      root.style.removeProperty('--conversation-text-font-size')
      root.style.removeProperty('--conversation-caption-font-size')
      root.style.removeProperty('--conversation-caption-line-height')
      persistString(KEY, null)

      return
    }

    const scale = Number(size) / 100

    root.style.setProperty('--conversation-text-font-size', `${(0.8125 * scale).toFixed(4)}rem`)
    root.style.setProperty('--conversation-caption-font-size', `${(0.75 * scale).toFixed(4)}rem`)
    root.style.setProperty('--conversation-caption-line-height', `${(1 * scale).toFixed(4)}rem`)
    persistString(KEY, size)
  })
}

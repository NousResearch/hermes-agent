import { type Codec, persistentAtom } from '@/lib/persisted'

export type ChatWidth = 'narrow' | 'default' | 'wide' | 'full'

const STORAGE_KEY = 'hermes.desktop.chatWidth'

/**
 * Chat column width.
 *
 * The transcript, composer, and intro all key off one CSS variable,
 * `--composer-width` (see styles.css). `default` is the stylesheet value
 * (100% — full bleed), so it needs no override; the other presets write a
 * capped max-width onto :root that every consumer picks up at once.
 */
const CHAT_WIDTH_VALUES = new Set<ChatWidth>(['narrow', 'default', 'wide', 'full'])

/** CSS max-width applied per preset. rem caps read well on any display; the
 *  wide preset scales with the window via min(). */
export const CHAT_WIDTH_CSS: Record<ChatWidth, null | string> = {
  narrow: '44rem',
  default: null,
  full: null,
  wide: 'min(72rem, 90vw)'
}

const chatWidthCodec: Codec<ChatWidth> = {
  decode: raw => (CHAT_WIDTH_VALUES.has(raw as ChatWidth) ? (raw as ChatWidth) : 'default'),
  encode: value => value
}

export const $chatWidth = persistentAtom<ChatWidth>(STORAGE_KEY, 'default', chatWidthCodec)

/** Apply (or clear) the :root override for the current preset. Idempotent. */
export function applyChatWidth(width: ChatWidth): void {
  const css = CHAT_WIDTH_CSS[width]

  if (css) {
    document.documentElement.style.setProperty('--composer-width', css)
  } else {
    document.documentElement.style.removeProperty('--composer-width')
  }
}

export function setChatWidth(width: ChatWidth): void {
  $chatWidth.set(width)
  applyChatWidth(width)
}

// Re-apply on load and whenever the preference changes from anywhere.
if (typeof window !== 'undefined') {
  applyChatWidth($chatWidth.get())
  $chatWidth.subscribe(applyChatWidth)
}

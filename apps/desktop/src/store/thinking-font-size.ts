import { type Codec, persistentAtom } from '@/lib/persisted'

export const THINKING_FONT_SIZE_MIN = 11
export const THINKING_FONT_SIZE_MAX = 15
export const THINKING_FONT_SIZE_DEFAULT = 12

const STORAGE_KEY = 'hermes.desktop.thinkingFontSizePx'

export function normalizeThinkingFontSize(value: unknown): number {
  const parsed = typeof value === 'number' ? value : Number(value)

  if (!Number.isFinite(parsed)) {
    return THINKING_FONT_SIZE_DEFAULT
  }

  return Math.min(THINKING_FONT_SIZE_MAX, Math.max(THINKING_FONT_SIZE_MIN, Math.round(parsed)))
}

export const thinkingFontSizeCodec: Codec<number> = {
  decode: raw => normalizeThinkingFontSize(raw),
  encode: value => String(normalizeThinkingFontSize(value))
}

export const $thinkingFontSize = persistentAtom<number>(
  STORAGE_KEY,
  THINKING_FONT_SIZE_DEFAULT,
  thinkingFontSizeCodec
)

function applyThinkingFontSize(value: number) {
  if (typeof document === 'undefined') {
    return
  }

  document.documentElement.style.setProperty('--conversation-tool-font-size', `${normalizeThinkingFontSize(value)}px`)
}

$thinkingFontSize.subscribe(applyThinkingFontSize)

export function setThinkingFontSize(value: number) {
  $thinkingFontSize.set(normalizeThinkingFontSize(value))
}

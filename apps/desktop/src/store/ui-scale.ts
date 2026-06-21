/**
 * User display scale — conversation text size + chat column width.
 *
 * Two independent levers, each a small set of named steps mapped to a unitless
 * multiplier. styles.css defines `--user-text-scale` / `--user-chat-width-scale`
 * at 1 (so an un-run build keeps upstream sizing) and multiplies every
 * conversation font / line-height / the composer width by them. This store
 * persists the chosen step and writes the multiplier onto documentElement at
 * boot, so the preference survives reloads and self-updates (localStorage lives
 * in userData, never in the git tree).
 *
 * Mirrors store/translucency.ts: renderer owns the value, applies it as a CSS
 * var. No IPC needed — this is pure renderer styling.
 */

import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

export type TextScaleId = 'compact' | 'default' | 'large' | 'huge'
export type ChatWidthId = 'cozy' | 'wide' | 'full'

const TEXT_SCALE_KEY = 'hermes.desktop.textScale.v1'
const CHAT_WIDTH_KEY = 'hermes.desktop.chatWidth.v1'

// Step → multiplier. Body text is 0.8125rem (13px) at 1.0; 'large' lands ~15px,
// 'huge' ~17.5px. Composer width is 48.75rem (780px) at 1.0; 'wide' ~1170px,
// 'full' ~1560px.
export const TEXT_SCALE_VALUES: Record<TextScaleId, number> = {
  compact: 0.92,
  default: 1,
  large: 1.18,
  huge: 1.35
}

export const CHAT_WIDTH_VALUES: Record<ChatWidthId, number> = {
  cozy: 1,
  wide: 1.5,
  full: 2
}

// First-run defaults. Larger/wider than upstream on purpose — the common ask is
// "make it bigger"; anyone who wants the dense look picks 'default' / 'cozy'.
const DEFAULT_TEXT_SCALE: TextScaleId = 'large'
const DEFAULT_CHAT_WIDTH: ChatWidthId = 'wide'

const normalizeText = (value: null | string): TextScaleId =>
  value && value in TEXT_SCALE_VALUES ? (value as TextScaleId) : DEFAULT_TEXT_SCALE

const normalizeWidth = (value: null | string): ChatWidthId =>
  value && value in CHAT_WIDTH_VALUES ? (value as ChatWidthId) : DEFAULT_CHAT_WIDTH

export const $textScale = atom<TextScaleId>(
  typeof window === 'undefined' ? DEFAULT_TEXT_SCALE : normalizeText(storedString(TEXT_SCALE_KEY))
)

export const $chatWidth = atom<ChatWidthId>(
  typeof window === 'undefined' ? DEFAULT_CHAT_WIDTH : normalizeWidth(storedString(CHAT_WIDTH_KEY))
)

export function setTextScale(id: TextScaleId): void {
  $textScale.set(id in TEXT_SCALE_VALUES ? id : DEFAULT_TEXT_SCALE)
}

export function setChatWidth(id: ChatWidthId): void {
  $chatWidth.set(id in CHAT_WIDTH_VALUES ? id : DEFAULT_CHAT_WIDTH)
}

if (typeof window !== 'undefined') {
  const root = document.documentElement

  $textScale.subscribe(id => {
    persistString(TEXT_SCALE_KEY, id)
    root.style.setProperty('--user-text-scale', String(TEXT_SCALE_VALUES[id]))
  })

  $chatWidth.subscribe(id => {
    persistString(CHAT_WIDTH_KEY, id)
    root.style.setProperty('--user-chat-width-scale', String(CHAT_WIDTH_VALUES[id]))
  })
}

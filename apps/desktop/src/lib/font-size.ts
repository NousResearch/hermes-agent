import type { HermesConfig } from '@/types/hermes'

const MIN_FONT_SIZE = 10
const MAX_FONT_SIZE = 32

export function normalizeUiFontSize(value: unknown): number {
  const n = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(n)) return 0
  const rounded = Math.round(n)
  if (rounded <= 0) return 0
  return Math.min(Math.max(rounded, MIN_FONT_SIZE), MAX_FONT_SIZE)
}

export function applyConfiguredFontSize(config: HermesConfig): number {
  const fontSize = normalizeUiFontSize(config.display?.font_size)
  const root = document.documentElement
  if (fontSize > 0) {
    root.style.fontSize = `${fontSize}px`
  } else {
    root.style.removeProperty('font-size')
  }
  return fontSize
}

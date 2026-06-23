import { atom } from 'nanostores'
import { useStore } from '@nanostores/react'

export const FONT_OPTIONS = [
  {
    id: 'system',
    label: 'System',
    value: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
  },
  { id: 'lora', label: 'Lora', value: '"Lora", system-ui, serif' },
  { id: 'space-grotesk', label: 'Space Grotesk', value: '"Space Grotesk", system-ui, sans-serif' },
  {
    id: 'jetbrains',
    label: 'JetBrains Mono',
    value:
      '"JetBrains Mono", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
  }
] as const

export const FONT_SIZE_OPTIONS = [
  { id: 'small', label: 'Small', textSize: '0.75rem', toolSize: '0.625rem', captionSize: '0.6875rem' },
  { id: 'medium', label: 'Medium', textSize: '0.8125rem', toolSize: '0.6875rem', captionSize: '0.75rem' },
  { id: 'large', label: 'Large', textSize: '0.875rem', toolSize: '0.75rem', captionSize: '0.8125rem' },
  { id: 'extraLarge', label: 'Extra Large', textSize: '1rem', toolSize: '0.875rem', captionSize: '0.9375rem' }
] as const

export type FontOption = (typeof FONT_OPTIONS)[number]['id']
export type FontSizeOption = (typeof FONT_SIZE_OPTIONS)[number]['id']

export const fontStore = atom<FontOption>('system')
export const fontSizeStore = atom<FontSizeOption>('medium')

export function useFont() {
  const font = useStore(fontStore)
  const fontSize = useStore(fontSizeStore)

  return {
    font,
    setFont: (value: FontOption) => {
      fontStore.set(value)
      localStorage.setItem('hermes-desktop-font', value)
    },
    fontSize,
    setFontSize: (value: FontSizeOption) => {
      fontSizeStore.set(value)
      localStorage.setItem('hermes-desktop-font-size', value)
    }
  }
}

export function initializeFontSettings() {
  const savedFont = localStorage.getItem('hermes-desktop-font')
  const savedFontSize = localStorage.getItem('hermes-desktop-font-size')

  const validFont = savedFont && FONT_OPTIONS.some(opt => opt.id === savedFont) ? (savedFont as FontOption) : null
  const validFontSize =
    savedFontSize && FONT_SIZE_OPTIONS.some(opt => opt.id === savedFontSize) ? (savedFontSize as FontSizeOption) : null

  if (validFont) {
    fontStore.set(validFont)
  }

  if (validFontSize) {
    fontSizeStore.set(validFontSize)
  }
}

// Hydrate stores at module-load time so any importer — including
// context.tsx's boot-time paint — sees persisted values, not defaults.
if (typeof window !== 'undefined') {
  initializeFontSettings()
}

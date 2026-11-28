import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.terminal-font-size.v1'
export const DEFAULT_TERMINAL_FONT_SIZE = 11

export const TERMINAL_FONT_SIZE_PRESETS = [10, 11, 12, 13, 14, 16, 18] as const
export type TerminalFontSize = (typeof TERMINAL_FONT_SIZE_PRESETS)[number]

function normalizeTerminalFontSize(value: unknown): TerminalFontSize {
  return TERMINAL_FONT_SIZE_PRESETS.find(preset => preset === Number(value)) ?? DEFAULT_TERMINAL_FONT_SIZE
}

export const $terminalFontSize = atom<TerminalFontSize>(normalizeTerminalFontSize(storedString(KEY)))

export function setTerminalFontSize(value: number): void {
  $terminalFontSize.set(normalizeTerminalFontSize(value))
}

if (typeof window !== 'undefined') {
  $terminalFontSize.subscribe(value => persistString(KEY, value === DEFAULT_TERMINAL_FONT_SIZE ? null : String(value)))
}

import type { ITheme, Terminal } from '@xterm/xterm'
import type { CSSProperties } from 'react'

// Solarized-derived palette, but with bright ANSI 8–15 promoted to real
// accent variants instead of Schoonover's UI grays. Hermes' TUI skins (gold,
// crimson, ...) emit bright SGR codes that would otherwise wash out to gray.
// We always render the dark canvas — the app's light surfaces can't host the
// default skin without dropping below readable contrast.
export const TERMINAL_BG = '#002b36'

const THEME: ITheme = {
  background: TERMINAL_BG,
  foreground: '#839496',
  cursor: '#93a1a1',
  cursorAccent: TERMINAL_BG,
  selectionBackground: '#586e7555',
  black: '#073642',
  red: '#dc322f',
  green: '#859900',
  yellow: '#b58900',
  blue: '#268bd2',
  magenta: '#d33682',
  cyan: '#2aa198',
  white: '#eee8d5',
  brightBlack: '#586e75',
  brightRed: '#f25c54',
  brightGreen: '#b3d437',
  brightYellow: '#f7c948',
  brightBlue: '#5fb3ff',
  brightMagenta: '#ff6ab4',
  brightCyan: '#5cd9c8',
  brightWhite: '#fdf6e3'
}

export const terminalTheme = (): ITheme => THEME

export const isMacPlatform = () => navigator.platform.toLowerCase().includes('mac')

export const addSelectionShortcutLabel = () => (isMacPlatform() ? '⌘L' : 'Ctrl+L')

export function isAddSelectionShortcut(event: KeyboardEvent) {
  const mod = isMacPlatform() ? event.metaKey : event.ctrlKey

  return mod && !event.shiftKey && event.key.toLowerCase() === 'l'
}

export function terminalSelectionLabel(term: Terminal, shellName: string, text: string) {
  const pos = term.getSelectionPosition()

  if (pos) {
    return pos.start.y === pos.end.y ? `${shellName}:${pos.start.y}` : `${shellName}:${pos.start.y}-${pos.end.y}`
  }

  const lines = Math.max(1, text.trim().split(/\r?\n/).length)

  return `${shellName}:${lines} line${lines === 1 ? '' : 's'}`
}

export function terminalSelectionAnchor(host: HTMLDivElement): CSSProperties | null {
  const rect = Array.from(host.querySelectorAll<HTMLElement>('.xterm-selection div'))
    .map(node => node.getBoundingClientRect())
    .filter(r => r.width > 0 && r.height > 0)
    .at(-1)

  if (!rect) {
    return null
  }

  const hostRect = host.getBoundingClientRect()
  const buttonWidth = 128
  const left = Math.min(Math.max(rect.left - hostRect.left, 8), Math.max(8, host.clientWidth - buttonWidth - 8))
  const top = Math.min(Math.max(rect.bottom - hostRect.top + 4, 8), Math.max(8, host.clientHeight - 34))

  return { left, top }
}

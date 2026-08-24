import { atom } from 'nanostores'

export const DEFAULT_TERMINAL_FONT_FAMILY = "'JetBrains Mono', 'Cascadia Code', 'SF Mono', Menlo, Consolas, monospace"

export const TERMINAL_FONT_SUGGESTIONS = [
  'MesloLGS NF',
  'JetBrainsMono Nerd Font',
  'CaskaydiaCove Nerd Font',
  'FiraCode Nerd Font',
  'Hack Nerd Font',
  'SauceCodePro Nerd Font',
  'JetBrains Mono',
  'SF Mono',
  'Menlo',
  'Cascadia Code'
] as const

/** The profile-backed value as written in config.yaml. Empty means bundled default. */
export const $terminalFontFamily = atom('')

export function normalizeTerminalFontFamily(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function quoteSingleFamily(value: string): string {
  return `'${value.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`
}

/** Accept a friendly single family name or an authored CSS font stack. */
export function resolveTerminalFontFamily(value: unknown): string {
  const configured = normalizeTerminalFontFamily(value)

  if (!configured) {
    return DEFAULT_TERMINAL_FONT_FAMILY
  }

  const preferred = configured.includes(',') || /['"]/.test(configured) ? configured : quoteSingleFamily(configured)

  return `${preferred}, ${DEFAULT_TERMINAL_FONT_FAMILY}`
}

export function setTerminalFontFamilyFromConfig(value: unknown): void {
  $terminalFontFamily.set(normalizeTerminalFontFamily(value))
}

type FontFaceLoader = Pick<FontFaceSet, 'load'>

function browserFontSet(): FontFaceLoader | undefined {
  return typeof document === 'undefined' ? undefined : document.fonts
}

/** Warm every face xterm uses before WebGL builds its glyph texture atlas. */
export async function warmTerminalFontFamily(
  fontFamily: string,
  fontSet: FontFaceLoader | undefined = browserFontSet()
): Promise<void> {
  if (!fontSet?.load) {
    return
  }

  await Promise.allSettled(
    ['400', '700', 'italic 400'].map(descriptor =>
      Promise.resolve().then(() => fontSet.load(`${descriptor} 11px ${fontFamily}`))
    )
  )
}

/**
 * Wait for the newest requested family before mounting xterm. Config can arrive
 * after the terminal component renders; this loop prevents opening WebGL with
 * stale fallback metrics and then immediately rebuilding it.
 */
export async function prepareTerminalFontFamily(
  getLatest: () => string,
  isCurrent: () => boolean,
  warm: (fontFamily: string) => Promise<void> = warmTerminalFontFamily
): Promise<string | null> {
  let candidate = getLatest()

  while (isCurrent()) {
    await warm(candidate)

    if (!isCurrent()) {
      return null
    }

    const latest = getLatest()

    if (latest === candidate) {
      return candidate
    }

    candidate = latest
  }

  return null
}

export interface TerminalFontTarget {
  options: { fontFamily?: string }
  rows: number
  refresh: (start: number, end: number) => void
}

interface ApplyTerminalFontOptions {
  clearTextureAtlas: () => void
  fit: () => void
  fontFamily: string
  isCurrent: () => boolean
  term: TerminalFontTarget
  warm?: (fontFamily: string) => Promise<void>
}

/** Apply a live font change without recreating the xterm instance or its PTY. */
export async function applyTerminalFontFamily({
  clearTextureAtlas,
  fit,
  fontFamily,
  isCurrent,
  term,
  warm = warmTerminalFontFamily
}: ApplyTerminalFontOptions): Promise<boolean> {
  await warm(fontFamily)

  if (!isCurrent()) {
    return false
  }

  term.options.fontFamily = fontFamily
  fit()
  clearTextureAtlas()

  if (term.rows > 0) {
    term.refresh(0, term.rows - 1)
  }

  return true
}

// Keybind combo normalization + display.
//
// A combo is a canonical lowercase string like "mod+k", "mod+shift+]", "shift+x",
// or "r". `mod` is Cmd on macOS / Ctrl elsewhere, so a single binding works on
// both. We derive the base key from `event.code` (not `event.key`) so Shift never
// mutates it ("shift+/" stays "shift+/" instead of becoming "shift+?").

export const IS_MAC =
  typeof navigator !== 'undefined' && /mac/i.test(navigator.platform || navigator.userAgent || '')

// event.code → canonical base token. Letters/digits map to their lowercase
// character; everything else uses an explicit name so combos read cleanly.
const CODE_TO_KEY: Record<string, string> = {
  Backquote: '`',
  Backslash: '\\',
  BracketLeft: '[',
  BracketRight: ']',
  Comma: ',',
  Equal: '=',
  Minus: '-',
  Period: '.',
  Quote: "'",
  Semicolon: ';',
  Slash: '/',
  Space: 'space',
  Enter: 'enter',
  Escape: 'escape',
  Backspace: 'backspace',
  Tab: 'tab',
  ArrowUp: 'up',
  ArrowDown: 'down',
  ArrowLeft: 'left',
  ArrowRight: 'right'
}

const MODIFIER_CODES = new Set([
  'AltLeft',
  'AltRight',
  'ControlLeft',
  'ControlRight',
  'MetaLeft',
  'MetaRight',
  'ShiftLeft',
  'ShiftRight'
])

function baseKeyFromCode(code: string): string | null {
  if (code.startsWith('Key')) {
    return code.slice(3).toLowerCase()
  }

  if (code.startsWith('Digit')) {
    return code.slice(5)
  }

  if (code.startsWith('Numpad')) {
    const rest = code.slice(6)

    return /^[0-9]$/.test(rest) ? rest : null
  }

  if (code.startsWith('F') && /^F\d{1,2}$/.test(code)) {
    return code.toLowerCase()
  }

  return CODE_TO_KEY[code] ?? null
}

// Returns the canonical combo for a keydown, or null while only modifiers are
// held (so capture mode keeps waiting for a real key).
export function comboFromEvent(event: KeyboardEvent): string | null {
  if (MODIFIER_CODES.has(event.code)) {
    return null
  }

  const base = baseKeyFromCode(event.code)

  if (!base) {
    return null
  }

  const parts: string[] = []

  if (event.metaKey || event.ctrlKey) {
    parts.push('mod')
  }

  if (event.altKey) {
    parts.push('alt')
  }

  if (event.shiftKey) {
    parts.push('shift')
  }

  parts.push(base)

  return parts.join('+')
}

const TOKEN_LABELS: Record<string, string> = {
  enter: '↵',
  escape: 'Esc',
  backspace: '⌫',
  tab: '⇥',
  space: 'Space',
  up: '↑',
  down: '↓',
  left: '←',
  right: '→'
}

function labelForBase(base: string): string {
  if (TOKEN_LABELS[base]) {
    return TOKEN_LABELS[base]
  }

  if (/^f\d{1,2}$/.test(base)) {
    return base.toUpperCase()
  }

  return base.length === 1 ? base.toUpperCase() : base
}

// Human-readable label, e.g. "⌘⇧K" on macOS, "Ctrl+Shift+K" elsewhere.
export function formatCombo(combo: string): string {
  const parts = combo.split('+')
  const base = parts.pop() ?? ''
  const mods = parts

  const modLabels = mods.map(mod => {
    if (mod === 'mod') {
      return IS_MAC ? '⌘' : 'Ctrl'
    }

    if (mod === 'alt') {
      return IS_MAC ? '⌥' : 'Alt'
    }

    if (mod === 'shift') {
      return IS_MAC ? '⇧' : 'Shift'
    }

    return mod
  })

  const tokens = [...modLabels, labelForBase(base)]

  return IS_MAC ? tokens.join('') : tokens.join('+')
}

// True when focus is in a text-entry surface, so bare-key shortcuts don't fire
// while the user is typing.
export function isEditableTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null

  return Boolean(
    el?.isContentEditable ||
      el instanceof HTMLInputElement ||
      el instanceof HTMLTextAreaElement ||
      el instanceof HTMLSelectElement
  )
}

// Combos with a primary modifier (Cmd/Ctrl) are safe to fire even while typing
// (e.g. ⌘K from the composer); bare/Shift-only combos are suppressed in inputs.
export function comboAllowedInInput(combo: string): boolean {
  return combo.startsWith('mod+') || combo === 'mod'
}

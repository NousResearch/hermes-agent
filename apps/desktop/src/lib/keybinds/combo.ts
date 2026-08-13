// Keybind combo normalization + display.
//
// A combo is a canonical lowercase string like "mod+k", "mod+shift+]", "shift+x",
// or "r". `mod` is Cmd on macOS / Ctrl elsewhere, so a single binding works on
// both. We derive the base key from `event.key` where the layout matters
// (letters, unshifted punctuation) and from `event.code` otherwise, so a
// binding follows the character the user's layout actually types while Shift
// never mutates it ("shift+/" stays "shift+/" instead of becoming "shift+?").
//
// `ctrl` is physical Control, distinct from `mod`. It only matters on macOS,
// where `mod` is Cmd and Cmd+Tab is OS-reserved — so `ctrl+tab` is literally
// Control+Tab. Off macOS, Control already *is* `mod`, so `canonicalizeCombo`
// folds `ctrl` → `mod`.

export const IS_MAC = typeof navigator !== 'undefined' && /mac/i.test(navigator.platform || navigator.userAgent || '')

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

// Punctuation we ship as combo tokens, derived from CODE_TO_KEY so the two
// can't drift. Named tokens (space, tab, …) are excluded by the length check.
const PUNCTUATION_KEYS = new Set(Object.values(CODE_TO_KEY).filter(token => token.length === 1))

// The layout-aware half of the base key. `event.key` carries the character the
// user's layout actually produces, which is what a binding should match:
//
//   - Letters always win, shifted or not — `toLowerCase` normalizes the case.
//   - Punctuation only when Shift is UP, because a shifted `event.key` is the
//     shifted glyph ("?" for "/"), and combos stay anchored to the unshifted
//     token. Shifted punctuation falls through to `event.code` below.
//
// Digits deliberately stay physical: on AZERTY the number row is shifted, so
// `event.key` for the "1" key is "&" and only yields "1" with Shift held —
// `event.code` is what keeps `mod+1` reachable there.
//
// Anything else (Option glyphs like "˚", dead keys, non-Latin scripts) isn't a
// token we ship, so it fails both checks and falls back to the physical code.
function baseKeyFromEventKey(key: string, shiftKey: boolean): string | null {
  if (/^[a-z]$/i.test(key)) {
    return key.toLowerCase()
  }

  return !shiftKey && PUNCTUATION_KEYS.has(key) ? key : null
}

// Returns the canonical combo for a keydown, or null while only modifiers are
// held (so capture mode keeps waiting for a real key).
export function comboFromEvent(event: KeyboardEvent): string | null {
  if (MODIFIER_CODES.has(event.code)) {
    return null
  }

  const base = baseKeyFromEventKey(event.key, event.shiftKey) ?? baseKeyFromCode(event.code)

  if (!base) {
    return null
  }

  const parts: string[] = []

  // macOS reports Cmd (`mod`) and Control (`ctrl`) separately; elsewhere
  // Control IS the accelerator, so it folds into `mod`.
  if (event.metaKey || (event.ctrlKey && !IS_MAC)) {
    parts.push('mod')
  }

  if (event.ctrlKey && IS_MAC) {
    parts.push('ctrl')
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

// Rewrites a binding to the form `comboFromEvent` emits, so it indexes under
// the same key a live keypress produces. Off macOS, `ctrl+…` and `mod+…` are
// the one Control chord, so a shipped `ctrl+tab` matches a real Control+Tab.
export function canonicalizeCombo(combo: string): string {
  return IS_MAC ? combo : combo.replace(/\bctrl\b/g, 'mod')
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

function labelForMod(mod: string): string {
  if (mod === 'mod') {
    return IS_MAC ? '⌘' : 'Ctrl'
  }

  if (mod === 'ctrl') {
    return IS_MAC ? '⌃' : 'Ctrl'
  }

  if (mod === 'alt') {
    return IS_MAC ? '⌥' : 'Alt'
  }

  if (mod === 'shift') {
    return IS_MAC ? '⇧' : 'Shift'
  }

  return mod
}

// Per-key display tokens, e.g. ["⌘", "K"] on macOS, ["Ctrl", "K"] elsewhere —
// one cap per token for <KbdGroup>.
export function comboTokens(combo: string): string[] {
  const parts = combo.split('+')
  const base = parts.pop() ?? ''

  return [...parts.map(labelForMod), labelForBase(base)]
}

// Human-readable label, e.g. "⌘⇧K" on macOS, "Ctrl+Shift+K" elsewhere.
export function formatCombo(combo: string): string {
  const tokens = comboTokens(combo)

  return IS_MAC ? tokens.join('') : tokens.join('+')
}

// True when focus currently sits inside an element matching `selector`. The
// primitive for focus-scoped shortcuts — e.g. routing ⌘W to whichever surface
// (terminal, preview, …) owns focus.
export function isFocusWithin(selector: string): boolean {
  return document.activeElement?.closest(selector) != null
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

const INPUT_SAFE_ACTIONS = new Set([
  'composer.modelPicker',
  'composer.voice',
  'keybinds.openPanel',
  'nav.commandPalette',
  'session.next',
  'session.prev',
  'view.findInPage'
])

const TEXT_NAVIGATION_KEYS = new Set(['up', 'down', 'left', 'right', 'home', 'end', 'pageup', 'pagedown'])

// Only explicit text-entry-safe actions fire while typing. Editing/navigation
// chords such as Ctrl+Arrow/PageUp must stay with the input even if a user
// rebinds them to a global navigation action.
export function actionAllowedInInput(actionId: string, combo: string): boolean {
  const base = combo.split('+').pop()

  if (base && TEXT_NAVIGATION_KEYS.has(base)) {
    return false
  }

  return INPUT_SAFE_ACTIONS.has(actionId)
}

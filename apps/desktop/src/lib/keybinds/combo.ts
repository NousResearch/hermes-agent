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
  PageUp: 'pageup',
  PageDown: 'pagedown',
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

// Modifier names as reported by `event.key` on a bare modifier keydown.
const MODIFIER_KEYS = new Set(['Alt', 'Control', 'Meta', 'Shift'])

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
  // IME composition (Chinese/Japanese/Korean input): the keydown events
  // during composition carry preedit keystrokes and the commit keypress
  // (Enter/Space/Shift for candidate selection). Treating them as combos
  // fires unrelated keybinds — e.g. typing 你 with a Chinese IME sent a
  // keydown that dispatched `session.new` and silently opened a new session.
  // Bail out entirely while composing.
  if (event.isComposing || event.key === 'Process') {
    return null
  }

  if (MODIFIER_CODES.has(event.code)) {
    return null
  }

  // A keydown whose `key` is a modifier name but whose `code` is a regular
  // key is not a real modifier chord — legacy IMEs that synthesize keystrokes
  // (Q9 2002 sends key="Control" with code="KeyW") produce these, and they
  // would canonicalize to phantom combos (Ctrl+W → close active tab). Ignore.
  if (MODIFIER_KEYS.has(event.key)) {
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
  pageup: 'PgUp',
  pagedown: 'PgDn',
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

export function formatModifierToken(mod: string): string {
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

  return [...parts.map(formatModifierToken), labelForBase(base)]
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

// Only explicit text-entry-safe actions fire while typing. A primary-modifier
// chord (Cmd/Ctrl) is a deliberate two-key gesture that every browser and chat
// app fires even with focus in a text field (⌘N, ⌘T, ⌘K, ⌃Tab…), so those stay
// global — restoring the pre-#86586 behavior. Editing/navigation chords such
// as Ctrl+Arrow/PageUp must stay with the input even if a user rebinds them to
// a global navigation action, and bare/Shift-only combos (typed letters) are
// gated by the allowlist so they never hijack normal typing.
export function actionAllowedInInput(actionId: string, combo: string): boolean {
  const base = combo.split('+').pop()

  // A bare modifier (no key) is not a real chord — `comboFromEvent` never
  // yields one, but reject it here so a malformed stored binding can't pass
  // the shape-only mod/ctrl check below.
  if (!base || base === 'mod' || base === 'ctrl' || TEXT_NAVIGATION_KEYS.has(base)) {
    return false
  }

  if (/^(?:mod|ctrl)(?:\+|$)/.test(combo)) {
    return true
  }

  return INPUT_SAFE_ACTIONS.has(actionId)
}

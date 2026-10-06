export const BROWSER_TAB_ACTIONS = [
  'session.newTab',
  'view.closeTab',
  'session.next',
  'session.prev',
  'view.findInPage'
] as const

interface BrowserKeyInput {
  type: string
  key: string
  code?: string
  control: boolean
  meta: boolean
  alt: boolean
  shift: boolean
  isAutoRepeat?: boolean
  isComposing?: boolean
}

/** Only configured, modified chords. Text and IME input always belong to the
 * page, including editable guests. No application/global shortcut registration. */
export function browserTabShortcut(
  input: BrowserKeyInput,
  bindings: Record<string, string[]>,
  mac: boolean
): string | null {
  if (
    input.type !== 'keyDown' ||
    input.isAutoRepeat ||
    input.isComposing ||
    input.key === 'Process' ||
    ['Alt', 'Control', 'Meta', 'Shift'].includes(input.key) ||
    !(input.control || input.meta || input.alt)
  ) {
    return null
  }

  const aliases: Record<string, string> = {
    ArrowLeft: 'left',
    ArrowRight: 'right',
    ArrowUp: 'up',
    ArrowDown: 'down',
    ' ': 'space'
  }

  const code = input.code || ''

  const punctuation: Record<string, string> = {
    BracketLeft: '[',
    BracketRight: ']',
    Slash: '/',
    Backslash: '\\',
    Semicolon: ';',
    Quote: "'",
    Comma: ',',
    Period: '.',
    Minus: '-',
    Equal: '=',
    Backquote: '`'
  }

  const key = /^[a-z]$/i.test(input.key)
    ? input.key.toLowerCase()
    : code.startsWith('Digit')
      ? code.slice(5)
      : (input.shift && punctuation[code]) || aliases[input.key] || input.key.toLowerCase()

  const parts = [
    mac ? input.meta && 'mod' : input.control && 'mod',
    mac && input.control && 'ctrl',
    !mac && input.meta && 'meta',
    input.alt && 'alt',
    input.shift && 'shift',
    key
  ].filter(Boolean)

  const combo = parts.join('+')

  for (const action of BROWSER_TAB_ACTIONS) {
    if (bindings[action]?.some(binding => (mac ? binding : binding.replace(/\bctrl\b/g, 'mod')) === combo)) {
      return action
    }
  }

  return null
}

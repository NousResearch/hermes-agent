export interface LinuxWindowButtons {
  left: number
  right: number
}

/** Today's assumption when gsettings cannot tell us a layout: all three on the right. */
export const DEFAULT_LINUX_BUTTON_LAYOUT: LinuxWindowButtons = { left: 0, right: 3 }

const CONTROLLED_BUTTONS = new Set(['close', 'minimize', 'maximize'])

function stripQuotes(value: string): string {
  const quote = value[0]

  if (value.length >= 2 && (quote === "'" || quote === '"') && value.endsWith(quote)) {
    return value.slice(1, -1)
  }

  return value
}

function countButtons(value: string): number {
  let count = 0

  for (const token of value.split(',')) {
    if (CONTROLLED_BUTTONS.has(token.trim())) {
      count += 1
    }
  }

  return count
}

/**
 * Parse `gsettings get org.gnome.desktop.wm.preferences button-layout`.
 *
 * The value is a string like `'close,minimize,maximize:'` or
 * `'appmenu:minimize,maximize,close'`: buttons before the FIRST `:` sit on the
 * left, buttons after it on the right. Only close/minimize/maximize count —
 * appmenu, spacer, icon and unknown tokens are ignored. Any value that is not
 * a colon-separated layout (including gsettings failures and empty output)
 * falls back to today's all-on-the-right assumption.
 */
export function parseLinuxButtonLayout(raw: unknown): LinuxWindowButtons {
  if (typeof raw !== 'string') {
    return { ...DEFAULT_LINUX_BUTTON_LAYOUT }
  }

  const value = stripQuotes(raw.trim())
  const separator = value.indexOf(':')

  if (separator < 0) {
    return { ...DEFAULT_LINUX_BUTTON_LAYOUT }
  }

  return {
    left: countButtons(value.slice(0, separator)),
    right: countButtons(value.slice(separator + 1))
  }
}

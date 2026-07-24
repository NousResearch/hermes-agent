export interface CloseTabShortcutInput {
  alt?: boolean
  control?: boolean
  key?: string
  meta?: boolean
  shift?: boolean
}

interface WindowCloseTarget {
  close: () => void
}

/** Keep Window > Close without taking Ctrl+W from the renderer. */
export function createClickOnlyWindowCloseItem() {
  return {
    click: (_menuItem: unknown, window?: WindowCloseTarget) => window?.close(),
    label: 'Close'
  }
}

/** Main-process routing is only needed for macOS Cmd+W. */
export function shouldInterceptCloseTabShortcut(input: CloseTabShortcutInput, isMac: boolean): boolean {
  return Boolean(isMac && String(input.key ?? '').toLowerCase() === 'w' && input.meta && !input.alt && !input.shift)
}

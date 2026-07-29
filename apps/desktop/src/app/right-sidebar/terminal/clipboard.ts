// Clipboard keybindings for the GUI terminal.
//
// xterm renders to a canvas, so its selection is not a DOM selection and the
// platform's own copy command has nothing to grab. Two mechanisms fix that:
// this key map (explicit chords) and `mirrorSelection` below (which hands the
// selection to the OS through xterm's hidden helper textarea, so the Edit menu,
// ⌘C on macOS — swallowed by the menu before the renderer sees it — and the
// right-click menu all work).
//
// The chords follow VS Code (terminal.clipboard.contribution.ts): ⌘C/⌘V on
// macOS, Ctrl+Shift+C/V elsewhere, plus plain Ctrl+C as copy ONLY when text is
// selected — the "intelligent Ctrl-C" of Windows Terminal and Tabby. With no
// selection Ctrl+C stays SIGINT, so interrupting a process never breaks.

export type TerminalClipboardIntent = 'copy' | 'paste' | null

export function terminalClipboardIntent(
  event: KeyboardEvent,
  { hasSelection, isMac }: { hasSelection: boolean; isMac: boolean }
): TerminalClipboardIntent {
  if (event.type !== 'keydown' || event.altKey) {
    return null
  }

  const key = event.key.toLowerCase()

  if (isMac) {
    if (!event.metaKey || event.ctrlKey || event.shiftKey) {
      return null
    }

    // ⌘C with nothing selected falls through to the shell (⌘ isn't a terminal
    // modifier, so it's a no-op there rather than a lost keystroke).
    return key === 'c' ? (hasSelection ? 'copy' : null) : key === 'v' ? 'paste' : null
  }

  if (!event.ctrlKey || event.metaKey) {
    return null
  }

  if (event.shiftKey) {
    return key === 'c' ? (hasSelection ? 'copy' : null) : key === 'v' ? 'paste' : null
  }

  // Bare Ctrl+C: copy only when there's a selection to copy, else SIGINT.
  return key === 'c' && hasSelection ? 'copy' : null
}

// Hand the terminal's selection to the OS by mirroring it into xterm's hidden
// helper textarea (the same trick xterm uses for Linux middle-click paste,
// CoreBrowserTerminal.ts:531). Without it `webContents.copy()` — what the Edit
// menu, ⌘C, and the right-click Copy item all call — finds no DOM selection and
// copies nothing.
export function mirrorSelection(host: HTMLElement, text: string) {
  const textarea = host.querySelector<HTMLTextAreaElement>('.xterm-helper-textarea')

  if (!textarea) {
    return
  }

  if (!text) {
    textarea.value = ''

    return
  }

  textarea.value = text
  textarea.select()
}

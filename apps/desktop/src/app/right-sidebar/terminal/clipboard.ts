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

interface Osc52Terminal {
  parser: {
    registerOscHandler(identifier: number, handler: (data: string) => boolean): { dispose(): void }
  }
}

// Support only OSC 52 writes to the system clipboard. Reads would let terminal
// output retrieve private clipboard contents; empty payloads are clear requests.
export function installOsc52ClipboardHandler(
  terminal: Osc52Terminal,
  writeClipboardText: (text: string) => Promise<unknown>
) {
  return terminal.parser.registerOscHandler(52, data => {
    const separator = data.indexOf(';')

    if (separator < 0 || data.slice(0, separator) !== 'c') {
      return false
    }

    const payload = data.slice(separator + 1)

    if (!payload || payload === '?') {
      return false
    }

    try {
      const binary = atob(payload)
      const bytes = Uint8Array.from(binary, character => character.charCodeAt(0))
      const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes)

      void writeClipboardText(text).catch(() => {
        // Clipboard access may be denied by the platform or user settings.
      })

      return true
    } catch {
      return false
    }
  })
}

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
//
// `textarea.select()` replaces the document's live range. Only claim it while
// the terminal owns focus AND nothing outside the terminal is highlighted —
// otherwise a leftover terminal scrap wins ⌘C over text the user just
// selected in chat. The terminal's own ⌘C key handler still copies via
// writeClipboardText when focus is on the canvas path.
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

  if (!host.contains(document.activeElement)) {
    return
  }

  const live = window.getSelection()

  const foreign = live && !live.isCollapsed && live.anchorNode != null && !host.contains(live.anchorNode)

  if (foreign) {
    return
  }

  textarea.select()
}

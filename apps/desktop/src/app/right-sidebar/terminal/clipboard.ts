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
// macOS, Ctrl+Shift+C/V everywhere, plus plain Ctrl+C as copy ONLY when text is
// selected — the "intelligent Ctrl-C" of Windows Terminal and Tabby. With no
// selection Ctrl+C stays SIGINT, so interrupting a process never breaks.
//
// This map is the ONLY paste route on Windows, so its gaps are dead keys rather
// than fallbacks. Electron installs an application menu on macOS alone
// (main.ts: `IS_MAC ? setApplicationMenu(buildApplicationMenu()) :
// setApplicationMenu(null)`), and — the same trap the Edit submenu's
// `pasteAndMatchStyle` comment records — an accelerator with no menu entry is
// never translated into an editor command. Without a `role: 'paste'` on
// Windows/Linux, Chromium never runs Paste for Ctrl+V or Shift+Insert, so no
// `paste` event ever reaches the listener xterm puts on its helper textarea.
// Anything this function returns null for is a keystroke that lands on nothing.
//
// Hence the two chords below beyond VS Code's Linux row:
//   • Shift+Insert — every platform. xterm deliberately declines it
//     (Keyboard.ts case 45: "used to copy-paste on some systems") expecting the
//     host to paste, and no shell binds it. Dictation tools reach for it first.
//   • Ctrl+V — Windows only, matching VS Code, whose terminal binds Ctrl+V as
//     the PRIMARY paste there with Ctrl+Shift+V secondary; the Ctrl+Shift-only
//     row is Linux's, where Ctrl+V stays readline's quoted-insert.

export type TerminalClipboardIntent = 'copy' | 'paste' | null

export function terminalClipboardIntent(
  event: KeyboardEvent,
  { hasSelection, isMac, isWindows }: { hasSelection: boolean; isMac: boolean; isWindows: boolean }
): TerminalClipboardIntent {
  if (event.type !== 'keydown' || event.altKey) {
    return null
  }

  const key = event.key.toLowerCase()

  // Ahead of every Ctrl gate: Shift+Insert carries no Ctrl to gate on.
  if (key === 'insert' && event.shiftKey && !event.ctrlKey && !event.metaKey) {
    return 'paste'
  }

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

  // Bare Ctrl+V: paste on Windows (VS Code's primary chord there), left to the
  // shell on Linux so readline keeps quoted-insert.
  if (isWindows && key === 'v') {
    return 'paste'
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

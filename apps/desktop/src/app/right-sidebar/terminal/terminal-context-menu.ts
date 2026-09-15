/**
 * Verb registry for the GUI terminals.
 *
 * xterm paints to a canvas: a right-click inside it crosses no link, image,
 * editable, or DOM selection, so the app context menu would see a bare
 * surface and offer the window verbs. Focus-routed chords hit the same blind
 * spot from the other side: main claims Ctrl/Cmd+R on every window
 * (before-input-event + preventDefault) so Chromium's default host reload
 * cannot fire, and that claim also stops the keystroke before xterm sees it.
 * Both paths resolve through a handle registered for the terminal's
 * `[data-terminal]` scope.
 */

export interface TerminalMenuHandle {
  getSelection: () => string
  /** Null on the read-only agent mirror — it has no PTY to paste into. */
  paste: ((text: string) => void) | null
  /** Re-deliver the Ctrl/Cmd+R chord main claimed (#96482). User terminals
   *  write the ^R byte into their PTY (readline reverse-i-search). Read-only
   *  agent mirrors have no PTY, so they swallow it — focus is still in a
   *  terminal, and the app-level reload fallback must not fire there. */
  reload: () => void
  selectAll: () => void
}

const handles = new WeakMap<Element, TerminalMenuHandle>()

/** Register the handle for a terminal host. The registry keys by the
 *  `[data-terminal]` scope above the host: both resolvers start from a click
 *  target or the focused element and walk up to that scope, while the xterm
 *  host is a nested div inside it (instance.tsx). Returns an idempotent
 *  remove. */
export function registerTerminalContextMenu(host: Element, handle: TerminalMenuHandle): () => void {
  const scope = host.closest('[data-terminal]') ?? host
  handles.set(scope, handle)

  return () => {
    if (handles.get(scope) === handle) {
      handles.delete(scope)
    }
  }
}

/** The handle owning `element`, when the click landed inside a terminal. */
export function terminalMenuHandleFor(element: Element | null): TerminalMenuHandle | null {
  const host = element?.closest('[data-terminal]')

  return host ? (handles.get(host) ?? null) : null
}

/** Run a focus-routed chord on the terminal holding DOM focus — the terminal
 *  rung of the preview-nav routing (see commandFocusedPreview in
 *  chat/right-rail/preview-nav.ts). Only `reload` is a terminal verb; `back`
 *  and `forward` mean nothing there. False = focus is elsewhere in the app,
 *  so the caller falls back to the app-level meaning. */
export function commandFocusedTerminal(command: 'back' | 'forward' | 'reload'): boolean {
  if (command !== 'reload') {
    return false
  }

  const host = document.activeElement?.closest('[data-terminal]')
  const handle = host ? handles.get(host) : undefined

  handle?.reload()

  return Boolean(handle)
}

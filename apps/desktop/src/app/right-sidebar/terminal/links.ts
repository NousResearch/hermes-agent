import { WebLinksAddon } from '@xterm/addon-web-links'
import type { ILinkHandler } from '@xterm/xterm'

import { openLink } from '@/lib/external-link'

import { isMacPlatform } from './selection'

// Both of xterm's link paths — the web-links addon (URLs it finds in the
// buffer) and the core OSC 8 provider (hyperlinks a CLI emits explicitly) —
// activate through `window.open()`, which the window's setWindowOpenHandler
// denies: a click did nothing but log "Opening link blocked as opener could not
// be cleared", and OSC 8 fronted that dead end with a raw confirm() dialog.
// Route both through the shared link router, the path every other link in the
// app takes.
//
// ⌘-click on macOS, Ctrl-click elsewhere — VS Code's integrated terminal,
// Terminal.app, and iTerm2 all agree. A bare click belongs to the selection, so
// a misclick on a URL can't launch a browser mid-sentence. ⌥ stays out of it:
// that's the force-selection drag over mouse-mode TUIs.
export function isTerminalLinkActivation(
  event: Pick<MouseEvent, 'ctrlKey' | 'metaKey'>,
  isMac = isMacPlatform()
): boolean {
  return isMac ? event.metaKey : event.ctrlKey
}

// The terminal spends the ⌘/Ctrl modifier on activation itself, so it has no
// second chord left for destination — ⇧ takes that job here. Chat is the
// inverse: bare click is the OS browser and ⌘ opens the in-app pane. Here it's
// ⌘ to open (in-app) and ⇧⌘ to escape to the OS, so a bare click still belongs
// to selection.
const activate = (event: MouseEvent, uri: string) => {
  if (isTerminalLinkActivation(event)) {
    openLink(uri, { native: event.shiftKey })
  }
}

export const terminalLinkHandler: ILinkHandler = { activate }

export const terminalWebLinksAddon = () => new WebLinksAddon(activate)

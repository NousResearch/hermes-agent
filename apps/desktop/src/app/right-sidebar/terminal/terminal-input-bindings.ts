import type { Terminal } from '@xterm/xterm'

import { writeClipboardText } from '@/lib/clipboard'
import { triggerHaptic } from '@/lib/haptics'
import { keybindHandled } from '@/lib/keybinds/handled-keys'

import { terminalClipboardIntent } from './clipboard'
import { isMacPlatform } from './selection'
import { registerTerminalContextMenu } from './terminal-context-menu'

// `onData` also carries xterm-generated replies to DSR/CPR/DA queries during
// shell startup. Treat only real key events (plus explicit paste/drop/inject
// paths) as user activity, or an untouched refresh re-saves boot rows.
export function bindTerminalActivity(term: Terminal, host: HTMLElement, markActivity: () => void): () => void {
  const keyDisposable = term.onKey(markActivity)

  const onPointerActivity = (event: PointerEvent) => {
    if (event.button === 1 || term.modes.mouseTrackingMode !== 'none') {
      markActivity()
    }
  }

  const onWheelActivity = () => {
    if (term.modes.mouseTrackingMode !== 'none') {
      markActivity()
    }
  }

  host.addEventListener('beforeinput', markActivity)
  host.addEventListener('compositionstart', markActivity)
  host.addEventListener('pointerdown', onPointerActivity)
  host.addEventListener('wheel', onWheelActivity)

  return () => {
    keyDisposable.dispose()
    host.removeEventListener('beforeinput', markActivity)
    host.removeEventListener('compositionstart', markActivity)
    host.removeEventListener('pointerdown', onPointerActivity)
    host.removeEventListener('wheel', onWheelActivity)
  }
}

// DOM events xterm turns into typed data on its textarea. Pastes are left out on
// purpose: until restored output re-enables bracketed paste, a multi-line paste
// would run line by line in the shell.
const USER_INPUT_EVENT_TYPES = ['keydown', 'keypress', 'input'] as const

/** `onData` carries both what the user types and the replies xterm generates while
 *  parsing a write (DSR/CPR/DA). Replies only come out of parsing, never while a
 *  user input event is dispatching on the terminal, so `isUserInput()` tells the
 *  two apart: true exactly while such an event is still being dispatched. */
export function trackTerminalUserInput(host: HTMLElement): { dispose: () => void; isUserInput: () => boolean } {
  let latest: Event | null = null

  const record = (event: Event) => {
    latest = event
  }

  for (const type of USER_INPUT_EVENT_TYPES) {
    host.addEventListener(type, record, true)
  }

  return {
    dispose: () => {
      for (const type of USER_INPUT_EVENT_TYPES) {
        host.removeEventListener(type, record, true)
      }

      latest = null
    },
    isUserInput: () => (latest?.eventPhase ?? Event.NONE) !== Event.NONE
  }
}

// Right-click menu, the focus-routed Ctrl/Cmd+R, and copy/paste chords. Paste
// and ^R count as session activity. Returns the context-menu registration teardown.
export function bindTerminalClipboard(
  term: Terminal,
  host: HTMLElement,
  markActivity: () => void,
  writeInput: (data: string) => void,
  hasSession: () => boolean
): () => void {
  // The app context menu resolves right-clicks on this host through the
  // registered handle: xterm's selection is not a DOM selection, so the
  // DOM resolver would see nothing here. The same handle answers the
  // focus-routed Ctrl/Cmd+R: main claimed the keystroke, so re-deliver the
  // ^R byte to the PTY ourselves (#96482).
  const unregisterContextMenu = registerTerminalContextMenu(host, {
    getSelection: () => term.getSelection(),
    paste: text => {
      markActivity()
      term.focus()
      term.paste(text)
    },
    reload: () => {
      markActivity()
      writeInput('\x12')
    },
    selectAll: () => term.selectAll(),
    // Main claimed the close-tab chord; preserve readline word erase while
    // the shell is alive, and let the caller close a disconnected tab.
    wordErase: () => {
      markActivity()

      if (!hasSession()) {
        return false
      }

      writeInput('\x17')

      return true
    }
  })

  // Copy/paste chords. Returning false stops xterm from also sending the key
  // to the PTY; every path that doesn't copy or paste returns true, so plain
  // Ctrl+C with no selection still interrupts the running process.
  term.attachCustomKeyEventHandler(event => {
    // The window keybind dispatcher runs in capture, before xterm. A plain Alt
    // chord it ran as a Hermes action (a browser tab's Alt+T, Alt+[) must not
    // also reach the shell as its Meta key (ESC t, ESC [). Ctrl chords stay
    // shared, as natively: Ctrl+D is still EOF and Ctrl+B a tmux prefix.
    if (keybindHandled(event) && event.altKey && !event.ctrlKey && !event.metaKey) {
      return false
    }

    const intent = terminalClipboardIntent(event, {
      hasSelection: Boolean(term.getSelection()),
      isMac: isMacPlatform()
    })

    if (!intent) {
      return true
    }

    event.preventDefault()

    if (intent === 'copy') {
      const text = term.getSelection()
      // Write through the main process: the renderer's clipboard API throws
      // "Write permission denied" whenever the document isn't focused.
      void writeClipboardText(text).catch(() => {
        // Clipboard unavailable — the selection stays put so the user can retry.
      })
      term.clearSelection()
      triggerHaptic('selection')

      return false
    }
    void (async () => {
      const text = (await window.hermesDesktop?.readClipboard?.()) ?? ''

      if (text) {
        markActivity()
        term.paste(text)
      }
    })()

    return false
  })

  return unregisterContextMenu
}

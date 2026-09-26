import type { Terminal } from '@xterm/xterm'

import { writeClipboardText } from '@/components/ui/copy-button'
import { triggerHaptic } from '@/lib/haptics'

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

// Right-click menu and copy/paste chords. Paste counts as session activity.
// Returns the context-menu registration teardown.
export function bindTerminalClipboard(term: Terminal, host: HTMLElement, markActivity: () => void): () => void {
  // The app context menu resolves right-clicks on this host through the
  // registered handle: xterm's selection is not a DOM selection, so the
  // DOM resolver would see nothing here.
  const unregisterContextMenu = registerTerminalContextMenu(host, {
    getSelection: () => term.getSelection(),
    paste: text => {
      markActivity()
      term.focus()
      term.paste(text)
    },
    selectAll: () => term.selectAll()
  })

  // Copy/paste chords. Returning false stops xterm from also sending the key
  // to the PTY; every path that doesn't copy or paste returns true, so plain
  // Ctrl+C with no selection still interrupts the running process.
  term.attachCustomKeyEventHandler(event => {
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

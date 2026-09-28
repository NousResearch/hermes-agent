import { paletteToggle } from '@/app/command-palette/contrib'
import { isPaneVisible, togglePaneVisible } from '@/components/pane-shell/tree/store'
import { Terminal } from '@/lib/icons'
import { isFocusWithin } from '@/lib/keybinds/combo'

const TERMINAL_FOCUS_SCOPE = '[data-terminal]'

// Inactive tabs carry `invisible`. The on-screen instance does not, so this
// never targets a keep-alive tab that is not the keyboard surface. xterm's
// helper textarea is its real keyboard input — the same element the clipboard
// helper writes a selection into.
const FOCUS_TARGET = `${TERMINAL_FOCUS_SCOPE}:not(.invisible) .xterm-helper-textarea`

// The pane layout settles over a couple of frames (slot rect chase, theme
// repaint), and the composer focus bus can steal the keyboard on the macrotask
// between them. Three frames covers that settle plus one re-assert, then stops
// so a later deliberate click in the composer is not a focus fight.
const REVEAL_FOCUS_ATTEMPTS = 3

/**
 * Take keyboard focus for the active terminal once its pane has been revealed.
 *
 * Terminals stay mounted while the pane is hidden, so the activation effect
 * that calls `term.focus()` does not re-run on reveal. Every user reveal
 * (Ctrl+`, the palette row, the statusbar pill) has to claim the keyboard
 * itself and take it back if the composer wins the next frame.
 */
function focusRevealedTerminal(): void {
  scheduleFocusAttempt(REVEAL_FOCUS_ATTEMPTS)
}

function scheduleFocusAttempt(attempts: number): void {
  window.requestAnimationFrame(() => {
    // A hide that lands before the settle window ends must not pull the
    // keyboard back onto a pane the user just closed.
    if (!isPaneVisible('terminal') || isFocusWithin(TERMINAL_FOCUS_SCOPE)) {
      return
    }

    document.querySelector<HTMLTextAreaElement>(FOCUS_TARGET)?.focus()

    if (attempts > 1) {
      scheduleFocusAttempt(attempts - 1)
    }
  })
}

/**
 * User toggle of the terminal pane. The reveal direction claims keyboard
 * focus; the hide direction does not.
 */
export function toggleTerminalPane(): void {
  const revealed = !isPaneVisible('terminal')

  togglePaneVisible('terminal')

  if (revealed) {
    focusRevealedTerminal()
  }
}

/**
 * ⌘K door onto the same pane the keybind and statusbar pill flip — was a
 * one-way "open" row under Go to, so it never showed on/off and couldn't hide.
 * Reads the TREE like every other pane toggle: `$terminalTakeover` stays true
 * behind a stacked sibling tab or a minimized zone, which would light the row
 * "on" for a terminal that isn't on screen.
 *
 * Lives here, not inline in the contrib controller, so the reveal-focus test
 * drives the real row without importing the whole app graph.
 */
export const terminalPaletteToggle = paletteToggle({
  id: 'view.showTerminal',
  label: 'Toggle terminal',
  action: 'view.showTerminal',
  icon: Terminal,
  keywords: ['terminal', 'shell', 'console', 'pty'],
  get: () => isPaneVisible('terminal'),
  set: () => toggleTerminalPane()
})

import { isFocusWithin } from '@/lib/keybinds/combo'

const TERMINAL_FOCUS_SCOPE = '[data-terminal]'

// The on-screen instance's container carries `visible`; keep-alive tabs sit on
// `invisible`, so this never targets an off-screen terminal. xterm's hidden
// textarea is its real keyboard input — the same query the clipboard helper
// uses to reach it.
const FOCUS_TARGET = `${TERMINAL_FOCUS_SCOPE}:not(.invisible) .xterm-helper-textarea`

/** Frames to keep trying after a reveal. The pane layout settles over a couple
 *  of frames (slot rect chase, theme repaint), so a focus() issued sooner can
 *  land before the host is focusable; three frames covers that without turning
 *  into a focus fight with surfaces the user deliberately focuses later. */
const REVEAL_FOCUS_ATTEMPTS = 3

/**
 * Take keyboard focus for the active terminal once its pane has been revealed.
 *
 * Revealing the pane (Ctrl+`) restores the layout, but nothing on that path
 * focuses xterm: the terminals stay mounted while hidden (VS Code-style
 * keep-alive), so the `[active, status]` focus effect never re-runs and focus
 * stays wherever the user came from — usually the composer. Ctrl+` is also the
 * terminal's only focus affordance, so the reveal has to claim focus itself:
 * try over the next few animation frames and stop as soon as focus sits inside
 * the terminal scope.
 */
export function focusRevealedTerminal(): void {
  scheduleFocusAttempt(REVEAL_FOCUS_ATTEMPTS)
}

function scheduleFocusAttempt(attempts: number): void {
  window.requestAnimationFrame(() => {
    if (isFocusWithin(TERMINAL_FOCUS_SCOPE)) {
      return
    }

    document.querySelector<HTMLTextAreaElement>(FOCUS_TARGET)?.focus()

    if (attempts > 1) {
      scheduleFocusAttempt(attempts - 1)
    }
  })
}

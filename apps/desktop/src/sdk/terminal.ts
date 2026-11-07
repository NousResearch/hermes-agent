import type { ReadableAtom } from 'nanostores'

import { $terminalTakeover, setTerminalTakeover } from '@/app/right-sidebar/store'
import {
  activeGatewayHasShell,
  openSessionInTerminalPane,
  type OpenSessionTuiOptions,
  type OpenSessionTuiResult
} from '@/app/right-sidebar/terminal/open-session-tui'

export type { OpenSessionTuiOptions, OpenSessionTuiResult }

/** THE terminal-pane door for plugins. Deliberately narrow: a plugin cannot
 *  type arbitrary commands into the user's shell (`ctx.runAction` refuses the
 *  terminal actions for the same reason). What it can do is the one thing the
 *  app already does for the user — resume a Hermes session in the TUI — but in
 *  the in-app pane instead of an external emulator, plus show/collapse that
 *  pane. Enough for a "chat ⇄ TUI" view toggle; nothing more. */
export const terminalHost = {
  /** Resume a stored session in `hermes --tui` inside a fresh terminal tab on
   *  the active gateway, and take the terminal over. Result-shaped, never
   *  throws: `no-shell` means the active gateway is URL/token or cloud (its
   *  sessions live on a host this window has no shell on). */
  openSession: (storedSessionId: string, opts?: OpenSessionTuiOptions): OpenSessionTuiResult =>
    openSessionInTerminalPane(storedSessionId, opts),

  /** Whether `openSession` can work on the active gateway right now. */
  canOpenSession: (): boolean => activeGatewayHasShell(),

  /** Expand (`true`) or collapse to its rail (`false`) the terminal pane. The
   *  PTYs stay alive either way — a collapsed TUI keeps running. */
  setTakeover: (active: boolean): void => setTerminalTakeover(Boolean(active)),

  /** Live takeover state, for a chip that mirrors the pane: `.get()` in
   *  handlers, `useValue(host.terminal.takeover)` in components. */
  takeover: $terminalTakeover as ReadableAtom<boolean>
}

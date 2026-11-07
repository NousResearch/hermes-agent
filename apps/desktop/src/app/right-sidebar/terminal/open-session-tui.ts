// Resume a session in the Hermes TUI INSIDE the terminal pane — the in-app
// twin of "Open in terminal" (`electron/external-terminal.ts`), which hands the
// session to the user's own emulator. This verb is for switching VIEWS without
// leaving the app: the chat stays where it is, a fresh terminal tab runs
// `hermes --tui --resume <id>` on the gateway the session lives on, and the
// terminal takes over until the user drops it again (⌃` / the plugin's toggle).
//
// Only This device and SSH gateways carry a shell (`terminal-gateway.ts`); a
// URL/token or cloud gateway would run the TUI HERE against a session that
// lives THERE, so the verb refuses instead of resuming a session that does not
// exist on this machine. `hermes` is resolved by the interactive shell's own
// PATH: the bundled Desktop payload links it into ~/.local/bin, and an SSH
// gateway already requires Hermes on the remote host.

import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import { $activeConnectionId, $connectionsRegistry } from '@/store/connections'
import { $currentCwd } from '@/store/session'

import { posixQuote, tuiResumeArgs } from '../../../../electron/external-terminal'
import { backendProfileArg } from '../../../../electron/profile-id-guard'
import { $terminalInjection, setTerminalTakeover } from '../store'

import { defaultTerminalGateway } from './terminal-gateway'
import { createTerminal, renameTerminal } from './terminals'

export interface OpenSessionTuiOptions {
  /** Owner profile of the session (`host.state.focusedSessionProfile`). A
   *  non-slug value is dropped rather than pinned (#88842). */
  profile?: null | string
  /** Working directory for the shell; defaults to the current session cwd. */
  cwd?: null | string
  /** Tab label; defaults to `TUI`. */
  title?: null | string
}

export type OpenSessionTuiResult =
  | { ok: true; terminalId: string; command: string }
  | {
      ok: false
      /** `invalid-session`: no usable stored session id. `no-shell`: the active
       *  gateway has no shell transport (URL/token, cloud), so the session is
       *  not resumable from a terminal this window can open. */
      reason: 'invalid-session' | 'no-shell'
      error: string
    }

const SESSION_ID_RE = /^[A-Za-z0-9._:-]{1,128}$/

/** The shell line that resumes `sessionId` in the TUI, argv-quoted for /bin/sh.
 *  `profile` is pinned only when it is a legal slug — same guard as the
 *  external-terminal launcher. */
export function tuiResumeCommand(sessionId: string, profile?: null | string): string {
  const pinned = backendProfileArg(profile ?? undefined) ?? undefined

  return ['hermes', ...tuiResumeArgs(sessionId, pinned)].map(posixQuote).join(' ')
}

/** True when the gateway picked in Sessions runs shells on the host that owns
 *  its sessions: This device, or an SSH gateway (shell rides the tunnel). A
 *  window with no registry (the Webapp) has one local host and says yes. */
export function activeGatewayHasShell(): boolean {
  const registry = $connectionsRegistry.get()

  if (!registry) {
    return true
  }

  const id = $activeConnectionId.get()
  const source = id ? registry.connections.find(connection => connection.id === id) : undefined

  return !source || source.id === LOCAL_CONNECTION_ID || source.kind === 'ssh'
}

/** Open a terminal tab on the active gateway, take the terminal over, and
 *  resume `storedSessionId` in the TUI there. Never throws. */
export function openSessionInTerminalPane(storedSessionId: string, opts: OpenSessionTuiOptions = {}): OpenSessionTuiResult {
  const sessionId = typeof storedSessionId === 'string' ? storedSessionId.trim() : ''

  if (!SESSION_ID_RE.test(sessionId)) {
    return { ok: false, reason: 'invalid-session', error: 'A stored session id is required.' }
  }

  if (!activeGatewayHasShell()) {
    return {
      ok: false,
      reason: 'no-shell',
      error: 'The active gateway has no shell: add it as an SSH gateway to resume its sessions in the TUI.'
    }
  }

  const command = tuiResumeCommand(sessionId, opts.profile)
  const cwd = (opts.cwd ?? '').trim() || $currentCwd.get()
  const terminalId = createTerminal(cwd, defaultTerminalGateway())

  renameTerminal(terminalId, (opts.title ?? '').trim() || 'TUI')
  setTerminalTakeover(true)
  // The new tab is active; the injection flushes into it once its PTY is open
  // (`use-terminal-session.ts`), so ordering here does not race the spawn.
  $terminalInjection.set(command)

  return { ok: true, terminalId, command }
}

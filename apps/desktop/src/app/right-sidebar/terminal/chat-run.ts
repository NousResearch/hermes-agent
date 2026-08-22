import { $currentCwd } from '@/store/session'

import { $chatTerminalRunRequest, setTerminalTakeover, takeChatTerminalRunRequest } from '../store'

import { createTerminal } from './terminals'

// Eligibility guard, not a shell-safety classifier. Clicking Run is explicit
// user authorization for the visible command; reject bytes that can materially
// differ from what the terminal visibly presents. Newline and tab stay allowed.
const DEFAULT_IGNORABLE_CODE_POINT_RE = /\p{Default_Ignorable_Code_Point}/u

function hasUnsafeTerminalDisplayChars(command: string): boolean {
  if (DEFAULT_IGNORABLE_CODE_POINT_RE.test(command)) {
    return true
  }

  for (const char of command) {
    const code = char.codePointAt(0)

    if (
      code !== undefined &&
      ((code >= 0x00 && code <= 0x08) ||
        (code >= 0x0b && code <= 0x1f) ||
        (code >= 0x7f && code <= 0x9f) ||
        code === 0x2028 ||
        code === 0x2029)
    ) {
      return true
    }
  }

  return false
}

// 32 Ki UTF-16 code units is a deliberate one-click injection guardrail, not a
// PTY/shell line-buffer limit. Oversized fences remain available through Copy.
export const MAX_CHAT_RUN_CHARS = 32_768

export function isRunnableChatTerminalCommandText(command: string): boolean {
  return Boolean(command.trim()) && command.length <= MAX_CHAT_RUN_CHARS && !hasUnsafeTerminalDisplayChars(command)
}

export function hasEmbeddedTerminalBridge(): boolean {
  const terminal = typeof window === 'undefined' ? undefined : window.hermesDesktop?.terminal

  return typeof terminal?.start === 'function' && typeof terminal.write === 'function'
}

/** Deliver one exact-terminal request. Authorization is consumed before the PTY side effect. */
export function deliverChatTerminalRunRequest(
  terminalId: string,
  ptySessionId: string,
  write: (ptySessionId: string, data: string) => Promise<boolean>
): boolean {
  const command = takeChatTerminalRunRequest(terminalId)

  if (!command) {
    return false
  }

  void write(ptySessionId, `${command}\r`)

  return true
}

/** Queue a user-approved command into a brand-new user shell, never an existing
 * SSH/REPL/TUI/agent tab. Only one not-yet-flushed chat command may exist. */
export function queueChatCommandInFreshTerminal(command: string): string | null {
  if (!hasEmbeddedTerminalBridge() || !isRunnableChatTerminalCommandText(command) || $chatTerminalRunRequest.get()) {
    return null
  }

  const terminalId = createTerminal($currentCwd.get())
  setTerminalTakeover(true)
  $chatTerminalRunRequest.set({ command, terminalId })

  return terminalId
}

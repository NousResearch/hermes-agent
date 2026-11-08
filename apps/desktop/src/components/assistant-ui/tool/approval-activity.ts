import type { ToolCallLike } from './run-summary'

/** These calls render as activity, not file diffs or other deliverable cards. */
export const isApprovalActivity = (tool: ToolCallLike): boolean =>
  tool.toolName === 'terminal' || tool.toolName === 'execute_code'

export function isCurrentTurnMessage(messages: readonly { id: string; role: string }[], messageId: string): boolean {
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]

    if (message.id === messageId) {
      return message.role === 'assistant'
    }

    if (message.role === 'user') {
      return false
    }
  }

  return false
}

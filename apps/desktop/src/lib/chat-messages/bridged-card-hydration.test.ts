import { describe, expect, it } from 'vitest'

import { projectPendingConnection } from '@/app/session/hooks/use-session-actions/restore-pending-connection'
import type { ConnectionRequest } from '@/store/connection-request'
import type { SessionMessage } from '@/types/hermes'

import { toChatMessages } from './hydration'
import type { ChatMessage } from './types'

const install = { action: 'install', connectors: [{ mcp: true, name: 'linear' }] }

// A deferred tool's row as the agent persists it: the bridge name with one entry, stored before the tool runs.
function bridgedRow(name: string, args: Record<string, unknown>): SessionMessage {
  return {
    content: '',
    role: 'assistant',
    timestamp: 2,
    tool_calls: [
      { function: { arguments: JSON.stringify({ calls: [{ arguments: args, name }] }), name: 'tool_call' }, id: 'call_1', type: 'function' }
    ]
  } as SessionMessage
}

function partsWithId(messages: ChatMessage[]) {
  return messages.flatMap(message =>
    message.parts.filter(part => part.type === 'tool-call' && part.toolCallId === 'call_1')
  ) as Extract<ChatMessage['parts'][number], { type: 'tool-call' }>[]
}

const user = { content: 'Install the Linear MCP server', role: 'user', timestamp: 1 } as SessionMessage

describe('a card tool reached through the tool_call bridge', () => {
  it('hydrates under its own name, as the live tool.start part is named', () => {
    const [part] = partsWithId(toChatMessages([user, bridgedRow('manage_connections', install)]))

    expect(part.toolName).toBe('manage_connections')
    expect(part.args).toMatchObject(install)
  })

  it('is re-armed in place when a resume restores its open card', () => {
    const request: ConnectionRequest = {
      deadlineAt: 9e9,
      display: null,
      opId: 'op_1',
      receivedAt: 1,
      seq: 1,
      sessionId: 's1',
      settled: false,
      settledBy: null,
      targets: [{ action: 'install', kind: 'mcp', name: 'linear' } as ConnectionRequest['targets'][number]],
      toolCallId: 'call_1'
    }

    const projected = projectPendingConnection(toChatMessages([user, bridgedRow('manage_connections', install)]), request)

    expect(partsWithId(projected!.messages).map(part => part.toolName)).toEqual(['manage_connections'])
  })

  it('leaves a bridged activity tool as the bridge', () => {
    const [part] = partsWithId(toChatMessages([user, bridgedRow('session_search', { query: 'linear' })]))

    expect(part.toolName).toBe('tool_call')
  })
})

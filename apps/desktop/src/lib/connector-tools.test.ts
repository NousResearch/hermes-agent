import type { ToolCallMessagePart } from '@assistant-ui/react'
import { expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { latestConnectorPart } from './connector-tools'

it('keeps the live connector part through untargeted catalog checks', () => {
  const live = {
    type: 'tool-call' as const,
    toolCallId: 'wait',
    toolName: 'manage_connections',
    args: { action: 'wait', connectors: ['gmail'] }
  }
  const message: ChatMessage = { id: 'assistant', role: 'assistant', parts: [live] }

  const catalogInputs: ToolCallMessagePart['args'][] = [{}, { action: 'status' }, { action: 'status', connectors: [] }]

  for (const args of catalogInputs) {
    message.parts.push({ ...live, toolCallId: 'catalog', args })
    expect(latestConnectorPart([message])).toBe(live)
  }

  const targeted = { ...live, toolCallId: 'targeted', args: { connectors: ['notion'] } }
  message.parts.push(targeted)
  expect(latestConnectorPart([message])).toBe(targeted)

  const call = {
    ...live, toolCallId: 'app-tool', toolName: 'tool_call',
    args: { calls: [{ name: 'connectors__gmail__list', arguments: {} }] }
  }
  message.parts.push(call)
  expect(latestConnectorPart([message])).toBe(call)
})

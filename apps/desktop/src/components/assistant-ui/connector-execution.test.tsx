import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { ConnectorExecution } from './connector-tool'

vi.mock('@/components/assistant-ui/tool/fallback', () => ({
  ToolFallback: ({ toolName, result }: ToolCallMessagePartProps) => (
    <div data-testid="tool-result">{JSON.stringify({ toolName, result })}</div>
  )
}))
afterEach(cleanup)

it('keeps each connector result paired with its app', () => {
  const props: ToolCallMessagePartProps = {
    type: 'tool-call',
    argsText: '',
    status: { type: 'complete' },
    addResult: vi.fn(),
    resume: vi.fn(),
    respondToApproval: vi.fn(),
    toolName: 'tool_call',
    toolCallId: 'batch',
    args: {
      calls: [
        { name: 'connectors__gmail__LIST_MESSAGES', arguments: {} },
        { name: 'connectors__slack__SEARCH', arguments: {} }
      ]
    },
    result: { results: [{ response: 'mail result' }, { response: 'slack result' }] }
  }

  render(<ConnectorExecution {...props} />)
  const rows = screen.getAllByTestId('tool-result').map(row => JSON.parse(row.textContent ?? ''))

  expect(rows).toEqual([
    { toolName: 'Gmail: list messages', result: { response: 'mail result' } },
    { toolName: 'Slack: search', result: { response: 'slack result' } }
  ])
})

it('preserves the full disclosure for mixed remote batches', () => {
  const result = { results: [{ response: 'remote result' }, { response: 'mail result' }] }

  const props: ToolCallMessagePartProps = {
    type: 'tool-call',
    argsText: '',
    status: { type: 'complete' },
    addResult: vi.fn(),
    resume: vi.fn(),
    respondToApproval: vi.fn(),
    toolName: 'tool_call',
    toolCallId: 'mixed',
    args: {
      calls: [
        { name: 'connectors__gmail__LIST_MESSAGES', arguments: {} },
        { name: 'other_remote_tool', arguments: {} }
      ]
    },
    result
  }

  render(<ConnectorExecution {...props} />)
  expect(JSON.parse(screen.getByTestId('tool-result').textContent ?? '')).toEqual({ toolName: 'tool_call', result })
})

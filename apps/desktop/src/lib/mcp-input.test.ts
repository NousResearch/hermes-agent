import { describe, expect, it } from 'vitest'

import { readMcpInputValue } from './mcp-input'

describe('readMcpInputValue', () => {
  it('captures the value before a deferred state updater runs', () => {
    const input = { value: 'https://n8n.example.test' }
    const event: { currentTarget: { value: string } | null } = { currentTarget: input }
    const value = readMcpInputValue(event)

    event.currentTarget = null

    expect(value).toBe('https://n8n.example.test')
  })

  it('does not crash when the event target has already been cleared', () => {
    expect(readMcpInputValue({ currentTarget: null })).toBe('')
  })
})

import { describe, expect, it } from 'vitest'

import { buildToolView, compactPreview, looksRedundant, toolCopyPayload, type ToolPart } from './fallback-model'

describe('large tool output previews', () => {
  it('keeps the complete output available for expansion and Copy while shortening only its preview', () => {
    const output = `  first line\n${'output line\n'.repeat(10_000)}  `

    const part: ToolPart = {
      type: 'tool-call',
      toolCallId: 'large-output',
      toolName: 'terminal',
      args: { command: 'inspect' },
      result: { stdout: output, exit_code: 0 }
    }

    const view = buildToolView(part, '')

    expect(view.subtitle).toBe('first line')
    expect(view.detail).toBe(output.trim())
    expect(toolCopyPayload(part, view).text).toBe(output.trim())
    expect(compactPreview(output, 72)).toBe(`${output.replace(/\s+/g, ' ').trim().slice(0, 71)}…`)
  })

  it('does not hide output when a matching title is followed by more content, even after a long whitespace run', () => {
    const whitespace = ' \n\t'.repeat(10_000)

    expect(looksRedundant('Result is ready', ` RESULT${whitespace}IS READY `)).toBe(true)
    expect(looksRedundant('Result is ready', ` RESULT${whitespace}IS READY${whitespace}details`)).toBe(false)
    expect(looksRedundant('', whitespace)).toBe(true)
    expect(looksRedundant('İΣ', ' İΣ ')).toBe(true)
    expect(looksRedundant('İΣ', ' İΣ extra ')).toBe(false)
  })
})

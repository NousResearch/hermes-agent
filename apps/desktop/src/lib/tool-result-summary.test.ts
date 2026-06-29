import { describe, expect, it } from 'vitest'

import { extractToolErrorMessage, formatToolResultSummary } from './tool-result-summary'

describe('formatToolResultSummary', () => {
  it('unwraps wrapper payloads into structured key-value lines', () => {
    const summary = formatToolResultSummary({
      success: true,
      result: {
        data: {
          path: '/tmp/demo.txt',
          status: 'ok',
          lines_written: 12,
          checksum: 'abc123'
        }
      }
    })

    expect(summary).toContain('- Path: /tmp/demo.txt')
    expect(summary).toContain('- Status: ok')
    expect(summary).toContain('- Lines Written: 12')
    expect(summary).not.toContain('"path"')
  })

  it('summarizes object arrays as readable list items', () => {
    const summary = formatToolResultSummary([
      { title: 'First result', snippet: 'alpha preview text' },
      { title: 'Second result', status: 'cached' },
      { title: 'Third result', summary: 'more details' },
      { title: 'Fourth result', summary: 'line 4' },
      { title: 'Fifth result', summary: 'line 5' },
      { title: 'Sixth result', summary: 'line 6' },
      { title: 'Seventh result', summary: 'line 7' }
    ])

    expect(summary).toContain('- First result - alpha preview text')
    expect(summary).toContain('- Second result (cached)')
    expect(summary).toContain('- … 1 more item')
  })

  it('truncates long field values for compact display', () => {
    const summary = formatToolResultSummary({
      message: 'ok',
      details: `prefix ${'x'.repeat(500)}`
    })

    const detailsLine = summary.split('\n').find(line => line.startsWith('- Details:'))

    expect(detailsLine).toBeTruthy()
    expect(detailsLine?.length).toBeLessThan(230)
    expect(detailsLine).toContain('…')
  })

  it('formats stringified json payloads without raw dumps', () => {
    const summary = formatToolResultSummary(
      JSON.stringify({
        data: {
          title: 'Build report',
          completed: true
        }
      })
    )

    expect(summary).toContain('- Title: Build report')
    expect(summary).toContain('- Completed: true')
  })
})

describe('extractToolErrorMessage', () => {
  it('finds nested error messages through wrappers', () => {
    const error = extractToolErrorMessage({
      success: false,
      result: {
        output: {
          error: {
            message: 'Permission denied writing /tmp/demo.txt'
          }
        }
      }
    })

    expect(error).toBe('Permission denied writing /tmp/demo.txt')
  })

  it('does not treat successful payload messages as errors', () => {
    const error = extractToolErrorMessage({
      success: true,
      message: 'Completed successfully',
      data: { count: 3 }
    })

    expect(error).toBe('')
  })

  it('ignores placeholder error fields in successful payloads', () => {
    const error = extractToolErrorMessage({
      success: true,
      data: {
        error: 'none',
        status: 'ok'
      }
    })

    expect(error).toBe('')
  })

  it('does not count empty fields as hidden "more fields"', () => {
    // `tags: []` and `note: ''` render nothing (formatFieldValue returns
    // empty for empty arrays / empty strings), so they are not fields the
    // user is "missing" — only the cap-truncated remainder should be
    // reported. Before the fix this emitted "- … 2 more fields".
    const summary = formatToolResultSummary({ name: 'x', tags: [], note: '' })

    expect(summary).toBe('- Name: x')
    expect(summary).not.toContain('more fields')
  })

  it('still reports genuinely cap-truncated fields as "N more fields"', () => {
    // 10 content-bearing keys with an 8-field cap → 2 truly hidden fields.
    // This guards against over-correcting the empty-field fix into dropping
    // the real cap-overflow tail.
    const record: Record<string, string> = {}

    for (let i = 1; i <= 10; i += 1) {
      record[`field${i}`] = `value${i}`
    }

    const summary = formatToolResultSummary(record)

    expect(summary).toContain('… 2 more fields')
  })
})

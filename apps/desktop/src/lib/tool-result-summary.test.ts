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

  it('summarizes lifecycle review packets with closure signal provenance', () => {
    const summary = formatToolResultSummary({
      packet_type: 'lifecycle_review.packet',
      candidates: [
        {
          title: 'Old launch situation',
          target_ref: 'situations/2026-04-old-launch',
          recommended_action: 'archive',
          signals: {
            closure_signal: {
              source: 'inbox',
              polarity: 'complete',
              confidence: 'high'
            }
          }
        }
      ]
    })

    expect(summary).toContain('Lifecycle Review')
    expect(summary).toContain('Candidates: 1')
    expect(summary).toContain('Old launch situation · action archive · situations/2026-04-old-launch')
    expect(summary).toContain('Signal: inbox · complete · high')
  })

  it('summarizes lifecycle proposal draft packets', () => {
    const summary = formatToolResultSummary({
      packet_type: 'lifecycle_proposal_draft.packet',
      proposal_count: 1,
      proposals: [
        {
          target_ref: 'situations/2026-04-old-launch',
          recommended_action: 'archive',
          summary: 'Draft an archive proposal for reviewer approval.'
        }
      ]
    })

    expect(summary).toContain('Lifecycle Proposal Draft')
    expect(summary).toContain('Proposals: 1')
    expect(summary).toContain('situations/2026-04-old-launch · archive · Draft an archive proposal')
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
})

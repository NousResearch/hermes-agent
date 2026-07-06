import { describe, expect, it } from 'vitest'

import { activeTimelineIndex, deriveTimelineEntries, stripCronPromptHint, timelinePreview } from './timeline-data'

// Mirrors cron/scheduler.py cron_hint — a single bracketed paragraph with
// interior [SILENT] mentions, followed by a blank line and the real prompt.
const CRON_HINT =
  '[IMPORTANT: You are running as a scheduled cron job. ' +
  'DELIVERY: Your final response will be automatically delivered to the user — do NOT use send_message ' +
  'or try to deliver the output yourself. Just produce your report/output as your final response and the ' +
  'system handles the rest. SILENT: If there is genuinely nothing new to report, respond with exactly ' +
  '"[SILENT]" (nothing else) to suppress delivery. Never combine [SILENT] with content — either report ' +
  'your findings normally, or say [SILENT] and nothing more.]\n\n'

describe('timelinePreview', () => {
  it('collapses whitespace to a single line', () => {
    expect(timelinePreview('hello\n\n  world\tagain')).toBe('hello world again')
  })

  it('truncates with an ellipsis past the limit', () => {
    const out = timelinePreview('abcdefghij', 5)
    expect(out).toBe('abcd…')
    expect(out.length).toBe(5)
  })
})

describe('deriveTimelineEntries', () => {
  it('keeps non-empty user prompts in order', () => {
    expect(
      deriveTimelineEntries([
        { id: 'u1', role: 'user', text: 'first' },
        { id: 'a1', role: 'assistant', text: 'answer' },
        { id: 'u2', role: 'user', text: '  second  ' }
      ])
    ).toEqual([
      { id: 'u1', preview: 'first' },
      { id: 'u2', preview: 'second' }
    ])
  })

  it('drops blanks and background-process notifications', () => {
    expect(
      deriveTimelineEntries([
        { id: 'u1', role: 'user', text: '   ' },
        { id: 'u2', role: 'user', text: '[IMPORTANT: Background process 123 finished]' },
        { id: 'u3', role: 'user', text: 'real prompt' }
      ]).map(e => e.id)
    ).toEqual(['u3'])
  })

  it('previews the real prompt of a cron message, not the injected hint', () => {
    expect(deriveTimelineEntries([{ id: 'u1', role: 'user', text: `${CRON_HINT}Morning briefing` }])).toEqual([
      { id: 'u1', preview: 'Morning briefing' }
    ])
  })
})

describe('stripCronPromptHint', () => {
  it('removes the scheduler hint and keeps the real prompt (#59537)', () => {
    expect(stripCronPromptHint(`${CRON_HINT}Summarize today's news`)).toBe("Summarize today's news")
  })

  it('handles the interior [SILENT] brackets without truncating mid-hint', () => {
    const stripped = stripCronPromptHint(`${CRON_HINT}real prompt`)
    expect(stripped).not.toContain('[SILENT]')
    expect(stripped).toBe('real prompt')
  })

  it('returns empty for a hint-only message', () => {
    expect(stripCronPromptHint(CRON_HINT.trim())).toBe('')
  })

  it('leaves ordinary prompts untouched', () => {
    expect(stripCronPromptHint('just a normal prompt')).toBe('just a normal prompt')
  })
})

describe('activeTimelineIndex', () => {
  it('returns the last prompt scrolled to or above the top edge', () => {
    expect(activeTimelineIndex([-400, -10, 320])).toBe(1)
  })

  it('falls back to the first rendered entry', () => {
    expect(activeTimelineIndex([null, 120, 480])).toBe(1)
    expect(activeTimelineIndex([null, null])).toBe(0)
  })
})

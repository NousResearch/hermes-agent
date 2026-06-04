import { describe, expect, it } from 'vitest'

import {
  answerHasCompleteCompanyBrainTrace,
  formatCompanyBrainTrace,
  missingCompanyBrainTraceFields,
  parseCompanyBrainTrace
} from './company-brain-trace'

describe('Company Brain traceability primitive', () => {
  it('formats the minimum source/conflict/correction footer for an important answer', () => {
    const markdown = formatCompanyBrainTrace({
      changedAfterCorrection: [{ after: 'Qigawa spelling is canonical', before: 'Kigawa spelling' }],
      conflictWinner: 'Obsidian canon note beats stale model memory',
      sources: ['Obsidian: BBBB/Canon.md', 'session correction from Dave']
    })

    expect(markdown).toContain('### Company Brain trace')
    expect(markdown).toContain('- Source: Obsidian: BBBB/Canon.md')
    expect(markdown).toContain('- Conflict winner: Obsidian canon note beats stale model memory')
    expect(markdown).toContain('- Changed after correction: Kigawa spelling → Qigawa spelling is canonical')
    expect(answerHasCompleteCompanyBrainTrace(markdown)).toBe(true)
  })

  it('parses trace footers and de-duplicates semicolon-separated sources', () => {
    const trace = parseCompanyBrainTrace(`Answer body.

### Company Brain trace
- Sources: memory:BBBB; memory:BBBB; Obsidian:Canon
- Conflict winner: Dave correction wins over stale memory
- Changed after correction: old → new
`)

    expect(trace).toEqual({
      changedAfterCorrection: [{ after: 'new', before: 'old' }],
      conflictWinner: 'Dave correction wins over stale memory',
      sources: ['memory:BBBB', 'Obsidian:Canon']
    })
  })

  it('reports missing trace fields so the UI/gateway can gate important answers later', () => {
    expect(missingCompanyBrainTraceFields(null)).toEqual(['source', 'conflictWinner', 'changedAfterCorrection'])
    expect(
      missingCompanyBrainTraceFields(
        parseCompanyBrainTrace(`### Company Brain trace
- Source: session_search:abc
`)
      )
    ).toEqual(['conflictWinner', 'changedAfterCorrection'])
    expect(answerHasCompleteCompanyBrainTrace('plain answer')).toBe(false)
  })
})

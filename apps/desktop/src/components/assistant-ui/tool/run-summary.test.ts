import { describe, expect, it } from 'vitest'

import { summarizeToolRun, type ToolCallLike } from './run-summary'

function tool(toolName: string, args: Record<string, unknown> = {}, result?: unknown): ToolCallLike {
  return { args, result, toolCallId: `${toolName}-${Math.random()}`, toolName }
}

const read = (path: string) => tool('read_file', { path }, { content: '' })
const searched = (query: string) => tool('search_files', { query }, { hits: [] })
const ran = (command: string) => tool('terminal', { command }, { exit_code: 0 })

const settled = (tools: ToolCallLike[]) => summarizeToolRun(tools, false)
const running = (tools: ToolCallLike[]) => summarizeToolRun(tools, true)

// A run only ever holds ephemeral activity: reads, searches, commands. File
// edits and other cards are split out before a run is summarized, so there is
// no "Edited …" clause to test here — that work shows as its own diff card.
describe('summarizeToolRun', () => {
  it('names a lone target and counts the rest', () => {
    expect(settled([searched('toolRuns'), read('a.ts'), read('b.ts'), read('c.ts')])).toBe('Explored 4 files')
  })

  it('orders clauses explore then run regardless of call order', () => {
    expect(settled([ran('ls'), read('a.ts'), read('b.ts'), ran('pwd'), ran('id')])).toBe(
      'Explored 2 files, ran 3 commands'
    )
  })

  it('counts commands rather than naming them once they have run', () => {
    expect(settled([ran('git status')])).toBe('Ran 1 command')
    expect(settled([read('status.ts'), ran('a'), ran('b'), ran('c'), ran('d'), ran('e')])).toBe(
      'Explored status.ts, ran 5 commands'
    )
  })

  it('puts the running category in the present tense and leaves the rest past', () => {
    expect(running([read('a.ts'), tool('read_file', { path: 'b.ts' }), ran('x'), ran('y')])).toBe(
      'Exploring 2 files, ran 2 commands'
    )
  })

  it('names the command that is still running', () => {
    expect(running([tool('terminal', { command: 'npm run typecheck' })])).toMatch(/^Running /)
  })

  // Sequential calls leave a gap where the run is still going but nothing is
  // pending. Falling back to past tense there contradicted the ticker still
  // scrolling underneath, so the most recent call carries the present tense.
  it('stays in the present tense between two sequential calls', () => {
    expect(running([read('a.ts'), ran('x'), ran('y')])).toBe('Explored a.ts, running 2 commands')
  })

  // A turn can end — or the agent can simply move on — with a call that never
  // got a result. The run is history at that point and has to read as history,
  // or it narrates work that stopped happening and never offers its toggle.
  it('reads a run the turn left unresolved as finished', () => {
    expect(settled([read('a.ts'), tool('search_files', { query: 'toolRuns' })])).toBe('Explored 2 files')
  })

  it('names skill_view calls instead of counting anonymous tools', () => {
    expect(settled([tool('skill_view', { name: 'example-skill' }, { content: '' })])).toContain('example-skill')
    expect(settled([tool('skill_view', { name: 'example-skill' }, { content: '' })])).not.toMatch(/Used \d+ tools/)
  })

  it('keeps skill names when a run also explored files', () => {
    const summary = settled([
      read('wiring.tsx'),
      tool('skill_view', { name: 'example-skill' }, { content: '' }),
      tool('skill_view', { file_path: 'references/example.md', name: 'example-skill' }, { content: '' })
    ])

    expect(summary).toContain('example-skill')
    expect(summary).not.toMatch(/used \d+ tools/i)
  })
})

describe('summaryArgIdentity', () => {
  it('changes when a skill name or resource path arrives', async () => {
    const { summaryArgIdentity } = await import('./run-summary')

    expect(summaryArgIdentity({})).not.toBe(summaryArgIdentity({ name: 'example-skill' }))
    expect(summaryArgIdentity({ name: 'example-skill' })).not.toBe(
      summaryArgIdentity({ file_path: 'references/example.md', name: 'example-skill' })
    )
  })

  it('ignores unrelated streaming payload fields', async () => {
    const { summaryArgIdentity } = await import('./run-summary')

    expect(summaryArgIdentity({ blob: 'x'.repeat(10_000), name: 'example-skill' })).toBe(
      summaryArgIdentity({ blob: 'y'.repeat(10_000), name: 'example-skill' })
    )
  })
})

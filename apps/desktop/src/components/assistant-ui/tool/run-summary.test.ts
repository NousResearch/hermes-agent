import { describe, expect, it } from 'vitest'

import { summarizeToolRun, type ToolCallLike } from './run-summary'

function tool(toolName: string, args: Record<string, unknown> = {}, result?: unknown): ToolCallLike {
  return { args, result, toolCallId: `${toolName}-${Math.random()}`, toolName }
}

const read = (path: string) => tool('read_file', { path }, { content: '' })
const searched = (query: string) => tool('search_files', { query }, { hits: [] })
const ran = (command: string) => tool('terminal', { command }, { exit_code: 0 })
const webSearched = (query: string) => tool('web_search', { query }, { success: true, data: { web: [] } })
const fetched = (url: string) => tool('web_extract', { urls: [url] }, { success: true, results: [] })
const browsed = () => tool('browser_exec', { code: 'goto' }, { success: true })
const analyzed = (image: string) => tool('vision_analyze', { image_url: image }, { success: true, analysis: '' })

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

  // Web and vision work is not file work: counting two searches as "Explored 2
  // files" misreports what the run did (#123085).
  it('counts web searches as queries rather than files', () => {
    expect(settled([webSearched('release notes'), webSearched('changelog')])).toBe('Searched 2 queries')
    expect(running([webSearched('release notes'), webSearched('changelog')])).toBe('Searching 2 queries')
  })

  it('names page reads, browsing and vision after what they were', () => {
    expect(settled([fetched('https://a.example'), browsed(), analyzed('shot.png')])).toBe(
      'Read 1 page, browsed 1 page, analyzed 1 image'
    )
  })
})

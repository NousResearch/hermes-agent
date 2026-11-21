import { describe, expect, it } from 'vitest'

import { summarizeToolRun, type ToolCallLike } from './run-summary'

function tool(toolName: string, args: Record<string, unknown> = {}, result?: unknown): ToolCallLike {
  return { args, result, toolCallId: `${toolName}-${Math.random()}`, toolName }
}

const read = (path: string) => tool('read_file', { path }, { content: '' })
const searched = (query: string) => tool('search_files', { query }, { hits: [] })
const ran = (command: string) => tool('terminal', { command }, { exit_code: 0 })
// Fixtures carry the REAL tool schemas (tools/web_tools.py): web_search takes
// `query`, web_extract takes `urls` (a list, up to five per call).
const webSearched = (query: string) => tool('web_search', { query }, { success: true })
const webExtracted = (urls: string[]) => tool('web_extract', { urls }, { success: true })
const navigated = (url: string) => tool('browser_navigate', { url }, { success: true })
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

  it('puts only pending calls in the present tense', () => {
    expect(running([read('a.ts'), tool('read_file', { path: 'b.ts' }), ran('x'), ran('y')])).toBe(
      'Explored a.ts, exploring b.ts, ran 2 commands'
    )
  })

  it('keeps completed commands out of the running count even without a result', () => {
    const completed = [
      ...Array.from({ length: 21 }, () => ran('pwd')),
      { ...tool('terminal', { command: 'pwd' }), completedAt: 123 }
    ]

    expect(running([...completed, tool('terminal', { command: 'npm test' })])).toBe('Ran 22 commands, running npm test')
  })

  it('counts pending batch pages separately and narrates every pending category', () => {
    expect(
      running([
        tool('read_file', { path: 'next.ts' }),
        webExtracted(['https://a.example', 'https://b.example']),
        tool('web_extract', { urls: ['https://c.example', 'https://d.example', 'https://e.example'] })
      ])
    ).toBe('Exploring next.ts, read 2 pages, reading 3 pages')
  })

  it('names the command that is still running', () => {
    expect(running([tool('terminal', { command: 'npm run typecheck' })])).toMatch(/^Running /)
  })

  // The run stays live between sequential calls, but its completed work must
  // read as history while the ticker waits for the next call.
  it('uses the past tense between two sequential calls', () => {
    expect(running([read('a.ts'), ran('x'), ran('y')])).toBe('Explored a.ts, ran 2 commands')
  })

  // A turn can end — or the agent can simply move on — with a call that never
  // got a result. The run is history at that point and has to read as history,
  // or it narrates work that stopped happening and never offers its toggle.
  it('reads a run the turn left unresolved as finished', () => {
    expect(settled([read('a.ts'), tool('search_files', { query: 'toolRuns' })])).toBe('Explored 2 files')
  })

  // The web tools act on queries and pages, not files. Counted in the explore
  // bucket they read as "Explored 2 files" while their own rows say Searched
  // (#123085).
  it('counts web searches as queries, not explored files', () => {
    expect(settled([webSearched('hermes agent'), webSearched('kv cache')])).toBe('Searched 2 queries')
    expect(running([webSearched('hermes agent'), webSearched('kv cache')])).toBe('Searched 2 queries')
  })

  it('names a lone web search the way its row does', () => {
    expect(settled([webSearched('hermes agent')])).toBe('Searched “hermes agent”')
  })

  it('names a lone web extract by hostname the way its row does', () => {
    expect(settled([webExtracted(['https://example.com/docs'])])).toBe('Read example.com/docs')
    expect(running([webExtracted(['https://example.com/docs'])])).toBe('Read example.com/docs')
    expect(running([tool('web_extract', { urls: ['https://example.com/docs'] })])).toBe('Reading example.com/docs')
  })

  it('still names a legacy string-url web extract shape', () => {
    expect(settled([tool('web_extract', { url: 'https://example.com/docs' }, { success: true })])).toBe(
      'Read example.com/docs'
    )
  })

  it('gives web searches their own clause after explored files', () => {
    expect(settled([read('a.ts'), webSearched('x'), webSearched('y')])).toBe('Explored a.ts, searched 2 queries')
  })

  // One web_extract call fetches up to five URLs, so the page count follows
  // the URLs rather than the calls.
  it('counts every page a batched fetch read', () => {
    expect(settled([webExtracted(['https://a.example', 'https://b.example', 'https://c.example'])])).toBe(
      'Read 3 pages'
    )
    expect(
      settled([webExtracted(['https://a.example', 'https://b.example']), webExtracted(['https://c.example'])])
    ).toBe('Read 3 pages')
  })

  // Clicks, screenshots and scripts load nothing; only navigation opens pages.
  it('never counts browser interaction as pages', () => {
    const interaction = [browsed(), browsed(), tool('browser_screenshot'), tool('browser_scroll')]

    expect(settled(interaction)).toBe('Performed 4 browser actions')
    expect(settled([navigated('https://a.example'), navigated('https://b.example'), ...interaction])).toBe(
      'Opened 2 pages, performed 4 browser actions'
    )
    expect(settled([navigated('https://a.example')])).toBe('Opened a.example')
  })

  it('counts vision analysis as images', () => {
    expect(settled([analyzed('shot.png')])).toBe('Analyzed 1 image')
    expect(settled([analyzed('shot.png'), analyzed('other.png')])).toBe('Analyzed 2 images')
  })
})

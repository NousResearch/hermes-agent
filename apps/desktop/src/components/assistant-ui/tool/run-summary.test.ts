import { describe, expect, it } from 'vitest'

import { summarizeToolRun, type ToolCallLike, toolPresentVerb } from './run-summary'

function tool(toolName: string, args: Record<string, unknown> = {}, result?: unknown): ToolCallLike {
  return { args, result, toolCallId: `${toolName}-${Math.random()}`, toolName }
}

const read = (path: string) => tool('read_file', { path }, { content: '' })
const searched = (pattern: string) => tool('search_files', { pattern, path: 'src' }, { hits: [] })
const ran = (command: string) => tool('terminal', { command }, { exit_code: 0 })
const webSearched = (query: string) => tool('web_search', { query }, { success: true, data: { web: [] } })
const fetched = (...urls: string[]) => tool('web_extract', { urls }, { success: true, results: [] })
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
    expect(settled([read('wiring.tsx')])).toBe('Read wiring.tsx')
    expect(settled([searched('toolRuns'), read('a.ts'), read('b.ts'), read('c.ts')])).toBe(
      'Read 3 files, searched for toolRuns'
    )
  })

  // Reading a file, grepping for a pattern and listing a directory are three
  // different acts; folding them into "Explored 4 files" hid which one ran.
  it('keeps reads, pattern searches and listings as separate clauses', () => {
    expect(settled([searched('a'), searched('b'), read('x.ts')])).toBe('Read x.ts, searched for 2 patterns')
    expect(settled([tool('list_files', { path: 'src' }), tool('list_files', { path: 'lib' })])).toBe(
      'Listed 2 directories'
    )
    expect(running([read('x.ts'), searched('a'), tool('search_files', { pattern: 'b' })])).toBe(
      'Read x.ts, searching for 2 patterns'
    )
  })

  it('orders clauses read then run regardless of call order', () => {
    expect(settled([ran('ls'), read('a.ts'), read('b.ts'), ran('pwd'), ran('id')])).toBe('Read 2 files, ran 3 commands')
  })

  it('counts commands rather than naming them once they have run', () => {
    expect(settled([ran('git status')])).toBe('Ran 1 command')
    expect(settled([read('status.ts'), ran('a'), ran('b'), ran('c'), ran('d'), ran('e')])).toBe(
      'Read status.ts, ran 5 commands'
    )
  })

  it('puts the running category in the present tense and leaves the rest past', () => {
    expect(running([read('a.ts'), tool('read_file', { path: 'b.ts' }), ran('x'), ran('y')])).toBe(
      'Reading 2 files, ran 2 commands'
    )
  })

  it('names the command that is still running', () => {
    expect(running([tool('terminal', { command: 'npm run typecheck' })])).toMatch(/^Running /)
  })

  // Sequential calls leave a gap where the run is still going but nothing is
  // pending. Falling back to past tense there contradicted the ticker still
  // scrolling underneath, so the most recent call carries the present tense.
  it('stays in the present tense between two sequential calls', () => {
    expect(running([read('a.ts'), ran('x'), ran('y')])).toBe('Read a.ts, running 2 commands')
  })

  // A turn can end — or the agent can simply move on — with a call that never
  // got a result. The run is history at that point and has to read as history,
  // or it narrates work that stopped happening and never offers its toggle.
  it('reads a run the turn left unresolved as finished', () => {
    expect(settled([read('a.ts'), tool('search_files', { pattern: 'toolRuns' })])).toBe(
      'Read a.ts, searched for toolRuns'
    )
  })

  // Web and vision work is not file work: counting two searches as "Explored 2
  // files" misreports what the run did (#123085).
  it('says a web search searched the web, counting repeats', () => {
    expect(settled([webSearched('release notes')])).toBe('Searched the web')
    expect(settled([webSearched('release notes'), webSearched('changelog')])).toBe('Searched the web twice')
    expect(running([webSearched('a'), webSearched('b'), webSearched('c')])).toBe('Searching the web 3 times')
  })

  it('names page reads, browser work and vision after what they were', () => {
    expect(settled([fetched('https://a.example'), browsed(), analyzed('shot.png')])).toBe(
      'Read https://a.example, performed 1 browser action, analyzed 1 image'
    )
  })

  // One web_extract call fetches up to five URLs, so the page count follows
  // the URLs rather than the calls.
  it('counts every page a batched fetch read', () => {
    expect(settled([fetched('https://a.example', 'https://b.example', 'https://c.example')])).toBe('Read 3 pages')
    expect(settled([fetched('https://a.example', 'https://b.example'), fetched('https://c.example')])).toBe(
      'Read 3 pages'
    )
  })

  // Clicks, screenshots and scripts load nothing; only navigation opens pages.
  it('never counts browser interaction as pages', () => {
    const interaction = [browsed(), browsed(), tool('browser_screenshot'), tool('browser_scroll')]

    expect(settled(interaction)).toBe('Performed 4 browser actions')
    expect(settled([navigated('https://a.example'), navigated('https://b.example'), ...interaction])).toBe(
      'Opened 2 pages, performed 4 browser actions'
    )
    expect(settled([navigated('https://a.example')])).toBe('Opened https://a.example')
  })

  // An MCP call is a call to some service; "Used 5 tools" hid which one.
  it('names MCP calls by the server they went to', () => {
    const github = (tool_: string) => tool(`mcp__github__${tool_}`, {}, { ok: true })
    const context7 = () => tool('mcp__context7__get_library_docs', {}, { ok: true })

    expect(settled([github('list_issues'), github('get_pr')])).toBe('Called GitHub twice')
    expect(settled([github('get_pr'), context7(), context7(), context7()])).toBe(
      'Called Context7 3 times, called GitHub'
    )
    expect(settled([tool('mcp__acme_docs__lookup')])).toBe('Called Acme Docs')
    expect(running([github('get_pr'), tool('mcp__github__list_issues')])).toBe('Calling GitHub twice')
  })

  // Memory is memory whichever surface it came through — the built-in tool,
  // a plugin, or an MCP server — so recalls and writes read the same way.
  it('reads memory tools as checking and saving memory', () => {
    const recalls = [
      tool('mnemosyne_recall', { query: 'x' }, { results: [] }),
      tool('mcp__mnemosyne__search', { query: 'y' }, { results: [] }),
      tool('mcp__memory__get_entities', {}, { entities: [] })
    ]

    const saves = [
      tool('memory', { action: 'add', content: 'a' }),
      tool('mnemosyne_remember', { content: 'b' }),
      tool('mcp__mnemosyne__store', { content: 'c' })
    ]

    expect(settled(recalls.slice(0, 1))).toBe('Checked memory')
    expect(settled(recalls)).toBe('Checked memory 3 times')
    expect(settled(saves)).toBe('Saved 3 memories')
    expect(settled([tool('memory', { action: 'add', content: 'a' })])).toBe('Saved 1 memory')
    expect(running([...recalls, saves[0]])).toBe('Checked memory 3 times, saving 1 memory')
  })

  it('names todos updated', () => {
    expect(settled([tool('todo', { todos: [] }), tool('todo', { todos: [] })])).toBe('Updated todos twice')
    expect(running([tool('todo')])).toBe('Updating todos')
  })

  // A server is memory by its own name, not because the word appears in it;
  // the control is an ordinary server read the same way.
  it('does not claim an MCP server as memory because its name mentions memory', () => {
    expect(settled([tool('mcp__github__list_issues')])).toBe('Called GitHub')
    expect(settled([tool('mcp__memory_server__list_databases')])).toBe('Called Memory Server')
    expect(settled([tool('mcp__basic_memory__create_entities')])).toBe('Called Basic Memory')
    expect(settled([tool('mcp__memory__create_entities', {})])).toBe('Saved 1 memory')
    expect(settled([tool('mcp__memory__read_graph', {})])).toBe('Checked memory')
  })

  // The drafting status line speaks the same words the run header will.
  it('drafts a call in the words its run will use', () => {
    expect(toolPresentVerb('mcp__github__get_pr')).toBe('Calling GitHub')
    expect(toolPresentVerb('web_search')).toBe('Searching the web')
    expect(toolPresentVerb('mnemosyne_recall')).toBe('Checking memory')
    expect(toolPresentVerb('read_file')).toBe('Reading')
    expect(toolPresentVerb('search_files')).toBe('Searching')
    // Card tools never reach a run summary, but the drafting line still names them.
    expect(toolPresentVerb('clarify')).toBe('Asking')
    expect(toolPresentVerb('delegate_task')).toBe('Launching')
  })
})

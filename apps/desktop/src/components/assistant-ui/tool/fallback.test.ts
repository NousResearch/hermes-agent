import { describe, expect, it } from 'vitest'

import { isCardTool, isSilentTool } from '@/lib/tool-render-class'

import { splitRunItems, technicalTrace } from './fallback'

describe('isCardTool', () => {
  it('keeps what the user has to look at out of a summary', () => {
    // A diff is the deliverable, a clarify is a question waiting on an answer,
    // an image is the thing that was asked for. None of them survives being
    // folded into "used 3 tools".
    for (const toolName of ['clarify', 'image_generate', 'edit_file', 'patch', 'write_file']) {
      expect(isCardTool(toolName)).toBe(true)
    }
  })

  it('treats reads, searches and commands as ephemeral activity', () => {
    for (const toolName of ['read_file', 'search_files', 'terminal', 'execute_code', 'web_search']) {
      expect(isCardTool(toolName)).toBe(false)
    }
  })
})

describe('isSilentTool', () => {
  it('names the rows that render nothing in the transcript', () => {
    // `todo` is hoisted to its own panel; a reaction's UI is the emoji on the
    // bubble. The render budget must not charge for either.
    expect(isSilentTool('todo')).toBe(true)
    expect(isSilentTool('react_to_message')).toBe(true)
    expect(isSilentTool('terminal')).toBe(false)
  })
})

describe('splitRunItems', () => {
  it('collapses a stretch of activity into one run', () => {
    expect(splitRunItems(['read_file', 'search_files', 'terminal'])).toEqual([{ end: 2, kind: 'run', start: 0 }])
  })

  it('keeps a card at the point in the turn where it happened', () => {
    // Read, edit, read has to stay in that order — a summary, the diff, then a
    // second summary — rather than sorting the diffs to one end.
    expect(splitRunItems(['read_file', 'patch', 'read_file', 'terminal'])).toEqual([
      { end: 0, kind: 'run', start: 0 },
      { index: 1, kind: 'card' },
      { end: 3, kind: 'run', start: 2 }
    ])
  })

  it('does not let adjacent cards merge into a run', () => {
    expect(splitRunItems(['patch', 'write_file'])).toEqual([
      { index: 0, kind: 'card' },
      { index: 1, kind: 'card' }
    ])
  })

  it('passes a part that is not a tool call through as its own card', () => {
    expect(splitRunItems(['read_file', '', 'read_file'])).toEqual([
      { end: 0, kind: 'run', start: 0 },
      { index: 1, kind: 'card' },
      { end: 2, kind: 'run', start: 2 }
    ])
  })

  it('has nothing to split when the range is empty', () => {
    expect(splitRunItems([])).toEqual([])
  })
})

describe('technicalTrace', () => {
  it('indents object payloads and persisted JSON strings', () => {
    expect(technicalTrace({ offset: 2, path: '/tmp/demo.txt' }, '{"success":true,"lines":["a","b"]}')).toBe(
      'Arguments:\n{\n  "offset": 2,\n  "path": "/tmp/demo.txt"\n}\n\nResult:\n{\n  "success": true,\n  "lines": [\n    "a",\n    "b"\n  ]\n}'
    )
  })

  it('leaves scalar strings untouched', () => {
    expect(technicalTrace(undefined, 'plain text')).toBe('Result:\nplain text')
    expect(technicalTrace(undefined, '"already quoted"')).toBe('Result:\n"already quoted"')
  })
})

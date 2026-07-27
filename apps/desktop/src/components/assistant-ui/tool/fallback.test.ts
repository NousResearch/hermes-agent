import { describe, expect, it } from 'vitest'

import { isUnboundableTool, shouldBoundToolGroup, technicalTrace } from './fallback'

describe('shouldBoundToolGroup', () => {
  it('bounds long runs of ordinary tool calls', () => {
    expect(shouldBoundToolGroup(3, false)).toBe(true)
  })

  it('leaves short runs unbounded', () => {
    expect(shouldBoundToolGroup(2, false)).toBe(false)
  })

  it('never bounds a run holding an unboundable tool', () => {
    expect(shouldBoundToolGroup(3, true)).toBe(false)
  })
})

describe('isUnboundableTool', () => {
  it('exempts clarify forms and generated images from the window', () => {
    expect(isUnboundableTool('clarify')).toBe(true)
    expect(isUnboundableTool('image_generate')).toBe(true)
  })

  // Everything ToolEntry renders carries `data-tool-row`, so the
  // `:has([data-tool-row][data-tool-open])` rule in styles.css lifts the cap
  // on its own. A diff row mounts open and frees the group immediately; a
  // collapsed row has no body in the DOM to clip. Exempting these in JS
  // instead vetoed grouping for the whole run — and since reads and edits are
  // most of a coding session, runs of 19 calls never collapsed at all.
  it('bounds the rows the CSS break-out already covers', () => {
    for (const toolName of ['read_file', 'execute_code', 'edit_file', 'patch', 'write_file']) {
      expect(isUnboundableTool(toolName)).toBe(false)
    }
  })

  it('still bounds console output and other ordinary rows', () => {
    expect(isUnboundableTool('terminal')).toBe(false)
    expect(isUnboundableTool('web_search')).toBe(false)
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

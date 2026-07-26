import { describe, expect, it } from 'vitest'

import { isUnboundableTool, shouldBoundToolGroup, technicalTrace } from './fallback'
import { isFileEditTool } from './fallback-model'

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

  it('exempts tools whose body is a code block the user reads', () => {
    expect(isUnboundableTool('execute_code')).toBe(true)
    expect(isUnboundableTool('read_file')).toBe(true)
  })

  // The window clips a diff to a ~2-row viewport, so every tool that renders
  // one has to be exempt. Derived from isFileEditTool rather than re-listed,
  // so a newly supported edit tool can't be exempt in one place and clipped
  // in the other.
  it('exempts every file-edit tool, so diffs are never clipped', () => {
    for (const toolName of ['edit_file', 'patch', 'write_file']) {
      expect(isFileEditTool(toolName)).toBe(true)
      expect(isUnboundableTool(toolName)).toBe(true)
    }
  })

  // Console output is a log tail: the last lines are the ones that matter,
  // which is exactly what the bounded window pins. Keeping it boundable is
  // what stops the exemption from swallowing the feature whole.
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

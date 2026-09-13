import { describe, expect, it } from 'vitest'

import { canonicalToolName, isCardTool, isFileEditTool, isSilentTool } from './tool-render-class'

describe('canonicalToolName', () => {
  it('maps Cursor chrome names onto Hermes tool ids', () => {
    expect(canonicalToolName('shell')).toBe('terminal')
    expect(canonicalToolName('read')).toBe('read_file')
    expect(canonicalToolName('edit')).toBe('edit_file')
    expect(canonicalToolName('get-mcp-tools')).toBe('get_mcp_tools')
  })

  it('classifies aliased edit/silent names the same as the Hermes id', () => {
    expect(isFileEditTool('search_replace')).toBe(true)
    expect(isCardTool('write')).toBe(true)
    expect(isSilentTool('todo-list')).toBe(true)
  })
})

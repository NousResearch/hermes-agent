import { describe, expect, it } from 'vitest'

import { type FileRowState, resolveFileRowClick } from './tree-gestures'

// The file browser's click contract (user-confirmed): a single click on a
// file opens it in the in-app Preview pane. Folders toggle; shift-click
// attaches. Component wiring is pinned separately in tree.wiring.test.tsx.

const state = (overrides: Partial<FileRowState> = {}): FileRowState => ({
  isFolder: false,
  isPlaceholder: false,
  isRenaming: false,
  ...overrides
})

describe('resolveFileRowClick (single click opens)', () => {
  it('opens a plain file', () => {
    expect(resolveFileRowClick(state())).toBe('open')
  })

  it('toggles a folder instead of opening it', () => {
    expect(resolveFileRowClick(state({ isFolder: true }))).toBe('toggle')
  })

  it('shift-click on a file attaches it', () => {
    expect(resolveFileRowClick(state({ shiftKey: true }))).toBe('attach-file')
  })

  it('shift-click on a folder attaches the folder', () => {
    expect(resolveFileRowClick(state({ isFolder: true, shiftKey: true }))).toBe('attach-folder')
  })

  it('ignores a click on a placeholder that is not interactive', () => {
    expect(resolveFileRowClick(state({ isPlaceholder: true }))).toBe('ignore')
  })

  it('ignores a click while an inline rename is active', () => {
    expect(resolveFileRowClick(state({ isRenaming: true }))).toBe('ignore')
  })
})

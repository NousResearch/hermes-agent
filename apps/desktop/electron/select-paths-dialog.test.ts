import { describe, expect, it } from 'vitest'

import { selectPathsDialogProperties } from './select-paths-dialog'

describe('selectPathsDialogProperties', () => {
  it('lets a directory picker create a new folder (macOS hides New Folder without createDirectory)', () => {
    const properties = selectPathsDialogProperties({ directories: true, multiple: false })

    expect(properties).toContain('openDirectory')
    expect(properties).toContain('createDirectory')
  })

  it('keeps file pickers file-only and multi-select by default', () => {
    const properties = selectPathsDialogProperties({})

    expect(properties).toContain('openFile')
    expect(properties).toContain('multiSelections')
    expect(properties).not.toContain('openDirectory')
    expect(properties).not.toContain('createDirectory')
  })

  it('drops multi-select only when the caller opts out', () => {
    expect(selectPathsDialogProperties({ directories: true })).toContain('multiSelections')
    expect(selectPathsDialogProperties({ directories: true, multiple: false })).not.toContain('multiSelections')
  })
})

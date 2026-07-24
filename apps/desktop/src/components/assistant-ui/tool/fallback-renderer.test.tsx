import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { shouldHideCompletedFileEdit, ToolHeaderSubtitle } from './fallback'

describe('ToolHeaderSubtitle', () => {
  it('renders the concise rich-surface reason or summary in a collapsed tool row', () => {
    render(<ToolHeaderSubtitle subtitle="Found 1 rendering component" />)

    expect(screen.getByText('Found 1 rendering component')).toBeTruthy()
  })

  it('does not add blank collapsed-row chrome for legacy events', () => {
    const { container } = render(<ToolHeaderSubtitle subtitle="" />)

    expect(container.childElementCount).toBe(0)
  })
})

describe('shouldHideCompletedFileEdit', () => {
  it('keeps a replayed successful file-edit card when safe activity metadata exists', () => {
    expect(
      shouldHideCompletedFileEdit({
        activity: 'Create the requested file · write_file: 1 file',
        inlineDiff: '',
        isFileEdit: true,
        isPending: false,
        status: 'success'
      })
    ).toBe(false)
  })

  it('still hides a legacy diff-less duplicate with no safe activity metadata', () => {
    expect(
      shouldHideCompletedFileEdit({
        activity: undefined,
        inlineDiff: '',
        isFileEdit: true,
        isPending: false,
        status: 'success'
      })
    ).toBe(true)
  })
})

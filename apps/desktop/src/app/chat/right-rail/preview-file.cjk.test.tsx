import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { MarkdownPreview } from './preview-file'

afterEach(cleanup)

// #92814 on the markdown file-preview surface: the CJK plugin must be wired
// into the preview renderer too, so a Korean quote followed by a particle
// renders as emphasis instead of literal `**` in a .md preview.
describe('MarkdownPreview CJK emphasis', () => {
  it('bolds a quoted Korean span followed by a particle', async () => {
    render(<MarkdownPreview filePath="/tmp/notes.md" text={'**“공개 문서는 점검한다”**는 원칙입니다.'} />)

    const strong = await screen.findByText('“공개 문서는 점검한다”')
    expect(strong.getAttribute('data-streamdown')).toBe('strong')
    expect(document.body.textContent).not.toContain('**')
  })
})

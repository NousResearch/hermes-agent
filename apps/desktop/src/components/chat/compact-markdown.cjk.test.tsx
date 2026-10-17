import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { CompactMarkdown } from './compact-markdown'

afterEach(cleanup)

// #92814 on the CompactMarkdown (tool detail body) surface: the CJK plugin
// must be wired here too, so Korean emphasis before particles doesn't leak
// literal `**` into tool output bodies.
describe('CompactMarkdown CJK emphasis', () => {
  it('bolds a quoted Korean span followed by a particle', async () => {
    render(<CompactMarkdown text={'**“공개 문서는 점검한다”**는 원칙입니다.'} />)

    const strong = await screen.findByText('“공개 문서는 점검한다”')
    expect(strong.getAttribute('data-streamdown')).toBe('strong')
    expect(document.body.textContent).not.toContain('**')
  })

  it('does not regress plain emphasis', async () => {
    render(<CompactMarkdown text="**bold** and normal text" />)

    const strong = await screen.findByText('bold')
    expect(strong.getAttribute('data-streamdown')).toBe('strong')
  })
})

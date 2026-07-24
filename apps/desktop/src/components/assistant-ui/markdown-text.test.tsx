import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { MarkdownLink } from './markdown-text'

afterEach(() => cleanup())

describe('MarkdownLink scheme allow-list', () => {
  it('does not emit a javascript: anchor (renders inert text instead)', () => {
    const { container } = render(
      <MarkdownLink href="javascript:alert(document.domain)">click me</MarkdownLink>
    )

    // The dangerous scheme must never reach an href in the DOM.
    const dangerous = Array.from(container.querySelectorAll('a')).filter((a) =>
      (a.getAttribute('href') ?? '').toLowerCase().includes('javascript:')
    )

    expect(dangerous).toHaveLength(0)
    // The link text is still shown to the user.
    expect(container.textContent).toContain('click me')
  })

  it('drops a data: URL link the same way', () => {
    const { container } = render(
      <MarkdownLink href="data:text/html,<script>alert(1)</script>">x</MarkdownLink>
    )

    const hrefs = Array.from(container.querySelectorAll('a')).map((a) =>
      (a.getAttribute('href') ?? '').toLowerCase()
    )

    expect(hrefs.some((h) => h.startsWith('data:'))).toBe(false)
  })
})

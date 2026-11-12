import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

afterEach(() => cleanup())

// End-to-end for the bare-autolink shape of #121007: a bare URL an agent sent
// must render with its full text visible — PrettyLink used to collapse it to a
// host-only label (`ncpssd.org`) with the address readable only via hover.
// Labeled links keep their authored label (that hiding is intended design).
describe('MarkdownTextContent bare URLs', () => {
  it('shows the full URL text for a bare autolink', () => {
    render(<MarkdownTextContent isRunning={false} text="See https://www.ncpssd.org/ for the database." />)

    const link = screen.getByRole('link') as HTMLAnchorElement

    expect(link.getAttribute('href')).toBe('https://www.ncpssd.org/')
    expect(link.textContent).toContain('https://www.ncpssd.org/')
  })

  it('does not collapse a host-only URL to a bare hostname', () => {
    render(<MarkdownTextContent isRunning={false} text="Docs live at https://example.com/ now." />)

    const link = screen.getByRole('link') as HTMLAnchorElement

    // urlSlugTitleLabel would render `example.com`; the full URL is the label.
    expect(link.textContent).toContain('https://example.com/')
  })

  it('keeps the authored label for a labeled markdown link', () => {
    render(<MarkdownTextContent isRunning={false} text="See [the database](https://www.ncpssd.org/) first." />)

    const link = screen.getByRole('link') as HTMLAnchorElement

    expect(link.getAttribute('href')).toBe('https://www.ncpssd.org/')
    expect(link.textContent).toContain('the database')
    expect(link.textContent).not.toContain('ncpssd.org/')
  })
})

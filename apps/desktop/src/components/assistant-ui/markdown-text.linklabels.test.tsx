import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

// Regression for #121321: authored markdown link labels are the text the
// model wrote, and formatted labels count. An inline-code label
// ([`v1.0.1`](url)) used to be dropped by childrenToText (it only handled
// plain strings), so the link fell through to a title fetch or a URL-slug
// fallback label that title-cased the identifier (`V1.0.1`). The authored
// label must win, with its exact casing preserved. Bare URLs keep their
// identifier casing too when no title is fetched (`README.md`, not
// `README.Md`).
describe('MarkdownLink authored labels', () => {
  afterEach(cleanup)

  it('preserves an inline-code link label with its exact casing', async () => {
    render(
      <MarkdownTextContent isRunning={false} text="Tagged [`v1.0.1`](https://example.com/releases/tag/v1.0.1)." />
    )

    await screen.findByText('v1.0.1')
    const anchor = screen.getByRole('link') as HTMLAnchorElement

    expect(anchor.getAttribute('href')).toBe('https://example.com/releases/tag/v1.0.1')
    expect(anchor.textContent).toContain('v1.0.1')
  })

  it('prefers the authored inline-code label over a fetched page title', async () => {
    const fetchLinkTitle = vi.fn().mockResolvedValue('Releases · example')

    ;(window as unknown as { hermesDesktop: object }).hermesDesktop = {
      fetchLinkTitle,
      openExternal: vi.fn().mockResolvedValue(undefined)
    }

    try {
      render(
        <MarkdownTextContent isRunning={false} text="Tagged [`v1.0.1`](https://example.com/releases/tag/v1.0.1)." />
      )

      await screen.findByText('v1.0.1')
      expect(fetchLinkTitle).not.toHaveBeenCalled()
    } finally {
      delete (window as unknown as { hermesDesktop?: object }).hermesDesktop
    }
  })

  it('keeps identifier casing in the fallback label of a bare URL', async () => {
    render(<MarkdownTextContent isRunning={false} text="See https://example.com/repository/blob/main/README.md" />)

    await screen.findByText('README.md')
  })

  it('keeps a plain authored label untouched', () => {
    render(<MarkdownTextContent isRunning={false} text="[v1.0.1](https://example.com/releases/tag/v1.0.1)" />)

    expect(screen.getByRole('link').textContent).toContain('v1.0.1')
  })
})

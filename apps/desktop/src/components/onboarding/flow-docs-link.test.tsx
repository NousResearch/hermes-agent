// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const openExternalLink = vi.fn()

vi.mock('@/lib/external-link', () => ({
  openExternalLink: (href: string) => openExternalLink(href)
}))

const { DocsLink } = await import('./flow')

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('DocsLink (onboarding sign-in)', () => {
  it('opens the provider docs through the validated external opener', () => {
    const docsUrl = 'https://docs.github.com/en/copilot'
    render(<DocsLink href={docsUrl}>GitHub Copilot (ACP) docs</DocsLink>)

    fireEvent.click(screen.getByText('GitHub Copilot (ACP) docs'))

    expect(openExternalLink).toHaveBeenCalledWith(docsUrl)
  })
})

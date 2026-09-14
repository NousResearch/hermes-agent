// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

const openExternalLink = vi.fn()

vi.mock('@/lib/external-link', () => ({
  openExternalLink: (href: string) => openExternalLink(href)
}))

const { CredentialDocsLink } = await import('./credential-key-ui')

function renderLink(href: string) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <CredentialDocsLink href={href} />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('CredentialDocsLink (provider API-key docs)', () => {
  it('opens the docs URL through the validated external opener', () => {
    const docsUrl = 'https://openrouter.ai/keys'
    renderLink(docsUrl)

    fireEvent.click(screen.getByRole('link'))

    expect(openExternalLink).toHaveBeenCalledWith(docsUrl)
  })
})

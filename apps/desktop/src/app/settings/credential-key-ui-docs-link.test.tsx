// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop
let openExternal: ReturnType<typeof vi.fn>

beforeEach(() => {
  openExternal = vi.fn().mockResolvedValue(undefined)
  desktopWindow.hermesDesktop = { openExternal } as unknown as Window['hermesDesktop']
})

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

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('CredentialDocsLink (provider API-key docs)', () => {
  it('opens the docs URL in the OS browser', () => {
    const docsUrl = 'https://openrouter.ai/keys'
    renderLink(docsUrl)

    fireEvent.click(screen.getByRole('link'))

    expect(openExternal).toHaveBeenCalledWith(docsUrl)
  })
})

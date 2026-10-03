import { cleanup, createEvent, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { CredentialDocsLink } from './credential-key-ui'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

function installDesktopBridge() {
  desktopWindow.hermesDesktop = {
    fetchLinkTitle: vi.fn().mockResolvedValue(''),
    openExternal: vi.fn().mockResolvedValue(undefined)
  } as unknown as Window['hermesDesktop']
}

beforeEach(() => {
  installDesktopBridge()
})

afterEach(() => {
  vi.restoreAllMocks()
  cleanup()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('CredentialDocsLink', () => {
  it('routes clicks through the audited openExternal bridge (not a raw target=_blank)', () => {
    render(
      <I18nProvider>
        <CredentialDocsLink href="https://platform.deepseek.com/api_keys" />
      </I18nProvider>
    )

    const link = screen.getByRole('link', { name: /get a key/i })
    fireEvent.click(link)

    expect(window.hermesDesktop?.openExternal).toHaveBeenCalledWith('https://platform.deepseek.com/api_keys')
  })

  it('renders the href the click opens', () => {
    render(
      <I18nProvider>
        <CredentialDocsLink href="https://platform.deepseek.com/api_keys" />
      </I18nProvider>
    )

    const link = screen.getByRole('link', { name: /get a key/i })
    expect(link.getAttribute('href')).toBe('https://platform.deepseek.com/api_keys')
  })

  it('stops the click so the parent card does not collapse when the link is clicked', () => {
    const parentClick = vi.fn()
    render(
      <div onClick={parentClick}>
        <I18nProvider>
          <CredentialDocsLink href="https://platform.deepseek.com/api_keys" />
        </I18nProvider>
      </div>
    )

    const link = screen.getByRole('link', { name: /get a key/i })
    fireEvent.click(link)

    expect(parentClick).not.toHaveBeenCalled()
    expect(window.hermesDesktop?.openExternal).toHaveBeenCalled()
  })

  it('cancels nothing when there is no desktop bridge to take over the click', () => {
    delete desktopWindow.hermesDesktop

    render(
      <I18nProvider>
        <CredentialDocsLink href="https://platform.deepseek.com/api_keys" />
      </I18nProvider>
    )

    const link = screen.getByRole('link', { name: /get a key/i })
    const click = createEvent.click(link)
    fireEvent(link, click)

    // preventDefault() here would cancel the navigation with nothing to replace it.
    expect(click.defaultPrevented).toBe(false)
    expect(link.getAttribute('target')).toBe('_blank')
  })
})
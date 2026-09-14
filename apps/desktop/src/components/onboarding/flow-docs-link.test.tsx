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

const { DocsLink } = await import('./flow')

function renderLink(href: string) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <DocsLink href={href}>GitHub Copilot (ACP) docs</DocsLink>
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

describe('DocsLink (onboarding sign-in)', () => {
  it('opens the provider docs in the OS browser', () => {
    const docsUrl = 'https://docs.github.com/en/copilot'
    renderLink(docsUrl)

    fireEvent.click(screen.getByText('GitHub Copilot (ACP) docs'))

    expect(openExternal).toHaveBeenCalledWith(docsUrl)
  })
})

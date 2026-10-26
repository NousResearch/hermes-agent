import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { CopyButton } from './copy-button'

describe('CopyButton rich HTML flavour', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('sends text + html through the desktop bridge, resolving html from the button element', async () => {
    const writeClipboardRich = vi.fn().mockResolvedValue(true)

    const writeClipboard = vi.fn().mockResolvedValue(true)

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { writeClipboard, writeClipboardRich }

    const html = vi.fn((anchor: HTMLElement | null) => (anchor?.tagName === 'BUTTON' ? '<p>hi</p>' : null))

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <CopyButton html={html} text="hi" />
      </I18nProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Copy' }))

    await waitFor(() => expect(writeClipboardRich).toHaveBeenCalledWith({ html: '<p>hi</p>', text: 'hi' }))
    expect(writeClipboard).not.toHaveBeenCalled()
  })

  it('stays text-only when the html resolver yields nothing', async () => {
    const writeClipboardRich = vi.fn().mockResolvedValue(true)

    const writeClipboard = vi.fn().mockResolvedValue(true)

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { writeClipboard, writeClipboardRich }

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <CopyButton html={() => null} text="plain" />
      </I18nProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Copy' }))

    await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith('plain'))
    expect(writeClipboardRich).not.toHaveBeenCalled()
  })
})

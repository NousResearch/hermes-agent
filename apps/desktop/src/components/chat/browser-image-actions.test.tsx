import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const browserActions = vi.hoisted(() => ({
  BrowserTabLimitError: class BrowserTabLimitError extends Error {},
  BrowserUnsupportedUrlError: class BrowserUnsupportedUrlError extends Error {},
  openBrowserQc: vi.fn(),
  openBrowserSurface: vi.fn()
}))

const notifications = vi.hoisted(() => ({ notifyError: vi.fn() }))

vi.mock('@/app/browser/store', () => browserActions)
vi.mock('@/store/notifications', () => notifications)
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { failed: 'Failed' },
      desktop: {
        browserTabLimit: 'Browser tab limit reached',
        browserInvalidUrl: 'Enter a valid URL',
        openInBrowser: 'Open in browser',
        openInQc: 'Open in QC'
      }
    }
  })
}))

import { BrowserImageActions } from './browser-image-actions'
const { BrowserTabLimitError, BrowserUnsupportedUrlError, openBrowserQc, openBrowserSurface } = browserActions
const { notifyError } = notifications

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('BrowserImageActions', () => {
  it('never opens a browser surface until the user chooses an action', () => {
    render(<BrowserImageActions src="https://example.test/image.png" />)

    expect(openBrowserSurface).not.toHaveBeenCalled()
    expect(openBrowserQc).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Open in browser' }))
    expect(openBrowserSurface).toHaveBeenCalledWith({ url: 'https://example.test/image.png' })
    expect(openBrowserQc).not.toHaveBeenCalled()
  })

  it('opens QC only from its explicit action', () => {
    render(<BrowserImageActions src="data:image/png;base64,AAAA" />)

    fireEvent.click(screen.getByRole('button', { name: 'Open in QC' }))
    expect(openBrowserQc).toHaveBeenCalledWith({ url: 'data:image/png;base64,AAAA' })
    expect(openBrowserSurface).not.toHaveBeenCalled()
  })

  it('uses the localized tab-limit message without exposing the internal error text', () => {
    const error = new BrowserTabLimitError('All browser tabs are pinned.')
    openBrowserSurface.mockImplementationOnce(() => {
      throw error
    })
    openBrowserQc.mockImplementationOnce(() => {
      throw error
    })
    render(<BrowserImageActions src="https://example.test/image.png" />)

    fireEvent.click(screen.getByRole('button', { name: 'Open in browser' }))
    fireEvent.click(screen.getByRole('button', { name: 'Open in QC' }))

    expect(notifyError).toHaveBeenNthCalledWith(1, undefined, 'Browser tab limit reached')
    expect(notifyError).toHaveBeenNthCalledWith(2, undefined, 'Browser tab limit reached')
  })
  it('shows the localized invalid URL error for an unsupported generic image source', () => {
    const error = new BrowserUnsupportedUrlError('Browser URL is unsupported.')
    openBrowserSurface.mockImplementationOnce(() => {
      throw error
    })
    openBrowserQc.mockImplementationOnce(() => {
      throw error
    })
    render(<BrowserImageActions src="data:image/svg+xml,<svg/>" />)

    fireEvent.click(screen.getByRole('button', { name: 'Open in browser' }))
    fireEvent.click(screen.getByRole('button', { name: 'Open in QC' }))

    expect(openBrowserSurface).toHaveBeenCalledWith({ url: 'data:image/svg+xml,<svg/>' })
    expect(openBrowserQc).toHaveBeenCalledWith({ url: 'data:image/svg+xml,<svg/>' })
    expect(notifyError).toHaveBeenNthCalledWith(1, undefined, 'Enter a valid URL')
    expect(notifyError).toHaveBeenNthCalledWith(2, undefined, 'Enter a valid URL')
  })

  it('preserves generic error handling', () => {
    const error = new Error('Unexpected failure')
    openBrowserSurface.mockImplementationOnce(() => {
      throw error
    })
    render(<BrowserImageActions src="https://example.test/image.png" />)

    fireEvent.click(screen.getByRole('button', { name: 'Open in browser' }))

    expect(notifyError).toHaveBeenCalledWith(error, 'Failed')
  })
  it('renders no action for an empty source', () => {
    const { container } = render(<BrowserImageActions />)

    expect(container.innerHTML).toBe('')
  })
})

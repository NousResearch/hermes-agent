import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ExternalOpenFailedDialog } from './external-open-failed-dialog'

const windowsMock = vi.hoisted(() => ({
  isHudWindow: vi.fn(() => false),
  isBrowserWindow: vi.fn(() => false)
}))

vi.mock('@/store/windows', () => windowsMock)

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

function installBridge() {
  const onExternalOpenFailed = vi.fn()
  const writeClipboard = vi.fn().mockResolvedValue(undefined)

  desktopWindow.hermesDesktop = {
    onExternalOpenFailed,
    writeClipboard
  } as unknown as Window['hermesDesktop']

  return { onExternalOpenFailed, writeClipboard }
}

function fail(listener: unknown, url: string, message?: string) {
  act(() => {
    ;(listener as (payload: { url: string; message?: string }) => void)({ url, message })
  })
}

afterEach(() => {
  windowsMock.isHudWindow.mockReturnValue(false)
  windowsMock.isBrowserWindow.mockReturnValue(false)
  vi.restoreAllMocks()
  cleanup()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('ExternalOpenFailedDialog', () => {
  it('subscribes to open-failure events on mount', () => {
    const { onExternalOpenFailed } = installBridge()
    render(<ExternalOpenFailedDialog />)
    expect(onExternalOpenFailed).toHaveBeenCalledTimes(1)
    expect(typeof onExternalOpenFailed.mock.calls[0][0]).toBe('function')
  })

  it('shows the URL when an open-failure event fires', () => {
    const { onExternalOpenFailed } = installBridge()
    render(<ExternalOpenFailedDialog />)

    const listener = onExternalOpenFailed.mock.calls[0][0]
    expect(screen.queryByText('https://example.com/dead')).toBeNull()
    fail(listener, 'https://example.com/dead')
    expect(screen.getByText('https://example.com/dead')).not.toBeNull()
  })

  it('copies the URL when the copy action is used', async () => {
    const { onExternalOpenFailed, writeClipboard } = installBridge()
    render(<ExternalOpenFailedDialog />)

    const listener = onExternalOpenFailed.mock.calls[0][0]
    fail(listener, 'https://example.com/dead')

    fireEvent.click(screen.getByRole('button', { name: /copy/i }))

    await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith('https://example.com/dead'))
  })

  it('closes when dismissed', async () => {
    const { onExternalOpenFailed } = installBridge()
    render(<ExternalOpenFailedDialog />)

    const listener = onExternalOpenFailed.mock.calls[0][0]
    fail(listener, 'https://example.com/dead')

    fireEvent.click(screen.getByRole('button', { name: 'Close' }))

    await waitFor(() => expect(screen.queryByText('https://example.com/dead')).toBeNull())
  })

  it('renders nothing in a HUD window', () => {
    installBridge()
    windowsMock.isHudWindow.mockReturnValue(true)

    const view = render(<ExternalOpenFailedDialog />)

    expect(view.container.firstChild).toBeNull()
  })
})

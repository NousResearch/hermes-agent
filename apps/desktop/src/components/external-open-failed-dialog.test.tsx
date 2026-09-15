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
  it('subscribes, displays, copies and dismisses an external-open failure', async (): Promise<void> => {
    const { onExternalOpenFailed, writeClipboard }: ReturnType<typeof installBridge> = installBridge()
    render(<ExternalOpenFailedDialog />)
    expect(onExternalOpenFailed).toHaveBeenCalledTimes(1)
    expect(screen.queryByText('https://example.com/dead')).toBeNull()
    const listener: (payload: { url: string; message?: string }) => void = onExternalOpenFailed.mock.calls[0][0]
    fail(listener, 'https://example.com/dead')
    expect(screen.getByText('https://example.com/dead')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /copy/i }))
    await waitFor((): void => { expect(writeClipboard).toHaveBeenCalledWith('https://example.com/dead') })
    fireEvent.click(screen.getByRole('button', { name: 'Close' }))
    await waitFor((): void => { expect(screen.queryByText('https://example.com/dead')).toBeNull() })
  })

  it('renders nothing in a HUD window', () => {
    installBridge()
    windowsMock.isHudWindow.mockReturnValue(true)

    const view = render(<ExternalOpenFailedDialog />)

    expect(view.container.firstChild).toBeNull()
  })
})

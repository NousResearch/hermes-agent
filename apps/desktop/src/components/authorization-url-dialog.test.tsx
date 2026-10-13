import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { dismissAuthorizationUrl, presentAuthorizationUrl } from '@/lib/authorization-url'

import { AuthorizationUrlDialog } from './authorization-url-dialog'

const windowsMock = vi.hoisted(() => ({
  isHudWindow: vi.fn(() => false),
  isBrowserWindow: vi.fn(() => false)
}))

vi.mock('@/store/windows', () => windowsMock)

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

afterEach(() => {
  dismissAuthorizationUrl()
  windowsMock.isHudWindow.mockReturnValue(false)
  windowsMock.isBrowserWindow.mockReturnValue(false)
  cleanup()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('AuthorizationUrlDialog', () => {
  it('shows a copyable auth URL and does not open the default browser until asked', async () => {
    const openExternal = vi.fn().mockResolvedValue(undefined)
    const writeClipboard = vi.fn().mockResolvedValue(undefined)

    desktopWindow.hermesDesktop = { openExternal, writeClipboard } as unknown as Window['hermesDesktop']
    render(<AuthorizationUrlDialog />)
    act(() => {
      presentAuthorizationUrl('https://idp.example/authorize?state=abc')
    })

    const url = screen.getByText('https://idp.example/authorize?state=abc')

    expect(url.className).toContain('select-text')
    expect(openExternal).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Copy link' }))
    await waitFor(() => {
      expect(writeClipboard).toHaveBeenCalledWith('https://idp.example/authorize?state=abc')
    })
    fireEvent.click(screen.getByRole('button', { name: 'Open in default browser' }))
    await waitFor(() => {
      expect(openExternal).toHaveBeenCalledExactlyOnceWith('https://idp.example/authorize?state=abc')
    })
  })
})

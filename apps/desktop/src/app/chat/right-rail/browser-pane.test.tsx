import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $browserCurrentState, driveBrowser, resetBrowserRegistryForTests, setBrowserSessionState } from '@/store/browser'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

import { BrowserPane } from './browser-pane'

describe('BrowserPane', () => {
  beforeEach(() => {
    resetBrowserRegistryForTests()
    $activeSessionId.set('session-1')
    $selectedStoredSessionId.set(null)
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('renders an isolated browser webview', () => {
    render(<BrowserPane setTitlebarToolGroup={vi.fn()} />)

    const webview = document.querySelector('webview')

    expect(webview).toBeInstanceOf(HTMLElement)
    expect(webview?.getAttribute('partition')).toBe('persist:hermes-browser-session-1')
    expect(webview?.getAttribute('webpreferences')).toContain('nodeIntegration=no')
    expect(webview?.getAttribute('src')).toBe('about:blank')
  })

  it('normalizes address submissions and persists the session URL', () => {
    render(<BrowserPane setTitlebarToolGroup={vi.fn()} />)

    const input = screen.getByLabelText('Browser address')

    fireEvent.change(input, { target: { value: 'example.com' } })
    fireEvent.submit(input.closest('form')!)

    expect($browserCurrentState.get().url).toBe('https://example.com/')
    expect(document.querySelector('webview')?.getAttribute('src')).toBe('https://example.com/')
  })

  it('updates the address and store when the webview navigates', () => {
    setBrowserSessionState('session-1', { url: 'https://example.com' })
    render(<BrowserPane setTitlebarToolGroup={vi.fn()} />)

    const webview = document.querySelector('webview')!

    act(() => {
      webview.dispatchEvent(Object.assign(new Event('did-navigate'), { url: 'https://nousresearch.com/' }))
    })

    expect((screen.getByLabelText('Browser address') as HTMLInputElement).value).toBe('https://nousresearch.com/')
    expect($browserCurrentState.get().url).toBe('https://nousresearch.com/')
  })

  it('applies drive commands to the live webview', () => {
    render(<BrowserPane setTitlebarToolGroup={vi.fn()} />)

    const webview = document.querySelector('webview') as HTMLElement & { reload?: () => void }
    webview.reload = vi.fn()

    act(() => driveBrowser({ action: 'reload' }))

    expect(webview.reload).toHaveBeenCalled()
  })

  it('ignores drive commands for a different session', () => {
    render(<BrowserPane setTitlebarToolGroup={vi.fn()} />)

    const webview = document.querySelector('webview') as HTMLElement & { reload?: () => void }
    webview.reload = vi.fn()

    act(() => driveBrowser({ action: 'reload', sessionId: 'session-2' }))

    expect(webview.reload).not.toHaveBeenCalled()
  })
})

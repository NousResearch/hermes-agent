import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { browserNavigate, BrowserSlot, captureBrowserTab, PersistentBrowser, saveBrowserCapture } from './persistent'
import { $browserCapture, $browserOpen, $browserState, BROWSER_QC_DIMENSIONS, type BrowserQc } from './store'

const emptyQc = (): BrowserQc =>
  Object.fromEntries(
    BROWSER_QC_DIMENSIONS.map(dimension => [dimension, { evidence: '', note: '', status: 'unchecked' }])
  ) as BrowserQc

const originalDesktop = window.hermesDesktop
const capture = vi.fn()
const saveCapture = vi.fn()

beforeEach(() => {
  capture.mockReset()
  saveCapture.mockReset()
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { ...originalDesktop, browser: { capture, saveCapture } } as unknown as Window['hermesDesktop']
  })
  $browserCapture.set(null)
  $browserOpen.set(true)
  $browserState.set({
    activeTabId: 'tab-1',
    qcOpen: false,
    tabs: [
      { id: 'tab-1', pinned: false, qc: emptyQc(), title: 'One', url: 'https://example.test/one' },
      { id: 'tab-2', pinned: false, qc: emptyQc(), title: 'Two', url: 'https://example.test/two' }
    ]
  })
})

afterEach(() => {
  cleanup()
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: originalDesktop })
})

function Harness() {
  return (
    <>
      <BrowserSlot />
      <PersistentBrowser />
    </>
  )
}

describe('PersistentBrowser', () => {
  it('keeps every tab guest mounted while the pane is hidden', async () => {
    const { container } = render(<Harness />)
    const guests = container.querySelectorAll('webview')
    expect(guests).toHaveLength(2)
    const first = guests[0]

    act(() => $browserOpen.set(false))

    await waitFor(() => expect(first.parentElement?.getAttribute('aria-hidden')).toBe('true'))
    expect(container.querySelectorAll('webview')).toHaveLength(2)
    expect(container.querySelectorAll('webview')[0]).toBe(first)
  })

  it('navigates through the existing guest controlled by store state', () => {
    const { container } = render(<Harness />)
    const guest = container.querySelector('webview')

    act(() => browserNavigate('tab-1', 'https://example.test/next'))

    expect($browserState.get().tabs[0].url).toBe('https://example.test/next')
    expect(container.querySelector('webview')).toBe(guest)
    expect(guest?.getAttribute('src')).toBe('https://example.test/next')
  })

  it('updates only the owning tab from guest navigation events', () => {
    const { container } = render(<Harness />)
    const event = new Event('did-navigate') as Event & { url: string }
    event.url = 'https://example.test/navigated'
    act(() => container.querySelectorAll('webview')[0].dispatchEvent(event))

    expect($browserState.get().tabs.map(tab => tab.url)).toEqual([
      'https://example.test/navigated',
      'https://example.test/two'
    ])
  })

  it('throws when the tab guest id is unavailable for capture', async () => {
    render(<Harness />)

    await expect(captureBrowserTab('tab-1')).rejects.toThrow(
      'Browser tab is unavailable for capture: guest id is missing.'
    )
  })

  it('throws when the capture bridge is unavailable', async () => {
    const { container } = render(<Harness />)
    const guest = container.querySelector('webview') as HTMLElement & { getWebContentsId?: () => number }
    guest.getWebContentsId = () => 42
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { ...originalDesktop, browser: undefined } as unknown as Window['hermesDesktop']
    })

    await expect(captureBrowserTab('tab-1')).rejects.toThrow(
      'Browser capture is unavailable: desktop bridge is missing.'
    )
  })

  it.each([undefined, null])('throws when the capture bridge returns no capture', async captureResult => {
    const { container } = render(<Harness />)
    const guest = container.querySelector('webview') as HTMLElement & { getWebContentsId?: () => number }
    guest.getWebContentsId = () => 42
    capture.mockResolvedValue(captureResult)

    await expect(captureBrowserTab('tab-1')).rejects.toThrow('Browser capture failed: capture result is missing.')
    expect($browserCapture.get()).toBeNull()
  })

  it('captures through the typed guest bridge and keeps bytes transient', async () => {
    const { container } = render(<Harness />)
    const guest = container.querySelector('webview') as HTMLElement & { getWebContentsId?: () => number }
    guest.getWebContentsId = () => 42

    const expectedCapture = {
      captureId: 'capture-1',
      createdAt: 123,
      dataUrl: 'data:image/png;base64,AAAA',
      height: 720,
      width: 1280
    }

    capture.mockResolvedValue(expectedCapture)

    await expect(captureBrowserTab('tab-1')).resolves.toEqual(expectedCapture)

    expect(capture).toHaveBeenCalledWith(42)
    expect($browserCapture.get()).toEqual({
      ...expectedCapture,
      tabId: 'tab-1'
    })
  })

  it('throws when the save bridge is unavailable', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { ...originalDesktop, browser: undefined } as unknown as Window['hermesDesktop']
    })

    await expect(saveBrowserCapture('capture-1')).rejects.toThrow(
      'Browser capture saving is unavailable: desktop bridge is missing.'
    )
  })

  it('saves a capture through the typed bridge', async () => {
    const result = { canceled: false, path: '/tmp/capture.png' }
    saveCapture.mockResolvedValue(result)

    await expect(saveBrowserCapture('capture-1', 'capture.png')).resolves.toEqual(result)
    expect(saveCapture).toHaveBeenCalledWith('capture-1', 'capture.png')
  })
})

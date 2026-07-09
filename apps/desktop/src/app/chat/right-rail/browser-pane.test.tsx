import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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

  it('captures a semantic browser snapshot from the live webview', async () => {
    const completeSnapshot = vi.fn().mockResolvedValue({ ok: true })

    window.hermesDesktop = {
      ...window.hermesDesktop,
      browser: {
        ...window.hermesDesktop?.browser,
        completeSnapshot,
        navigate: vi.fn().mockResolvedValue(undefined),
        open: vi.fn().mockResolvedValue(undefined),
        updateState: vi.fn()
      }
    }

    render(<BrowserPane setTitlebarToolGroup={vi.fn()} />)

    const webview = document.querySelector('webview') as HTMLElement & {
      executeJavaScript?: ReturnType<typeof vi.fn>
    }

    webview.executeJavaScript = vi.fn().mockResolvedValue({
      capturedAt: 123,
      elements: [{ index: 0, tag: 'button', text: '发布', visible: true }],
      headings: [{ level: 1, text: '抖音商家后台' }],
      ok: true,
      tables: [],
      text: '订单 数据 客户消息',
      title: '商家后台',
      url: 'https://fxg.jinritemai.com/'
    })

    act(() => driveBrowser({ action: 'snapshot', requestId: 'req-1' }))

    await waitFor(() => expect(completeSnapshot).toHaveBeenCalled())
    expect(webview.executeJavaScript).toHaveBeenCalledWith(expect.stringContaining('document.querySelectorAll'), false)
    expect(completeSnapshot).toHaveBeenCalledWith({
      requestId: 'req-1',
      sessionId: 'session-1',
      snapshot: expect.objectContaining({
        elements: [expect.objectContaining({ text: '发布' })],
        ok: true,
        sessionId: 'session-1',
        text: '订单 数据 客户消息',
        title: '商家后台',
        url: 'https://fxg.jinritemai.com/'
      })
    })
  })

  it('executes DOM actions in the live webview and returns action results', async () => {
    const completeAction = vi.fn().mockResolvedValue({ ok: true })

    window.hermesDesktop = {
      ...window.hermesDesktop,
      browser: {
        ...window.hermesDesktop?.browser,
        completeAction,
        navigate: vi.fn().mockResolvedValue(undefined),
        open: vi.fn().mockResolvedValue(undefined),
        updateState: vi.fn()
      }
    }

    render(<BrowserPane setTitlebarToolGroup={vi.fn()} />)

    const webview = document.querySelector('webview') as HTMLElement & {
      executeJavaScript?: ReturnType<typeof vi.fn>
    }

    webview.executeJavaScript = vi.fn().mockResolvedValue({
      action: 'type',
      capturedAt: 456,
      ok: true,
      target: { index: 0, tag: 'textarea', text: 'hello' },
      title: '商家后台',
      url: 'https://fxg.jinritemai.com/im'
    })

    act(() => driveBrowser({ action: 'act', domAction: { index: 0, kind: 'type', text: 'hello' }, requestId: 'act-1' }))

    await waitFor(() => expect(completeAction).toHaveBeenCalled())
    expect(webview.executeJavaScript).toHaveBeenCalledWith(expect.stringContaining('const action = {"index":0'), true)
    expect(completeAction).toHaveBeenCalledWith({
      requestId: 'act-1',
      result: expect.objectContaining({
        action: 'type',
        ok: true,
        sessionId: 'session-1',
        target: expect.objectContaining({ text: 'hello' }),
        url: 'https://fxg.jinritemai.com/im'
      }),
      sessionId: 'session-1'
    })
  })
})

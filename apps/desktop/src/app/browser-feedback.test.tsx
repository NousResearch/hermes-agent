import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  BrowserFeedbackWindow,
  buildElementChangePrompt,
  normalizeBrowserFeedbackUrl,
  type PickedBrowserElement
} from './browser-feedback'

const element: PickedBrowserElement = {
  attributes: {
    'aria-label': 'Save changes',
    type: 'button'
  },
  className: 'btn primary rounded',
  id: 'save-button',
  outerHtml: '<button id="save-button" class="btn primary rounded" aria-label="Save changes">Save</button>',
  rect: {
    height: 32,
    width: 96,
    x: 120,
    y: 48
  },
  role: 'button',
  selector: 'button#save-button',
  tagName: 'button',
  text: 'Save',
  url: 'https://example.com/settings',
  xpath: '//*[@id="save-button"]'
}

interface FakeWebview extends HTMLElement {
  canGoBack: () => boolean
  canGoForward: () => boolean
  executeJavaScript: ReturnType<typeof vi.fn>
  getURL: () => string
  goBack: ReturnType<typeof vi.fn>
  goForward: ReturnType<typeof vi.fn>
  isLoading: () => boolean
  loadURL: ReturnType<typeof vi.fn>
  reload: ReturnType<typeof vi.fn>
  setUserAgent: ReturnType<typeof vi.fn>
}

function installFakeWebview() {
  const originalCreateElement = document.createElement.bind(document)
  const webviews: FakeWebview[] = []

  vi.spyOn(document, 'createElement').mockImplementation(((tagName: string, options?: ElementCreationOptions) => {
    if (tagName.toLowerCase() !== 'webview') {
      return originalCreateElement(tagName, options)
    }

    let url = ''
    const webview = originalCreateElement('div') as unknown as FakeWebview
    webview.canGoBack = vi.fn(() => true)
    webview.canGoForward = vi.fn(() => false)
    webview.executeJavaScript = vi.fn(async () => true)
    webview.getURL = vi.fn(() => url || webview.getAttribute('src') || '')
    webview.goBack = vi.fn()
    webview.goForward = vi.fn()
    webview.isLoading = vi.fn(() => false)
    webview.loadURL = vi.fn(async (nextUrl: string) => {
      url = nextUrl
      webview.dispatchEvent(new Event('did-start-loading'))
      webview.dispatchEvent(Object.assign(new Event('did-navigate'), { url: nextUrl }))
      webview.dispatchEvent(new Event('did-stop-loading'))
    })
    webview.reload = vi.fn()
    webview.setUserAgent = vi.fn()
    webviews.push(webview)

    return webview
  }) as typeof document.createElement)

  return webviews
}

function TestBrowserFeedbackWindow({
  onClose = vi.fn(),
  onInsertPrompt = vi.fn()
}: {
  onClose?: () => void
  onInsertPrompt?: (prompt: string) => void
}) {
  const [minimized, setMinimized] = useState(false)

  return (
    <BrowserFeedbackWindow
      minimized={minimized}
      onClose={onClose}
      onInsertPrompt={onInsertPrompt}
      onMinimizedChange={setMinimized}
      open
    />
  )
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('browser feedback helpers', () => {
  it('normalizes typed addresses into loadable browser URLs', () => {
    expect(normalizeBrowserFeedbackUrl('')).toBe('about:blank')
    expect(normalizeBrowserFeedbackUrl('example.com/path')).toBe('https://example.com/path')
    expect(normalizeBrowserFeedbackUrl('http://localhost:5174')).toBe('http://localhost:5174')
    expect(normalizeBrowserFeedbackUrl('  https://nousresearch.com/docs  ')).toBe('https://nousresearch.com/docs')
  })

  it('builds a chat prompt with the selected element metadata and user comment', () => {
    const prompt = buildElementChangePrompt(element, 'Make this button larger and easier to notice.')

    expect(prompt).toContain('Visual change request')
    expect(prompt).toContain('URL: https://example.com/settings')
    expect(prompt).toContain('tag: button')
    expect(prompt).toContain('id: save-button')
    expect(prompt).toContain('classes: btn primary rounded')
    expect(prompt).toContain('selector: button#save-button')
    expect(prompt).toContain('aria-label="Save changes"')
    expect(prompt).toContain('Make this button larger and easier to notice.')
  })
})

describe('BrowserFeedbackWindow', () => {
  it('loads typed URLs through the existing webview instead of recreating it', async () => {
    const webviews = installFakeWebview()

    const { container } = render(<TestBrowserFeedbackWindow />)
    const webview = webviews[0]

    expect(webview.getAttribute('src')).toBe('about:blank')

    act(() => {
      webview.dispatchEvent(new Event('dom-ready'))
    })

    const addressInput = container.querySelector('input')
    expect(addressInput).not.toBeNull()
    fireEvent.change(addressInput!, {
      target: { value: 'example.com/changed' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Load' }))

    await vi.waitFor(() => {
      expect(webview.loadURL).toHaveBeenCalledWith(
        'https://example.com/changed',
        expect.objectContaining({ userAgent: expect.any(String) })
      )
    })
    expect(webviews).toHaveLength(1)
  })

  it('wires browser controls and picker after the webview is ready', async () => {
    const webviews = installFakeWebview()
    const onInsertPrompt = vi.fn()

    render(<TestBrowserFeedbackWindow onInsertPrompt={onInsertPrompt} />)
    const webview = webviews[0]

    act(() => {
      webview.dispatchEvent(new Event('dom-ready'))
    })

    fireEvent.click(screen.getByTitle('Back'))
    fireEvent.click(screen.getByTitle('Reload'))

    expect(webview.goBack).toHaveBeenCalledTimes(1)
    expect(webview.reload).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole('button', { name: 'Pick element' }))

    await vi.waitFor(() => {
      expect(webview.executeJavaScript).toHaveBeenCalledTimes(1)
    })

    act(() => {
      webview.dispatchEvent(
        Object.assign(new Event('console-message'), {
          message: `__HERMES_BROWSER_PICK__${JSON.stringify({ cancelled: false, element })}`
        })
      )
    })

    await vi.waitFor(() => {
      expect(screen.getAllByText(/button#save-button/).length).toBeGreaterThan(0)
    })

    const commentInput = screen.getByPlaceholderText(/make this button larger/i)
    fireEvent.change(commentInput, {
      target: { value: 'Make this control more visible.' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Add request to chat' }))

    expect(webview.executeJavaScript).toHaveBeenCalledTimes(1)
    expect(onInsertPrompt).toHaveBeenCalledWith(expect.stringContaining('Make this control more visible.'))
    expect((commentInput as HTMLTextAreaElement).value).toBe('')
  })

  it('switches viewport presets without recreating the webview', () => {
    const webviews = installFakeWebview()

    render(<TestBrowserFeedbackWindow />)
    const webview = webviews[0]

    act(() => {
      webview.dispatchEvent(new Event('dom-ready'))
    })

    fireEvent.change(screen.getByRole('combobox', { name: /viewport/i }), {
      target: { value: 'mobile-390x844' }
    })

    expect(webview.setUserAgent).toHaveBeenCalledWith(expect.stringContaining('Mobile'))
    expect(webviews).toHaveLength(1)
  })

  it('can minimize without showing a separate restore button or closing the browser view', () => {
    const webviews = installFakeWebview()

    render(<TestBrowserFeedbackWindow />)
    expect(webviews).toHaveLength(1)

    fireEvent.click(screen.getByTitle('Minimize Web Browser'))

    expect(screen.queryByTitle('Restore Web Browser')).toBeNull()
    expect(screen.queryByTitle('Minimize Web Browser')).not.toBeNull()
    expect(webviews).toHaveLength(1)
  })
})

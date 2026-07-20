import { webcrypto } from 'node:crypto'

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { readDesktopFileText } from '@/lib/desktop-fs'

import { SANDBOXED_HTML_CSP, SANDBOXED_HTML_PERMISSIONS } from './sandboxed-html-approval'
import { SandboxedHtmlDocument, SandboxedHtmlPreview } from './sandboxed-html-preview'

vi.mock('@/lib/desktop-fs', () => ({
  desktopFsCacheKey: vi.fn(() => 'remote:work:https://gateway'),
  readDesktopFileText: vi.fn()
}))

const target = {
  kind: 'file' as const,
  label: 'view.html',
  path: '/workspace/view.html',
  previewKind: 'html' as const,
  renderMode: 'preview' as const,
  source: '/workspace/view.html',
  url: 'file:///workspace/view.html'
}

const authoredSource = '<!doctype html><title>Agent view</title><p id="authored-marker">Hello</p>'

function readResult(text = authoredSource) {
  return {
    byteSize: new TextEncoder().encode(text).byteLength,
    path: '/canonical/workspace/view.html',
    text
  }
}

describe('SandboxedHtmlPreview', () => {
  const registerFrame = vi.fn(async () => true)
  const unregisterFrame = vi.fn(async () => true)

  beforeEach(() => {
    Object.defineProperty(globalThis, 'crypto', { configurable: true, value: webcrypto })
    window.localStorage.clear()
    vi.mocked(readDesktopFileText).mockResolvedValue(readResult())
    window.hermesDesktop = { sandboxedHtml: { registerFrame, unregisterFrame } } as never
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('constructs no iframe before exact-content approval', async () => {
    render(<SandboxedHtmlPreview reloadKey={0} target={target} />)

    expect(await screen.findByText('Run sandboxed preview')).not.toBeNull()
    expect(globalThis.document.querySelector('iframe')).toBeNull()
    expect(screen.getByText('/canonical/workspace/view.html')).not.toBeNull()
    expect(screen.getByText(/^sha256:[a-f0-9]{12}$/)).not.toBeNull()
  })

  it('registers an inert frame before loading authored HTML', async () => {
    render(<SandboxedHtmlPreview reloadKey={0} target={target} />)
    fireEvent.click(await screen.findByText('Run sandboxed preview'))

    const iframe = globalThis.document.querySelector('iframe')

    expect(iframe).not.toBeNull()
    expect(iframe?.getAttribute('srcdoc')).not.toContain('authored-marker')
    expect(iframe?.getAttribute('sandbox')).toBe('allow-scripts')
    expect(iframe?.getAttribute('referrerpolicy')).toBe('no-referrer')
    expect(iframe?.getAttribute('allow')).toBe(SANDBOXED_HTML_PERMISSIONS)

    fireEvent.load(iframe as HTMLIFrameElement)

    await waitFor(() => expect(registerFrame).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(iframe?.getAttribute('srcdoc')).toContain('authored-marker'))
    expect(iframe?.getAttribute('srcdoc')).toContain(SANDBOXED_HTML_CSP)
  })

  it('fails closed when the Electron frame guard cannot be installed', async () => {
    registerFrame.mockResolvedValueOnce(false)
    render(<SandboxedHtmlPreview reloadKey={0} target={target} />)
    fireEvent.click(await screen.findByText('Run sandboxed preview'))

    const iframe = globalThis.document.querySelector('iframe') as HTMLIFrameElement
    fireEvent.load(iframe)

    expect(await screen.findByText('Preview isolation unavailable')).not.toBeNull()
    expect(globalThis.document.querySelector('iframe')).toBeNull()
    expect(globalThis.document.body.textContent).not.toContain('authored-marker')
  })

  it('persists approval only for the same digest and revokes it after a change', async () => {
    render(<SandboxedHtmlPreview key="0" reloadKey={0} target={target} />)
    fireEvent.click(await screen.findByText('Run sandboxed preview'))
    expect(globalThis.document.querySelector('iframe')).not.toBeNull()

    cleanup()
    render(<SandboxedHtmlPreview key="same" reloadKey={0} target={target} />)
    await waitFor(() => expect(globalThis.document.querySelector('iframe')).not.toBeNull())
    expect(screen.queryByText('Run sandboxed preview')).toBeNull()

    vi.mocked(readDesktopFileText).mockResolvedValueOnce(readResult(`${authoredSource}<p>changed</p>`))
    cleanup()
    render(<SandboxedHtmlPreview key="changed" reloadKey={1} target={target} />)

    expect(await screen.findByText('Run sandboxed preview')).not.toBeNull()
    expect(globalThis.document.querySelector('iframe')).toBeNull()
  })

  it('unmounts an approved frame synchronously when its digest changes', async () => {
    const view = render(
      <SandboxedHtmlDocument
        digest={'a'.repeat(64)}
        documentSource="<p>first</p>"
        frameTitle="Generated view"
        identity="generated:demo"
        path="/views/demo/view.json"
        source="<p>first</p>"
      />
    )

    fireEvent.click(screen.getByText('Run sandboxed preview'))
    expect(globalThis.document.querySelector('iframe')).not.toBeNull()

    view.rerender(
      <SandboxedHtmlDocument
        digest={'b'.repeat(64)}
        documentSource="<p>changed</p>"
        frameTitle="Generated view"
        identity="generated:demo"
        path="/views/demo/view.json"
        source="<p>changed</p>"
      />
    )

    expect(globalThis.document.querySelector('iframe')).toBeNull()
    expect(screen.getByText('Run sandboxed preview')).not.toBeNull()
  })

  it('forwards bridge messages only from the exact iframe window', async () => {
    const bridge = { handleMessage: vi.fn() }
    render(
      <SandboxedHtmlDocument
        bridge={bridge}
        digest={'c'.repeat(64)}
        documentSource="<p>bridge</p>"
        frameTitle="Generated view"
        identity="generated:bridge"
        path="/views/bridge/view.json"
        source="<p>bridge</p>"
      />
    )
    fireEvent.click(screen.getByText('Run sandboxed preview'))
    const iframe = globalThis.document.querySelector('iframe') as HTMLIFrameElement
    fireEvent.load(iframe)
    await waitFor(() => expect(registerFrame).toHaveBeenCalled())
    await waitFor(() => expect(iframe.getAttribute('srcdoc')).toContain('bridge'))

    window.dispatchEvent(new MessageEvent('message', { data: { v: 1 }, source: window }))
    expect(bridge.handleMessage).not.toHaveBeenCalled()

    window.dispatchEvent(new MessageEvent('message', { data: { v: 1 }, source: iframe.contentWindow }))
    expect(bridge.handleMessage).toHaveBeenCalledWith({ v: 1 }, expect.any(Function))
  })

  it.each([
    ['binary', { ...readResult(), binary: true }],
    ['truncated', { ...readResult(), truncated: true }],
    ['malformed', { ...readResult(), path: '' }],
    ['oversized', { ...readResult(), byteSize: 512 * 1024 + 1 }]
  ])('rejects %s input without constructing an iframe', async (_label, result) => {
    vi.mocked(readDesktopFileText).mockResolvedValueOnce(result)
    render(<SandboxedHtmlPreview reloadKey={0} target={target} />)

    expect(await screen.findByText('Sandboxed preview unavailable')).not.toBeNull()
    expect(globalThis.document.querySelector('iframe')).toBeNull()
  })
})

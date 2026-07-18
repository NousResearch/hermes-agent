import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import DrawioRenderer from './drawio-embed'

const SAMPLE_XML = `<mxGraphModel><root><mxCell id="0" /><mxCell id="1" parent="0" /></root></mxGraphModel>`

function postMessageFromIframe(iframe: HTMLIFrameElement, data: unknown) {
  const event = new MessageEvent('message', {
    source: iframe.contentWindow,
    origin: window.location.origin,
    data,
  })

  window.dispatchEvent(event)
}

describe('DrawioRenderer', () => {
  it('renders source preview while streaming', () => {
    render(<DrawioRenderer code={SAMPLE_XML} streaming />)
    expect(screen.getByText(/mxGraphModel/)).toBeTruthy()
  })

  it('creates an iframe and sends the load message after the receiver is ready', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true } as Response)

    render(<DrawioRenderer code={SAMPLE_XML} />)

    await waitFor(() => {
      const iframe = window.document.querySelector('iframe')
      expect(iframe).toBeTruthy()
      expect(iframe?.src).toContain('/drawio/render.html')
    })

    const iframe = window.document.querySelector('iframe')!
    const postMessageSpy = vi.spyOn(iframe.contentWindow!, 'postMessage')

    postMessageFromIframe(iframe, { event: 'ready' })

    await waitFor(() => {
      expect(postMessageSpy).toHaveBeenCalledWith(
        { action: 'load', xml: SAMPLE_XML, mode: 'inline' },
        '*'
      )
    })
  })

  it('falls back to source preview when the render asset is unavailable', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: false, status: 404 } as Response)

    render(<DrawioRenderer code={SAMPLE_XML} />)

    await waitFor(() => {
      expect(screen.getByText(/mxGraphModel/)).toBeTruthy()
    })

    expect(window.document.querySelector('iframe')).toBeNull()
  })

  it('marks failure when the receiver reports an error', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true } as Response)

    render(<DrawioRenderer code={SAMPLE_XML} />)

    await waitFor(() => {
      expect(window.document.querySelector('iframe')).toBeTruthy()
    })

    const iframe = window.document.querySelector('iframe')!
    postMessageFromIframe(iframe, { event: 'error', error: 'boom' })

    await waitFor(() => {
      expect(screen.getByText(/mxGraphModel/)).toBeTruthy()
      expect(window.document.querySelector('iframe')).toBeNull()
    })
  })
})

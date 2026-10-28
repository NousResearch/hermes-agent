import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $selectedStoredSessionId } from '@/store/session'
import { deferred } from '@/test/deferred'

import { PreviewPane } from './preview-pane'

vi.mock('./real-profile-consent-dialog', () => ({ RealProfileConsentDialog: () => null }))

const originalDesktop = window.hermesDesktop

afterEach(() => {
  cleanup()
  window.hermesDesktop = originalDesktop
  $selectedStoredSessionId.set(null)
})

async function setup() {
  const pick = deferred<unknown>()
  const captured = deferred<string>()
  const leases = new Set<number>()
  let token = 0
  let waits = 0
  const capture = vi.fn(() => captured.promise)
  const rendered = render(
    <PreviewPane
      target={{ kind: 'url', label: 'Preview', source: 'https://example.com', url: 'https://example.com' }}
    />
  )
  const webview = rendered.container.querySelector('webview')!
  Object.assign(webview, {
    getWebContentsId: () => 42,
    executeJavaScript: async (code: string) => {
      if (code.includes('.wait()')) {
        waits++

        return waits === 1 ? pick.promise : new Promise(() => {})
      }

      return undefined
    }
  })
  window.hermesDesktop = {
    ...originalDesktop,
    beginPreviewGuestActivity: async () => {
      leases.add(++token)

      return token
    },
    endPreviewGuestActivity: async held => {
      leases.delete(held)
    },
    executePreviewGuestScript: async () => undefined,
    capturePreview: capture
  }
  fireEvent.click(rendered.getByRole('button', { name: 'Annotate' }))
  await waitFor(() => expect(waits).toBe(1))

  return { captured, capture, leases, pick, rendered, webview }
}

const area = { rect: { height: 20, width: 40, x: 10, y: 20 }, type: 'pick-area' }

describe('annotation document lifetime', () => {
  it.each(['cancelled-load', 'in-page'])('keeps picking usable after %s without replacing the document', async kind => {
    const fixture = await setup()
    await act(async () => {
      fixture.webview.dispatchEvent(
        Object.assign(new Event('did-start-navigation'), { isMainFrame: true, isInPlace: false })
      )
      fixture.webview.dispatchEvent(new Event(kind === 'in-page' ? 'did-navigate-in-page' : 'did-stop-loading'))
      fixture.pick.resolve(area)
    })
    await waitFor(() => expect(fixture.capture).toHaveBeenCalledOnce())
    await act(async () => fixture.captured.resolve('data:image/png;base64,AA=='))
    expect(await fixture.rendered.findByRole('form', { name: 'Comment 1' })).toBeTruthy()
    expect(fixture.leases.size).toBe(0)
  })

  it.each(['document-commit', 'conversation-change'])(
    'releases a pending crop and rejects its late draft on %s',
    async kind => {
      const fixture = await setup()
      await act(async () => fixture.pick.resolve(area))
      await waitFor(() => expect(fixture.capture).toHaveBeenCalledOnce())
      expect(fixture.leases.size).toBe(1)
      await act(async () => {
        if (kind === 'document-commit') {
          fixture.webview.dispatchEvent(Object.assign(new Event('did-navigate'), { url: 'https://example.com/next' }))
        } else {
          $selectedStoredSessionId.set('another-conversation')
        }
      })
      expect(fixture.leases.size).toBe(0)
      await act(async () => fixture.captured.resolve('data:image/png;base64,AA=='))
      expect(fixture.rendered.queryByRole('form', { name: 'Comment 1' })).toBeNull()
      expect(fixture.rendered.getByRole('button', { name: 'Annotate' })).toBeTruthy()
    }
  )
})

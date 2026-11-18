import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { usePreviewTextComments } from './use-preview-text-comments'

const submit = vi.hoisted(() => vi.fn(() => true))
vi.mock('@/app/chat/composer/focus', () => ({
  getActiveComposer: () => 'main',
  getVisibleComposerSurfaceId: () => 'surface-a',
  requestComposerSubmit: submit
}))
vi.mock('@/lib/preview-annotate/handoff', () => ({ capturePreviewAnnotateDestination: () => ({ kind: 'composer' }) }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn() }))

const ownerScope = vi.hoisted(() => vi.fn())
vi.mock('@/store/session', () => ({ knownSessionOwner: ownerScope, ownerLookupSessionRows: () => [] }))
beforeEach(() => { submit.mockReturnValue(true); ownerScope.mockReturnValue(undefined) })

const url = 'http://localhost:5173/'

function setup(annotateMode = false) {
  const webview = Object.assign(window.document.createElement('div'), { send: vi.fn(), getURL: () => url })
  const ref = { current: webview }

  const hook = renderHook(
    ({ session, annotate }) => usePreviewTextComments({ available: true, annotateMode: annotate, webviewRef: ref, url, sessionKey: session }),
    { initialProps: { session: 'chat-a', annotate: annotateMode } }
  )

  const select = () =>
    act(() =>
      webview.dispatchEvent(
        Object.assign(new Event('ipc-message'), {
          channel: 'preview-text-selection',
          args: [{ viewportWidth: 800, text: '  selected\ntext  ', url, rect: { x: 5, y: 6, width: 30, height: 20 } }]
        })
      )
    )

  return { ...hook, webview, select }
}

describe('preview text comment lifecycle', () => {
  it('is opt-in and submits exact context to the captured composer once', () => {
    submit.mockClear()
    const h = setup()
    h.select()
    expect(h.result.current.draft).toBeNull()
    act(() => h.result.current.toggle())
    h.select()
    expect(h.result.current.draft?.selection.text).toBe('  selected\ntext  ')
    act(() => expect(h.result.current.submit('rewrite this')).toBe(true))
    expect(submit).toHaveBeenCalledWith(expect.stringContaining('rewrite this'), {
      target: 'main',
      surfaceId: 'surface-a',
      requireAccepted: true
    })
    expect(h.result.current.draft).toBeNull()
    act(() => expect(h.result.current.submit('again')).toBe(false))
    expect(submit).toHaveBeenCalledTimes(1)
  })

  it('clears pending comments on document commit or conversation change', () => {
    const h = setup()
    act(() => h.result.current.toggle())
    h.select()
    act(() => h.webview.dispatchEvent(Object.assign(new Event('did-navigate'), { isMainFrame: true })))
    expect(h.result.current.draft).toBeNull()
    h.select()
    h.rerender({ session: 'chat-b', annotate: false })
    expect(h.result.current.draft).toBeNull()
    expect(h.result.current.enabled).toBe(false)
  })
})


it('switches modes in one gesture and scales the selection across guest zoom', () => {
  const h = setup(true)
  Object.defineProperty(h.webview, 'clientWidth', { value: 1200 })
  act(() => {
    h.result.current.toggle()
    h.rerender({ session: 'chat-a', annotate: false })
  })
  expect(h.result.current.enabled).toBe(true)
  h.select()
  expect(h.result.current.draft?.selection.rect).toEqual({ x: 7.5, y: 9, width: 45, height: 30 })
})

it('retains rejected comments and binds dictation to the backend target profile', () => {
  ownerScope.mockReturnValue({ connectionId: 'remote', profile: 'desktop-alias', targetProfile: 'backend-profile' })
  const h = setup()
  act(() => h.result.current.toggle())
  h.select()
  expect(h.result.current.draft?.owner).toEqual({ connectionId: 'remote', profile: 'backend-profile' })
  submit.mockReturnValue(false)
  act(() => expect(h.result.current.submit('rewrite')).toBe(false))
  expect(h.result.current.draft).not.toBeNull()
  submit.mockClear()
  ownerScope.mockReturnValue({ connectionId: 'other', profile: 'other' })
  act(() => expect(h.result.current.submit('rewrite')).toBe(false))
  expect(submit).not.toHaveBeenCalled()
})

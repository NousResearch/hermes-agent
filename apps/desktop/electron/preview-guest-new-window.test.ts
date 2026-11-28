import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import {
  createPopupLimiter,
  installPreviewGuestNewWindow,
  PREVIEW_GUEST_NEW_WINDOW_CHANNEL,
  type PreviewGuestNewWindow,
  previewGuestNewWindowRequest
} from './preview-guest-new-window'

const bytes = (text: string) => new TextEncoder().encode(text)
const text = (data: Uint8Array) => new TextDecoder().decode(data)
const open = () => createPopupLimiter(100, 1000)

describe('previewGuestNewWindowRequest', () => {
  test('a _blank link becomes a foreground preview tab request', () => {
    const out = previewGuestNewWindowRequest(7, { disposition: 'foreground-tab', url: 'http://app.example.test/page2.html' }, open(), 0)

    assert.deepEqual(out, { request: { background: false, guestId: 7, url: 'http://app.example.test/page2.html' } })
  })

  test('a ctrl/middle-click opens behind the current tab', () => {
    const out = previewGuestNewWindowRequest(7, { disposition: 'background-tab', url: 'https://example.com/' }, open(), 0)

    assert.ok('request' in out && out.request.background)
  })

  test('a _blank form POST carries its body and content type', () => {
    const out = previewGuestNewWindowRequest(
      7,
      {
        disposition: 'foreground-tab',
        postBody: {
          contentType: 'application/x-www-form-urlencoded',
          data: [
            { bytes: bytes('user=demo'), type: 'rawData' },
            { bytes: bytes('&x=1'), type: 'rawData' }
          ]
        },
        url: 'https://panel.example.test/instances/1/login-as/2'
      },
      open(),
      0
    )

    assert.ok('request' in out)
    assert.equal(out.request.post?.contentType, 'application/x-www-form-urlencoded')
    assert.equal(text(out.request.post!.body), 'user=demo&x=1')
  })

  test('a multipart POST keeps its boundary in the header', () => {
    const out = previewGuestNewWindowRequest(
      7,
      {
        postBody: { boundary: 'XyZ', contentType: 'multipart/form-data', data: [{ bytes: bytes('--XyZ--'), type: 'rawData' }] },
        url: 'https://example.com/form'
      },
      open(),
      0
    )

    assert.ok('request' in out)
    assert.equal(out.request.post?.contentType, 'multipart/form-data; boundary=XyZ')
  })

  test('a POST with a file upload part is dropped, not half-replayed', () => {
    const out = previewGuestNewWindowRequest(
      7,
      { postBody: { contentType: 'multipart/form-data', data: [{ filePath: '/tmp/upload.txt', type: 'file' }] }, url: 'https://example.com/up' },
      open(),
      0
    )

    assert.deepEqual(out, { drop: 'file-upload' })
  })

  test('only http and https become preview tabs', () => {
    for (const url of ['file:///etc/hosts', 'javascript:alert(1)', 'mailto:someone@example.test', 'hermes://x', 'not a url']) {
      assert.deepEqual(previewGuestNewWindowRequest(7, { url }, open(), 0), { drop: 'scheme' }, url)
    }
  })

  test('a page cannot flood the strip: the limiter is per guest and per window', () => {
    const allow = createPopupLimiter(2, 1000)
    const req = (guest: number, at: number) => 'request' in previewGuestNewWindowRequest(guest, { url: 'https://a.test/' }, allow, at)

    assert.equal(req(1, 0), true)
    assert.equal(req(1, 10), true)
    assert.equal(req(1, 20), false)
    assert.equal(req(2, 20), true, 'another guest has its own budget')
    assert.equal(req(1, 1500), true, 'the window slides')
  })
})

describe('installPreviewGuestNewWindow', () => {
  function rig(hostDestroyed = false) {
    const sent: Array<{ channel: string; payload: PreviewGuestNewWindow }> = []
    const logs: string[] = []
    let handler: ((details: { url: string }) => { action: 'deny' }) | undefined

    installPreviewGuestNewWindow(
      { id: 42, isDestroyed: () => false, setWindowOpenHandler: h => (handler = h) },
      { isDestroyed: () => hostDestroyed, send: (channel, payload) => sent.push({ channel, payload }) },
      open(),
      line => logs.push(line),
      () => 0
    )

    return { call: (url: string) => handler!({ url }), logs, sent }
  }

  test('always denies the real popup and forwards an admitted request to the host', () => {
    const { call, sent } = rig()

    assert.deepEqual(call('https://example.com/'), { action: 'deny' })
    assert.deepEqual(sent, [
      { channel: PREVIEW_GUEST_NEW_WINDOW_CHANNEL, payload: { background: false, guestId: 42, url: 'https://example.com/' } }
    ])
  })

  test('a refused request is denied, logged and never forwarded', () => {
    const { call, logs, sent } = rig()

    assert.deepEqual(call('file:///etc/passwd'), { action: 'deny' })
    assert.equal(sent.length, 0)
    assert.match(logs[0], /dropped \(scheme\)/)
  })

  test('nothing is sent to a destroyed host', () => {
    const { call, sent } = rig(true)

    assert.deepEqual(call('https://example.com/'), { action: 'deny' })
    assert.equal(sent.length, 0)
  })
})

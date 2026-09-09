/**
 * Tests for electron/embed-referer.ts — the YouTube Referer stamp for chat
 * embeds. Chat embeds are plain iframes in the window's default session and
 * the production renderer loads from a file:// URL (no Referer is sent), so
 * YouTube's embed player rejects the config request with error 153 unless the
 * session listener stamps one in. The stamp must also survive composition with
 * the default-session listener that applies remote gateway headers: Electron
 * keeps a single listener per webRequest hook, so both behaviors live in one.
 *
 * Run with: vitest run --project electron embed-referer
 */

import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

type PartitionListener = (
  details: { url: string; requestHeaders?: Record<string, string> },
  callback: (result: { requestHeaders?: Record<string, string> }) => void
) => void

const partitionListeners = new Map<string, PartitionListener>()

vi.mock('electron', () => ({
  session: {
    fromPartition: (name: string) => ({
      webRequest: {
        onBeforeSendHeaders: (listener: PartitionListener) => {
          partitionListeners.set(name, listener)
        }
      }
    })
  }
}))

const { installEmbedReferer, stampEmbedRefererHeaders, withEmbedRefererStamp } = await import(
  './embed-referer'
)

function partitionListener(name: string): PartitionListener {
  const listener = partitionListeners.get(name)

  assert.ok(listener, `onBeforeSendHeaders listener installed for ${name}`)

  return listener!
}

const YOUTUBE_HOSTS = [
  'https://www.youtube.com/watch?v=9MVqxnIb98Y',
  'https://www.youtube-nocookie.com/embed/9MVqxnIb98Y?modestbranding=1&rel=0',
  'https://rr3---sn-npoe7ney.googlevideo.com/videoplayback?Id=o-AH',
  'https://i.ytimg.com/vi/9MVqxnIb98Y/hqdefault.jpg',
  'https://www.youtubei.googleapis.com/youtubei/v1/player?key=AIza'
]

test('stampEmbedRefererHeaders adds the YouTube referer for every YouTube host', () => {
  for (const url of YOUTUBE_HOSTS) {
    assert.equal(
      stampEmbedRefererHeaders(url, { Accept: '*/*' }).Referer,
      'https://www.youtube.com/',
      `expected stamp for ${url}`
    )
  }
})

test('stampEmbedRefererHeaders keeps unrelated headers as they were', () => {
  const stamped = stampEmbedRefererHeaders(YOUTUBE_HOSTS[0], { Accept: '*/*', 'X-Test': 'a' })

  assert.equal(stamped.Accept, '*/*')
  assert.equal(stamped['X-Test'], 'a')
})

test('stampEmbedRefererHeaders never overrides an existing Referer', () => {
  const stamped = stampEmbedRefererHeaders(YOUTUBE_HOSTS[0], { Referer: 'https://example.com/' })

  assert.equal(stamped.Referer, 'https://example.com/')
  assert.equal('referer' in stamped, false)
})

test('stampEmbedRefererHeaders honors a lowercase referer instead of adding one', () => {
  const stamped = stampEmbedRefererHeaders(YOUTUBE_HOSTS[0], { referer: 'https://example.com/' })

  assert.equal(stamped.referer, 'https://example.com/')
  assert.equal('Referer' in stamped, false)
})

test('stampEmbedRefererHeaders leaves non-YouTube hosts completely untouched', () => {
  const original = { Accept: '*/*' }

  assert.equal(stampEmbedRefererHeaders('https://example.com/video', original), original)

  // Substring matches must not fool the host test either.
  assert.equal(
    stampEmbedRefererHeaders('https://notyoutube.com/embed/x', original),
    original
  )
  assert.equal(
    stampEmbedRefererHeaders('https://youtube.com.evil.io/embed/x', original),
    original
  )
})

test('stampEmbedRefererHeaders tolerates unparseable URLs', () => {
  const original = { Accept: '*/*' }

  assert.equal(stampEmbedRefererHeaders('not a url', original), original)
})

test('installEmbedRefererForSession stamps through the session listener', () => {
  installEmbedReferer()

  const listener = partitionListener('persist:hermes-embed')

  let result: { requestHeaders?: Record<string, string> } = {}
  listener({ url: YOUTUBE_HOSTS[1], requestHeaders: { Accept: '*/*' } }, (r) => {
    result = r
  })

  assert.equal(result.requestHeaders?.Referer, 'https://www.youtube.com/')
})

test('withEmbedRefererStamp applies the stamp after the wrapped listener', () => {
  const calls: string[] = []

  const wrapped = withEmbedRefererStamp((details, callback) => {
    calls.push(details.url)

    // The remote-headers listener reports no changes for non-gateway URLs.
    callback({})
  })

  let result: { requestHeaders?: Record<string, string> } = {}
  wrapped({ url: YOUTUBE_HOSTS[1], requestHeaders: { Accept: '*/*' } }, (r) => {
    result = r
  })

  assert.equal(calls.length, 1)
  assert.equal(result.requestHeaders?.Accept, '*/*')
  assert.equal(result.requestHeaders?.Referer, 'https://www.youtube.com/')
})

test('withEmbedRefererStamp preserves headers merged by the wrapped listener', () => {
  const wrapped = withEmbedRefererStamp((details, callback) => {
    callback({ requestHeaders: { ...details.requestHeaders, Authorization: 'Bearer t' } })
  })

  let result: { requestHeaders?: Record<string, string> } = {}
  wrapped({ url: YOUTUBE_HOSTS[1], requestHeaders: { Accept: '*/*' } }, (r) => {
    result = r
  })

  assert.equal(result.requestHeaders?.Authorization, 'Bearer t')
  assert.equal(result.requestHeaders?.Referer, 'https://www.youtube.com/')
})

test('withEmbedRefererStamp leaves non-YouTube requests to the wrapped listener alone', () => {
  const wrapped = withEmbedRefererStamp((_details, callback) => {
    callback({})
  })

  let result: { requestHeaders?: Record<string, string> } | undefined
  wrapped({ url: 'https://gateway.internal/api', requestHeaders: { Accept: '*/*' } }, (r) => {
    result = r
  })

  // No requestHeaders in the result means "no changes" — the wrapper must not
  // turn that into an empty header replacement or add anything itself.
  assert.deepEqual(result, {})
})

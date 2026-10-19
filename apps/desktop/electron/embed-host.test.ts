import assert from 'node:assert/strict'
import http from 'node:http'

import { test } from 'vitest'

import { startEmbedHost, youtubeWrapperFor } from './embed-host'

const ORIGIN = 'http://127.0.0.1:5555'

function get(origin: string, path: string, host = new URL(origin).host) {
  return new Promise<{ body: string; headers: http.IncomingHttpHeaders; status: number }>((resolve, reject) => {
    const request = http.get(`${origin}${path}`, { headers: { host } }, response => {
      let body = ''
      response.on('data', chunk => (body += chunk))
      response.on('end', () => resolve({ body, headers: response.headers, status: response.statusCode ?? 0 }))
    })

    request.on('error', reject)
  })
}

test('wraps the player with an origin YouTube can match', () => {
  const html = youtubeWrapperFor('/youtube/M7lc1UVf-VE?start=42&rel=0&modestbranding=1', ORIGIN)!
  const src = new URL(/src="([^"]+)"/.exec(html)![1].replaceAll('&amp;', '&'))

  assert.equal(src.origin, 'https://www.youtube-nocookie.com')
  assert.equal(src.pathname, '/embed/M7lc1UVf-VE')
  assert.equal(src.searchParams.get('origin'), ORIGIN)
  assert.equal(src.searchParams.get('start'), '42')
  assert.equal(src.searchParams.get('rel'), '0')
  assert.match(html, /allowfullscreen/)
})

test('serves nothing but a well-formed video id', () => {
  for (const path of [
    '/',
    '/youtube/',
    '/youtube/short',
    '/youtube/M7lc1UVf-VE/x',
    '/youtube/"><script>x',
    '/other',
    '//',
    '//['
  ]) {
    assert.equal(youtubeWrapperFor(path, ORIGIN), null, path)
  }
})

test('drops unknown and non-numeric params instead of forwarding them', () => {
  const html = youtubeWrapperFor('/youtube/M7lc1UVf-VE?start=1"x&autoplay=1&origin=https://evil.test', ORIGIN)!
  const src = new URL(/src="([^"]+)"/.exec(html)![1].replaceAll('&amp;', '&'))

  assert.equal(src.searchParams.get('start'), null)
  assert.equal(src.searchParams.get('autoplay'), null)
  assert.equal(src.searchParams.get('origin'), ORIGIN)
})

test('the live host serves the wrapper locked down, and only to its own Host', async t => {
  const host = await startEmbedHost()
  t.onTestFinished(() => host.close())

  assert.match(host.origin, /^http:\/\/127\.0\.0\.1:\d+$/)

  const ok = await get(host.origin, '/youtube/M7lc1UVf-VE')
  assert.equal(ok.status, 200)
  assert.match(
    String(ok.headers['content-security-policy']),
    /default-src 'none'; frame-src https:\/\/www\.youtube-nocookie\.com/
  )
  assert.match(ok.body, new RegExp(`origin=${encodeURIComponent(host.origin)}`))

  assert.equal((await get(host.origin, '/youtube/nope')).status, 404)
  assert.equal((await get(host.origin, '/youtube/M7lc1UVf-VE', 'evil.test')).status, 404)
  assert.equal((await get(host.origin, '//[')).status, 404)
})

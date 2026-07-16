import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import fs from 'node:fs/promises'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { test, vi } from 'vitest'

import {
  createLinkTitleCurlFetcher,
  linkTitleCurlRequestArgs,
  parseLinkTitleCurlHeaders
} from './link-title-curl'

const execFileAsync = promisify(execFile)

function admitPublicUrl(value: string): null | string {
  const url = new URL(value)

  if (url.hostname === '127.0.0.1') {
    return null
  }

  return url.protocol === 'http:' || url.protocol === 'https:' ? url.href : null
}

test('curl request arguments never enable automatic redirect following', () => {
  const args = linkTitleCurlRequestArgs('https://example.com/', {
    connectTimeoutSeconds: 4,
    headerPath: 'C:\\Temp\\hermes-link-title.headers',
    timeoutSeconds: 5,
    userAgent: 'Hermes test'
  })

  assert.equal(args[0], '--disable')
  assert.equal(args.includes('--no-location'), true)
  assert.equal(args.includes('--location'), false)
  assert.equal(args.includes('--max-redirs'), false)
  assert.deepEqual(args.slice(args.indexOf('--dump-header'), args.indexOf('--dump-header') + 2), [
    '--dump-header',
    'C:\\Temp\\hermes-link-title.headers'
  ])
})

test('curl ignores a default config that enables automatic redirects', async () => {
  const curlHome = await fs.mkdtemp(path.join(os.tmpdir(), 'hermes-link-title-curlrc-'))
  const headerPath = path.join(curlHome, 'headers')
  let redirectedRequests = 0

  const server = http.createServer((request, response) => {
    if (request.url === '/start') {
      response.writeHead(302, { Location: '/redirected' })
      response.end()

      return
    }

    redirectedRequests += 1
    response.end('redirected')
  })

  try {
    await Promise.all([
      fs.writeFile(path.join(curlHome, '.curlrc'), 'location\n'),
      fs.writeFile(path.join(curlHome, '_curlrc'), 'location\n')
    ])
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject)
      server.listen(0, '127.0.0.1', resolve)
    })
    const address = server.address()

    assert.ok(address && typeof address !== 'string')

    const args = linkTitleCurlRequestArgs(`http://127.0.0.1:${address.port}/start`, {
      connectTimeoutSeconds: 2,
      headerPath,
      timeoutSeconds: 2,
      userAgent: 'Hermes test'
    })

    await execFileAsync('curl', args, {
      encoding: 'utf8',
      env: { ...process.env, CURL_HOME: curlHome },
      timeout: 5_000
    })
    assert.equal(redirectedRequests, 0)
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
    await fs.rm(curlHome, { force: true, recursive: true })
  }
})

test('public redirects to loopback stop before a second request', async () => {
  const request = vi.fn().mockResolvedValue({ body: '', location: 'http://127.0.0.1/private', statusCode: 302 })

  const fetchTitle = createLinkTitleCurlFetcher({
    admitUrl: admitPublicUrl,
    maxRedirects: 3,
    now: () => 0,
    readTitle: vi.fn(() => ''),
    request,
    timeoutMs: 5_000
  })

  assert.equal(await fetchTitle('https://example.com/start'), '')
  assert.deepEqual(request.mock.calls, [['https://example.com/start', 5_000]])
})

test('public redirects resolve with WHATWG URL semantics and retain title fetching', async () => {
  const request = vi
    .fn()
    .mockResolvedValueOnce({ body: '', location: '../next', statusCode: 302 })
    .mockResolvedValueOnce({ body: '<title>Public destination</title>', location: null, statusCode: 200 })

  const readTitle = vi.fn(() => 'Public destination')

  const fetchTitle = createLinkTitleCurlFetcher({
    admitUrl: admitPublicUrl,
    maxRedirects: 3,
    now: () => 0,
    readTitle,
    request,
    timeoutMs: 5_000
  })

  assert.equal(await fetchTitle('https://example.com/path/start'), 'Public destination')
  assert.deepEqual(request.mock.calls, [
    ['https://example.com/path/start', 5_000],
    ['https://example.com/next', 5_000]
  ])
  assert.deepEqual(readTitle.mock.calls, [['<title>Public destination</title>']])
})

test('curl header parsing retains the raw Location header for WHATWG resolution', () => {
  const parsed = parseLinkTitleCurlHeaders(
    'HTTP/1.1 200 Connection established\r\n\r\nHTTP/2 302\r\nlocation: ../next\r\ncontent-length: 0\r\n\r\n'
  )

  assert.deepEqual(parsed, { location: '../next', statusCode: 302 })
})

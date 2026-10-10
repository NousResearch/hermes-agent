import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { EventEmitter } from 'node:events'
import http from 'node:http'

import { afterEach, describe, test } from 'vitest'

import { curlTitleTargetArgs } from './link-title-curl'
import {
  literalTitleBlockReason,
  resolveTitleFetchTarget,
  type TitleFetchTarget,
  titleProxyFor
} from './link-title-guard'
import { type MetadataRequestFn, metadataRequestOnce } from './metadata-http-hop'
import { fetchWithSafeRedirects } from './safe-http-redirects'

// Automatic link metadata (titles and favicons) must dial only what the
// destination guard vetted: every redirect hop is re-admitted, and the curl
// tier connects to the checked DNS answer instead of resolving the name again.

const PROXY_VARS = ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy'] as const
const servers = new Set<http.Server>()

afterEach(async () => {
  for (const name of PROXY_VARS) {
    delete process.env[name]
  }

  await Promise.all([...servers].map(server => new Promise<void>(resolve => server.close(() => resolve()))))
  servers.clear()
})

const answers = (map: Record<string, string[]>) => async (host: string) =>
  (map[host] ?? []).map(address => ({ address, family: address.includes(':') ? 6 : 4 }))

async function listen(handler: http.RequestListener): Promise<number> {
  const server = http.createServer(handler)
  servers.add(server)
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')

  return address.port
}

describe('destination classes beyond the private ranges', () => {
  test('refuses multicast, the IETF protocol block, .internal and single-label names without DNS', () => {
    for (const host of ['224.0.0.1', '239.255.255.250', 'ff02::1', '192.0.0.8', 'db.ec2.internal', 'intranet']) {
      assert.ok(literalTitleBlockReason(host), host)
    }

    assert.equal(literalTitleBlockReason('LOCALHOST..'), 'loopback-host')
    assert.equal(literalTitleBlockReason('fd00:ec2::254'), 'metadata')
    assert.equal(literalTitleBlockReason('2606:4700::1111'), null)
  })
})

describe('resolveTitleFetchTarget', () => {
  test('returns every vetted answer with the port the hop will dial', async () => {
    const target = await resolveTitleFetchTarget('https://example.test:8443/a', {
      lookup: answers({ 'example.test': ['93.184.216.34', '2606:2800:220:1::1'] })
    })

    assert.deepEqual(target, {
      addresses: ['93.184.216.34', '2606:2800:220:1::1'],
      hostname: 'example.test',
      port: '8443',
      proxy: ''
    })
  })

  test('a public literal needs no lookup and nothing to pin', async () => {
    let looked = false

    const target = await resolveTitleFetchTarget('http://8.8.8.8/', {
      lookup: async () => {
        looked = true

        return []
      }
    })

    assert.deepEqual(target, { addresses: [], hostname: '8.8.8.8', port: '80', proxy: '' })
    assert.equal(looked, false)
  })

  test('fails closed on any private answer and on a lookup that never returns', async () => {
    assert.equal(
      await resolveTitleFetchTarget('https://mixed.test/', {
        lookup: answers({ 'mixed.test': ['93.184.216.34', '169.254.169.254'] })
      }),
      null
    )
    assert.equal(
      await resolveTitleFetchTarget('https://slow.test/', { dnsTimeoutMs: 20, lookup: () => new Promise(() => {}) }),
      null
    )
  })

  test('vets DNS answers for a proxied dial too, and hands the proxy only to its scheme', async () => {
    process.env.HTTPS_PROXY = 'http://proxy.corp:3128'
    let lookups = 0
    let answer = '93.184.216.34'

    const lookup = async () => {
      lookups += 1

      return [{ address: answer, family: 4 }]
    }

    assert.deepEqual(await resolveTitleFetchTarget('https://example.test/', { lookup }), {
      addresses: ['93.184.216.34'],
      hostname: 'example.test',
      port: '443',
      proxy: 'http://proxy.corp:3128'
    })
    assert.equal(lookups, 1)

    // http:// has no proxy here, so curl dials it directly: resolve and pin.
    assert.deepEqual((await resolveTitleFetchTarget('http://example.test/', { lookup }))?.addresses, ['93.184.216.34'])
    assert.equal(lookups, 2)

    // Chromium-stack callers never route through the env proxy.
    assert.equal((await resolveTitleFetchTarget('https://example.test/', { honorProxyEnv: false, lookup }))?.proxy, '')
    assert.equal(lookups, 3)

    // The #129647 review probe: same name and answer, only the env proxy differing.
    answer = '169.254.169.254'
    assert.equal(await resolveTitleFetchTarget('https://evil.example.com/', { lookup }), null)
    assert.equal(await resolveTitleFetchTarget('https://evil.example.com/', { honorProxyEnv: false, lookup }), null)
    assert.equal(lookups, 5)
  })

  test('titleProxyFor mirrors the per-scheme env lookup', () => {
    assert.equal(titleProxyFor(new URL('http://a.test/'), { ALL_PROXY: 'socks5://p:1080' }), 'socks5://p:1080')
    assert.equal(titleProxyFor(new URL('https://a.test/'), { http_proxy: 'http://p:1' }), '')
    assert.equal(titleProxyFor(new URL('http://a.test/'), { HTTP_PROXY: 'http://p:2' }), 'http://p:2')
  })
})

describe('curlTitleTargetArgs', () => {
  test('pins a direct dial to the vetted answer and ignores env proxies', () => {
    assert.deepEqual(
      curlTitleTargetArgs({ addresses: ['93.184.216.34', '10.0.0.1'], hostname: 'a.test', port: '443', proxy: '' }),
      ['--noproxy', '*', '--resolve', 'a.test:443:93.184.216.34']
    )
    assert.deepEqual(
      curlTitleTargetArgs({ addresses: ['2606:2800:220:1::1'], hostname: 'a.test', port: '80', proxy: '' }),
      ['--noproxy', '*', '--resolve', 'a.test:80:[2606:2800:220:1::1]']
    )
    assert.deepEqual(curlTitleTargetArgs({ addresses: [], hostname: '8.8.8.8', port: '80', proxy: '' }), [
      '--noproxy',
      '*'
    ])
  })

  test('hands curl exactly the proxy the guard saw, overriding NO_PROXY', () => {
    assert.deepEqual(
      curlTitleTargetArgs({ addresses: [], hostname: 'a.test', port: '443', proxy: 'http://proxy.corp:3128' }),
      ['--proxy', 'http://proxy.corp:3128', '--noproxy', '']
    )
  })

  const curlAvailable = spawnSync('curl', ['--version'], { stdio: 'ignore' }).status === 0

  test.skipIf(!curlAvailable)('real curl connects to the pinned address without resolving the name', async () => {
    const port = await listen((request, response) => response.end(`host=${request.headers.host}`))

    const args = curlTitleTargetArgs({
      addresses: ['127.0.0.1'],
      hostname: 'pinned.invalid',
      port: String(port),
      proxy: ''
    })

    // `.invalid` can never resolve (RFC 6761), so a response proves the pin was used.
    const result = await new Promise<string>(resolve => {
      const child = spawn('curl', ['--silent', '--max-time', '5', ...args, `http://pinned.invalid:${port}/`])

      let out = ''
      child.stdout.on('data', (chunk: Buffer) => (out += chunk.toString()))
      child.on('close', () => resolve(out))
    })

    assert.equal(result, `host=pinned.invalid:${port}`)
  })
})

describe('fetchWithSafeRedirects', () => {
  test('a public redirect onto a private answer is refused before the private hop', async () => {
    let privateHits = 0
    let fixturePort = 0

    const port = await listen((request, response) => {
      if (request.url === '/start') {
        response.writeHead(302, { Location: `http://private.test:${fixturePort}/secret` })

        return response.end()
      }

      privateHits += 1
      response.end('<title>private</title>')
    })

    fixturePort = port
    const lookup = answers({ 'public.test': ['93.184.216.34'], 'private.test': ['127.0.0.1'] })
    const requested: string[] = []

    const result = await fetchWithSafeRedirects(
      `http://public.test:${port}/start`,
      async (url, _remaining, target: TitleFetchTarget) => {
        requested.push(url)
        assert.deepEqual(target.addresses, ['93.184.216.34'])
        const parsed = new URL(url)

        return new Promise<{ redirectUrl: string; statusCode: number }>((resolve, reject) => {
          http
            .get({ headers: { Host: parsed.host }, host: '127.0.0.1', path: parsed.pathname, port }, response => {
              response.resume()
              response.on('end', () =>
                resolve({ redirectUrl: response.headers.location ?? '', statusCode: response.statusCode ?? 0 })
              )
            })
            .on('error', reject)
        })
      },
      { admit: url => resolveTitleFetchTarget(url, { lookup }), maxRedirects: 3, timeoutMs: 2_000 }
    )

    assert.equal(result.refused, true)
    assert.deepEqual(requested, [`http://public.test:${port}/start`])
    assert.equal(privateHits, 0)
  })

  test('an exhausted redirect budget, a bad Location and a refused first URL all count as refusals', async () => {
    const admit = async () => true
    const loop = async () => ({ redirectUrl: '/again', statusCode: 302 })

    assert.equal(
      (await fetchWithSafeRedirects('https://a.test/', loop, { admit, maxRedirects: 2, timeoutMs: 500 })).refused,
      true
    )
    assert.equal(
      (
        await fetchWithSafeRedirects('https://a.test/', async () => ({ redirectUrl: 'http://[', statusCode: 301 }), {
          admit,
          maxRedirects: 2,
          timeoutMs: 500
        })
      ).refused,
      true
    )

    let fetched = false

    const refusedFirst = await fetchWithSafeRedirects(
      'http://127.0.0.1/',
      async () => {
        fetched = true

        return { statusCode: 200 }
      },
      { admit: async () => null, maxRedirects: 2, timeoutMs: 500 }
    )

    assert.equal(refusedFirst.refused, true)
    assert.equal(fetched, false)
  })
})

describe('metadataRequestOnce', () => {
  class FakeRequest extends EventEmitter {
    aborted = false
    headers: Record<string, string> = {}
    abort() {
      this.aborted = true
    }
    end() {}
    setHeader(name: string, value: string) {
      this.headers[name] = value
    }
  }

  function fakeRequest(drive: (request: FakeRequest) => void): { fn: MetadataRequestFn; last: () => FakeRequest } {
    let last: FakeRequest

    return {
      fn: options => {
        assert.equal(options.redirect, 'manual')
        assert.equal(options.useSessionCookies, false)
        last = new FakeRequest()
        queueMicrotask(() => drive(last))

        return last
      },
      last: () => last
    }
  }

  const options = { headers: { Accept: 'image/*' }, maxBytes: 8, overflow: 'reject' as const, timeoutMs: 1_000 }

  test('reports a redirect with its Location and cancels it instead of following', async () => {
    const fake = fakeRequest(request => request.emit('redirect', 302, 'GET', 'http://10.0.0.1/x', {}))
    const hop = await metadataRequestOnce(fake.fn, 'https://a.test/', options)

    assert.deepEqual(hop, { body: null, contentType: '', redirectUrl: 'http://10.0.0.1/x', statusCode: 302 })
    assert.equal(fake.last().aborted, true)
    assert.equal(fake.last().headers.Accept, 'image/*')
  })

  function respond(chunks: string[]) {
    return (request: FakeRequest) => {
      const response = Object.assign(new EventEmitter(), {
        headers: { 'content-type': ['image/png'] },
        statusCode: 200
      })

      request.emit('response', response)

      for (const chunk of chunks) {
        response.emit('data', Buffer.from(chunk))
      }

      response.emit('end')
    }
  }

  test('returns the body and content type of a final response', async () => {
    const hop = await metadataRequestOnce(fakeRequest(respond(['abc', 'def'])).fn, 'https://a.test/', options)

    assert.equal(hop.statusCode, 200)
    assert.equal(hop.contentType, 'image/png')
    assert.equal(hop.body?.toString(), 'abcdef')
  })

  test('rejects or truncates an over-budget body', async () => {
    const big = ['12345', '67890']

    assert.equal((await metadataRequestOnce(fakeRequest(respond(big)).fn, 'https://a.test/', options)).body, null)
    assert.equal(
      (
        await metadataRequestOnce(fakeRequest(respond(big)).fn, 'https://a.test/', { ...options, overflow: 'truncate' })
      ).body?.toString(),
      '12345678'
    )
  })
})

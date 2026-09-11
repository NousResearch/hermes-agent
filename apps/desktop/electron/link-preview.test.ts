import assert from 'node:assert/strict'

import { afterEach, describe, test, vi } from 'vitest'

import {
  canonicalPreviewKey,
  decodeHtmlEntities,
  fetchWithGuardedRedirects,
  HostRateLimiter,
  isPrivateAddress,
  isPrivateHostname,
  type LinkPreviewIo,
  LinkPreviewStore,
  parseLinkMeta,
  PREVIEW_IMAGE_MAX_BYTES,
  resolveLinkPreview,
  resolveThumbnail,
  usableTitle
} from './link-preview'

const PAGE_URL = 'https://example.com/article'

function fakeIo(options: {
  html?: string
  addresses?: string[]
  renderedTitle?: string
  thumbnail?: string
} = {}): LinkPreviewIo & { calls: string[] } {
  const calls: string[] = []

  return {
    calls,
    fetchHtml: async url => {
      calls.push(`html:${url}`)

      return options.html ?? ''
    },
    fetchRenderedTitle: async url => {
      calls.push(`render:${url}`)

      return options.renderedTitle ?? ''
    },
    resolveHost: async () => options.addresses ?? ['93.184.216.34'],
    fetchThumbnail: async url => {
      calls.push(`thumb:${url}`)

      return options.thumbnail ?? ''
    }
  }
}

function makeDeps(io: LinkPreviewIo, options: { capacity?: number; ttlMs?: number } = {}) {
  const store = new LinkPreviewStore(null, { writeDebounceMs: 5, ...options })
  const limiter = new HostRateLimiter(0, 4)

  return { store, limiter }
}

afterEach(() => {
  vi.useRealTimers()
})

describe('canonicalPreviewKey', () => {
  test('strips www and trailing slashes, keeps query', () => {
    assert.equal(canonicalPreviewKey('https://WWW.Example.com/docs/'), 'example.com/docs')
    assert.equal(canonicalPreviewKey('https://example.com/search?q=1'), 'example.com/search?q=1')
  })

  test('returns the raw value for garbage input rather than throwing', () => {
    assert.equal(canonicalPreviewKey('not a url'), 'not a url')
    assert.equal(canonicalPreviewKey(''), '')
  })
})

describe('decodeHtmlEntities', () => {
  test('named, hex, and decimal entities', () => {
    assert.equal(decodeHtmlEntities('A &amp; B &#x27; &#39; &nbsp;'), "A & B ' '  ")
  })
})

describe('usableTitle', () => {
  test('scrubs captcha/error walls', () => {
    assert.equal(usableTitle('Just a moment...'), '')
    assert.equal(usableTitle('Example – Error'), '')
    assert.equal(usableTitle('A real title'), 'A real title')
    assert.equal(usableTitle(''), '')
  })
})

describe('parseLinkMeta', () => {
  test('prefers og over twitter over <title>', () => {
    const html =
      '<html><head>' +
      '<title>Fallback Title</title>' +
      '<meta property="og:title" content="OG Title">' +
      '<meta name="twitter:title" content="Twitter Title">' +
      '<meta property="og:description" content="OG desc">' +
      '<meta property="og:image" content="https://cdn.example.com/i.png">' +
      '<meta property="og:site_name" content="Example">' +
      '</head></html>'

    const meta = parseLinkMeta(html, PAGE_URL)

    assert.equal(meta.title, 'OG Title')
    assert.equal(meta.description, 'OG desc')
    assert.equal(meta.imageUrl, 'https://cdn.example.com/i.png')
    assert.equal(meta.siteName, 'Example')
    assert.equal(meta.url, PAGE_URL)
  })

  test('falls back to twitter meta, then description meta, then <title>', () => {
    const meta = parseLinkMeta('<meta name="twitter:title" content="T"><meta name="description" content="D">', PAGE_URL)

    assert.equal(meta.title, 'T')
    assert.equal(meta.description, 'D')

    const titled = parseLinkMeta('<title>Only Title</title>', PAGE_URL)

    assert.equal(titled.title, 'Only Title')
    assert.equal(titled.description, '')
  })

  test('decodes entities, collapses whitespace, and caps fields', () => {
    const long = 'x'.repeat(500)

    const meta = parseLinkMeta(
      `<meta property="og:description" content="  A &amp; B\t C  "><meta property="og:title" content="${long}">`,
      PAGE_URL
    )

    assert.equal(meta.description, 'A & B C')
    assert.equal(meta.title.length, 240)
  })

  test('resolves relative og:image against the page and drops non-http schemes', () => {
    const meta = parseLinkMeta('<meta property="og:image" content="/img/thumb.png">', PAGE_URL)

    assert.equal(meta.imageUrl, 'https://example.com/img/thumb.png')

    const data = parseLinkMeta('<meta property="og:image" content="data:image/png;base64,AAAA">', PAGE_URL)

    assert.equal(data.imageUrl, '')
  })

  test('first declared tag wins over later duplicates', () => {
    const meta = parseLinkMeta(
      '<meta property="og:title" content="First"><meta property="og:title" content="Second">',
      PAGE_URL
    )

    assert.equal(meta.title, 'First')
  })
})

describe('private-address guard', () => {
  test('rejects loopback, RFC1918, link-local, CGNAT, ULA, site-local, and unspecified IPv6', () => {
    for (const ip of ['127.0.0.1', '10.1.2.3', '192.168.0.9', '172.16.0.1', '172.31.255.255', '169.254.1.1', '100.64.0.1', '0.0.0.0', '224.0.0.1', '::1', '::', 'fe80::1', 'fec0::1', 'feff::9', 'fc00::1', '::ffff:10.0.0.5']) {
      assert.equal(isPrivateAddress(ip), true, ip)
    }
  })

  test('accepts public addresses', () => {
    for (const ip of ['93.184.216.34', '8.8.8.8', '2606:4700::6810:85e5']) {
      assert.equal(isPrivateAddress(ip), false, ip)
    }
  })

  test('treats unparseable input as private (fail closed)', () => {
    assert.equal(isPrivateAddress(''), true)
    assert.equal(isPrivateAddress('999.1.1.1'), true)
    assert.equal(isPrivateAddress('not-an-ip'), true)
  })

  test('hostname guard: localhost shapes, bare names, .local/.internal/.home.arpa', () => {
    for (const host of ['localhost', 'widget.local', 'svc.internal', 'box.home.arpa', 'intranet', '127.0.0.1', '10.0.0.9', '192.168.1.4', '']) {
      assert.equal(isPrivateHostname(host), true, host)
    }

    for (const host of ['example.com', 'www.example.co.uk', 'WWW.Example.COM']) {
      assert.equal(isPrivateHostname(host), false, host)
    }
  })
})

describe('LinkPreviewStore', () => {
  test('stores, returns by url, and stamps fetchedAt', async () => {
    const store = new LinkPreviewStore(null, { writeDebounceMs: 5 })

    store.set({ url: PAGE_URL, title: 'T', description: '', siteName: '', imageUrl: '', image: '' })

    const got = store.get(PAGE_URL)

    assert.ok(got)
    assert.equal(got.title, 'T')
    assert.ok(Number.isFinite(got.fetchedAt))
  })

  test('expires entries after the TTL', () => {
    let now = 1_000_000
    const store = new LinkPreviewStore(null, { ttlMs: 1_000, now: () => now })

    store.set({ url: PAGE_URL, title: 'T', description: '', siteName: '', imageUrl: '', image: '' })
    now += 2_000

    assert.equal(store.get(PAGE_URL), null)
  })

  test('evicts oldest when full', () => {
    const store = new LinkPreviewStore(null, { capacity: 2 })

    store.set({ url: 'https://a.com/', title: 'A', description: '', siteName: '', imageUrl: '', image: '' })
    store.set({ url: 'https://b.com/', title: 'B', description: '', siteName: '', imageUrl: '', image: '' })
    store.set({ url: 'https://c.com/', title: 'C', description: '', siteName: '', imageUrl: '', image: '' })

    assert.equal(store.get('https://a.com/'), null)
    assert.ok(store.get('https://b.com/'))
    assert.ok(store.get('https://c.com/'))
  })

  test('persists debounced and hydrates fresh entries only', () => {
    vi.useFakeTimers()

    const backing: Record<string, { url: string; title: string; description: string; siteName: string; imageUrl: string; image: string; at: number }> = {}
    let now = 50_000

    const persisted = new LinkPreviewStore(
      {
        read: () => ({}),
        write: entries => {
          for (const [k, v] of Object.entries(entries)) {
            backing[k] = v
          }
        }
      },
      { writeDebounceMs: 10, now: () => now }
    )

    persisted.set({ url: PAGE_URL, title: 'T', description: '', siteName: '', imageUrl: '', image: '' })
    vi.advanceTimersByTime(20)

    assert.ok(backing['example.com/article'])

    now = 200_000

    const hydrated = new LinkPreviewStore(
      { read: () => ({ ...backing }), write: () => undefined },
      { now: () => now, ttlMs: 1_000 }
    )

    assert.equal(hydrated.get(PAGE_URL), null, 'stale entry must not hydrate')
  })

  test('failed fetches are never stored — cache only holds successes', () => {
    const store = new LinkPreviewStore(null)

    store.set({ url: PAGE_URL, title: '', description: 'desc only', siteName: '', imageUrl: '', image: '' })

    assert.ok(store.get(PAGE_URL))
  })
})

describe('HostRateLimiter', () => {
  test('serializes same-host acquisitions', async () => {
    const limiter = new HostRateLimiter(0, 8)
    const order: string[] = []
    let firstRelease: (() => void) | null = null

    const first = limiter.acquire('example.com').then(release => {
      firstRelease = release
      order.push('first-in')
    })

    const second = limiter.acquire('example.com').then(release => {
      order.push('second-in')
      release()
    })

    await Promise.resolve()
    await Promise.resolve()

    assert.ok(firstRelease === null, 'second must wait for the first gate')

    firstRelease?.()
    await first
    await second

    assert.deepEqual(order, ['first-in', 'second-in'])
  })

  test('different hosts proceed independently', async () => {
    const limiter = new HostRateLimiter(0, 8)

    const a = limiter.acquire('a.com')
    const b = limiter.acquire('b.com')

    const [releaseA, releaseB] = await Promise.all([a, b])

    releaseA()
    releaseB()
    assert.ok(true)
  })
})

describe('resolveLinkPreview', () => {
  test('happy path: parses, stores, and returns the envelope', async () => {
    const io = fakeIo({
      html: '<meta property="og:title" content="Hello"><meta property="og:description" content="World">'
    })

    const deps = makeDeps(io)
    const result = await resolveLinkPreview(PAGE_URL, io, deps)

    assert.ok(result.ok)
    assert.equal(result.meta.title, 'Hello')
    assert.equal(result.meta.description, 'World')
    assert.equal(result.meta.url, PAGE_URL)
    assert.ok(deps.store.get(PAGE_URL), 'successful fetch lands in the cache')
  })

  test('non-http URL refused', async () => {
    const io = fakeIo()
    const result = await resolveLinkPreview('ftp://example.com/x', io, makeDeps(io))

    assert.deepEqual(result, { ok: false, reason: 'private-url' })
    assert.deepEqual(io.calls, [])
  })

  test('private hostname refused without any fetch', async () => {
    const io = fakeIo()
    const result = await resolveLinkPreview('https://widget.local/secret', io, makeDeps(io))

    assert.deepEqual(result, { ok: false, reason: 'private-url' })
    assert.deepEqual(io.calls, [])
  })

  test('public name resolving into private space is refused (SSRF door)', async () => {
    const io = fakeIo({ addresses: ['10.0.0.5'] })
    const result = await resolveLinkPreview('https://rebind.example/', io, makeDeps(io))

    assert.deepEqual(result, { ok: false, reason: 'private-url' })
    assert.deepEqual(io.calls, [])
  })

  test('failed DNS resolution refuses the fetch', async () => {
    const io = fakeIo({ addresses: [] })
    const result = await resolveLinkPreview('https://nx.example/', io, makeDeps(io))

    assert.deepEqual(result, { ok: false, reason: 'private-url' })
  })

  test('serves from cache without refetching', async () => {
    const io = fakeIo({ html: '<meta property="og:title" content="Once">' })
    const deps = makeDeps(io)

    const first = await resolveLinkPreview(PAGE_URL, io, deps)

    assert.ok(first.ok)

    const callsAfterFirst = io.calls.length
    const second = await resolveLinkPreview(PAGE_URL, io, deps)

    assert.ok(second.ok)
    assert.equal(second.meta.title, 'Once')
    assert.equal(io.calls.length, callsAfterFirst, 'cache hit must not touch the network')
  })

  test('tier 2 backfills title when tier 1 HTML has none', async () => {
    const io = fakeIo({ html: '<html><body>no meta</body></html>', renderedTitle: 'Rendered Title' })
    const result = await resolveLinkPreview(PAGE_URL, io, makeDeps(io))

    assert.ok(result.ok)
    assert.equal(result.meta.title, 'Rendered Title')
  })

  test('a nothing-readable page is an error miss and is not cached', async () => {
    const io = fakeIo({ html: '', renderedTitle: '' })
    const deps = makeDeps(io)
    const result = await resolveLinkPreview(PAGE_URL, io, deps)

    assert.deepEqual(result, { ok: false, reason: 'error' })
    assert.equal(deps.store.get(PAGE_URL), null, 'misses must not poison the cache')
  })

  test('error-scrubbed titles do not count as usable meta', async () => {
    const io = fakeIo({ html: '<title>Just a moment...</title>', renderedTitle: 'Just a moment...' })
    const result = await resolveLinkPreview(PAGE_URL, io, makeDeps(io))

    assert.deepEqual(result, { ok: false, reason: 'error' })
  })

  test('og:image thumbnail is fetched main-process-side and returned as a data URL', async () => {
    // A minimal PNG-shaped header so the data URL is plainly an image.
    const PNG = Buffer.from('89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489', 'hex')
    const dataUrl = `data:image/png;base64,${PNG.toString('base64')}`
    const io = fakeIo({
      html: '<meta property="og:title" content="Hello"><meta property="og:image" content="https://cdn.example.com/pic.png">',
      thumbnail: dataUrl
    })

    const result = await resolveLinkPreview(PAGE_URL, io, makeDeps(io))

    assert.ok(result.ok)
    assert.equal(result.meta.imageUrl, 'https://cdn.example.com/pic.png')
    assert.equal(result.meta.image, dataUrl)
    assert.ok(io.calls.includes('thumb:https://cdn.example.com/pic.png'), 'main process fetched the thumbnail')
  })

  test('private og:image is refused: no thumbnail call, card still renders', async () => {
    // parseLinkMeta resolves relative URLs against the page; a private host in
    // the og:image must never reach fetchThumbnail as a request.
    const io = fakeIo({
      html: '<meta property="og:title" content="Hello"><meta property="og:image" content="http://127.0.0.1:9222/secret.png">'
    })

    const result = await resolveLinkPreview(PAGE_URL, io, makeDeps(io))

    assert.ok(result.ok)
    assert.equal(result.meta.imageUrl, 'http://127.0.0.1:9222/secret.png')
    assert.equal(result.meta.image, '')
    assert.equal(io.calls.some(call => call.startsWith('thumb:')), false, 'zero thumbnail requests')
  })

  test('thumbnail fetch failure drops the image, not the card', async () => {
    const io = fakeIo({
      html: '<meta property="og:title" content="Hello"><meta property="og:image" content="https://cdn.example.com/pic.png">'
    })

    io.fetchThumbnail = async () => {
      throw new Error('boom')
    }

    const result = await resolveLinkPreview(PAGE_URL, io, makeDeps(io))

    assert.ok(result.ok)
    assert.equal(result.meta.image, '')
    assert.equal(result.meta.title, 'Hello')
  })

  test('never throws across the bridge envelope', async () => {
    const io: LinkPreviewIo = {
      fetchHtml: () => Promise.reject(new Error('boom')),
      fetchRenderedTitle: () => Promise.reject(new Error('boom')),
      resolveHost: () => Promise.reject(new Error('boom')),
      fetchThumbnail: () => Promise.reject(new Error('boom'))
    }

    const result = await resolveLinkPreview(PAGE_URL, io, makeDeps(io))

    assert.deepEqual(result, { ok: false, reason: 'private-url' })
  })
})

describe('fetchWithGuardedRedirects', () => {
  const PUBLIC_IP = '93.184.216.34'

  function hopIo(
    hops: Record<string, { status: number; location?: string; body?: string }>,
    options: { addresses?: string[] } = {}
  ) {
    const requested: string[] = []

    return {
      requested,
      fetchOnce: async (url: string, _addresses: string[]) => {
        requested.push(url)
        const hop = hops[url]

        if (!hop) {
          throw new Error(`unexpected hop: ${url}`)
        }

        return { status: hop.status, location: hop.location ?? '', body: hop.body ?? '' }
      },
      resolveHost: async () => options.addresses ?? [PUBLIC_IP]
    }
  }

  const HTML = '<title>Final Page</title>'

  test('no redirects: single hop, html and url returned', async () => {
    const io = hopIo({ 'https://example.com/a': { status: 200, body: HTML } })
    const result = await fetchWithGuardedRedirects('https://example.com/a', io)

    assert.ok(result.ok)
    assert.equal(result.url, 'https://example.com/a')
    assert.equal(result.body, HTML)
    assert.deepEqual(io.requested, ['https://example.com/a'])
  })

  test('public-to-public redirect chain is followed and reports the final URL', async () => {
    const io = hopIo({
      'https://example.com/a': { status: 302, location: 'https://cdn.example.org/b' },
      'https://cdn.example.org/b': { status: 301, location: '/c' },
      'https://cdn.example.org/c': { status: 200, body: HTML }
    })

    const result = await fetchWithGuardedRedirects('https://example.com/a', io)

    assert.ok(result.ok)
    assert.equal(result.url, 'https://cdn.example.org/c')
    assert.equal(result.body, HTML)
    assert.deepEqual(io.requested, [
      'https://example.com/a',
      'https://cdn.example.org/b',
      'https://cdn.example.org/c'
    ])
  })

  test('redirect to loopback is refused and the private host is never contacted', async () => {
    const io = hopIo({
      'https://example.com/a': { status: 302, location: 'http://127.0.0.1:8080/admin' }
    })

    const result = await fetchWithGuardedRedirects('https://example.com/a', io)

    assert.deepEqual(result, { ok: false, reason: 'private-url' })
    assert.deepEqual(io.requested, ['https://example.com/a'], 'only the initial hop may be requested')
  })

  test('redirect to link-local metadata address is refused', async () => {
    const io = hopIo({
      'https://example.com/a': { status: 302, location: 'http://169.254.169.254/latest/meta-data/' }
    })

    const result = await fetchWithGuardedRedirects('https://example.com/a', io)

    assert.deepEqual(result, { ok: false, reason: 'private-url' })
    assert.deepEqual(io.requested, ['https://example.com/a'])
  })

  test('redirect whose hostname resolves to a private address is refused', async () => {
    let calls = 0

    const io = hopIo(
      {
        'https://example.com/a': { status: 302, location: 'https://rebind.example.net/x' },
        'https://rebind.example.net/x': { status: 200, body: HTML }
      },
      {}
    )

    io.resolveHost = async () => {
      calls += 1

      return calls > 1 ? ['10.0.0.9'] : [PUBLIC_IP]
    }

    const result = await fetchWithGuardedRedirects('https://example.com/a', io)

    assert.deepEqual(result, { ok: false, reason: 'private-url' })
    assert.deepEqual(io.requested, ['https://example.com/a'])
  })

  test('redirect to a non-http scheme is refused', async () => {
    const io = hopIo({
      'https://example.com/a': { status: 302, location: 'file:///etc/passwd' }
    })

    const result = await fetchWithGuardedRedirects('https://example.com/a', io)

    assert.deepEqual(result, { ok: false, reason: 'private-url' })
    assert.deepEqual(io.requested, ['https://example.com/a'])
  })

  test('localhost-named redirect target is refused by name', async () => {
    const io = hopIo({
      'https://example.com/a': { status: 302, location: 'https://localhost:3000/x' }
    })

    const result = await fetchWithGuardedRedirects('https://example.com/a', io)

    assert.deepEqual(result, { ok: false, reason: 'private-url' })
    assert.deepEqual(io.requested, ['https://example.com/a'])
  })

  test('more redirects than the budget allow is an error, not a fetch', async () => {
    const io = hopIo({
      'https://example.com/a': { status: 302, location: 'https://example.com/b' },
      'https://example.com/b': { status: 302, location: 'https://example.com/c' },
      'https://example.com/c': { status: 302, location: 'https://example.com/d' },
      'https://example.com/d': { status: 302, location: 'https://example.com/e' }
    })

    const result = await fetchWithGuardedRedirects('https://example.com/a', io)

    assert.deepEqual(result, { ok: false, reason: 'error' })
    assert.equal(io.requested.length, 4)
  })

  test('initial private URL is refused before any request', async () => {
    const io = hopIo({})
    const result = await fetchWithGuardedRedirects('http://127.0.0.1:9222/json', io)

    assert.deepEqual(result, { ok: false, reason: 'private-url' })
    assert.deepEqual(io.requested, [])
  })

  test('transport failure mid-chain is an empty final body, never a throw', async () => {
    const io = hopIo({
      'https://example.com/a': { status: 302, location: 'https://example.com/b' }
    })

    io.fetchOnce = async (url: string, _addresses: string[]) => {
      if (url === 'https://example.com/b') {
        throw new Error('curl died')
      }

      return { status: 302, location: 'https://example.com/b', body: '' }
    }

    const result = await fetchWithGuardedRedirects('https://example.com/a', io)

    assert.ok(result.ok)
    assert.equal(result.body, '')
  })

  test('every hop request receives the addresses vetted for THAT hop (pinning contract)', async () => {
    // The reviewer's rebinding scenario, upstream #63171's shape: each hop's
    // request must be bound to the addresses the guard just vetted, so a DNS
    // answer swap between verdict and request cannot reach the transport.
    const requested: { addresses: string[]; url: string }[] = []

    const io = {
      requested,
      fetchOnce: async (url: string, addresses: string[]) => {
        requested.push({ addresses, url })

        if (url === 'https://example.com/a') {
          return { status: 302, location: 'https://cdn.example.org/b', body: '' }
        }

        return { status: 200, location: '', body: HTML }
      },
      resolveHost: async (hostname: string) => (hostname === 'example.com' ? ['93.184.216.34'] : ['203.0.113.7'])
    }

    const result = await fetchWithGuardedRedirects('https://example.com/a', io)

    assert.ok(result.ok)
    assert.deepEqual(requested, [
      { url: 'https://example.com/a', addresses: ['93.184.216.34'] },
      { url: 'https://cdn.example.org/b', addresses: ['203.0.113.7'] }
    ])
  })

  test('a hostname whose only answers are site-local (fec0::/10) is refused before any request', async () => {
    const io = hopIo({})

    io.resolveHost = async () => ['fec0::5']

    const result = await fetchWithGuardedRedirects('https://site-local.example.net/page', io)

    assert.deepEqual(result, { ok: false, reason: 'private-url' })
    assert.deepEqual(io.requested, [])
  })
})

describe('resolveThumbnail', () => {
  const PNG = Buffer.from(
    '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478da6360000002000148afa4710000000049454e44ae426082',
    'hex'
  )

  function binaryIo(hops: Record<string, { status: number; location?: string; body?: Buffer }>, options: { addresses?: string[] } = {}) {
    const requested: string[] = []

    return {
      requested,
      fetchOnce: async (url: string, _addresses: string[]) => {
        requested.push(url)
        const hop = hops[url]

        if (!hop) {
          throw new Error(`unexpected hop: ${url}`)
        }

        return { status: hop.status, location: hop.location ?? '', body: hop.body ?? Buffer.alloc(0) }
      },
      resolveHost: async (_hostname: string) => options.addresses ?? ['93.184.216.34']
    }
  }

  test('public png answers as a validated data URL', async () => {
    const io = binaryIo({ 'https://cdn.example.com/pic.png': { status: 200, body: PNG } })
    const dataUrl = await resolveThumbnail('https://cdn.example.com/pic.png', io)

    assert.equal(dataUrl, `data:image/png;base64,${PNG.toString('base64')}`)
    assert.deepEqual(io.requested, ['https://cdn.example.com/pic.png'])
  })

  test('redirect-to-private is refused and the private host is NEVER contacted (review proof)', async () => {
    const io = binaryIo({
      'https://example.com/thumb': { status: 302, location: 'http://169.254.169.254/latest/meta-data/' }
    })

    const dataUrl = await resolveThumbnail('https://example.com/thumb', io)

    assert.equal(dataUrl, '')
    assert.deepEqual(io.requested, ['https://example.com/thumb'], 'zero private requests')
  })

  test('redirect-to-rebinding-name is refused before its request', async () => {
    const io = binaryIo(
      {
        'https://example.com/thumb': { status: 302, location: 'https://swap.example.net/i.png' }
      },
      {}
    )

    io.resolveHost = async hostname => (hostname === 'example.com' ? ['93.184.216.34'] : ['10.0.0.9'])

    const dataUrl = await resolveThumbnail('https://example.com/thumb', io)

    assert.equal(dataUrl, '')
    assert.deepEqual(io.requested, ['https://example.com/thumb'], 'zero private requests')
  })

  test('non-image bytes yield no data URL', async () => {
    const io = binaryIo({ 'https://cdn.example.com/x.png': { status: 200, body: Buffer.from('<html>challenge</html>') } })

    assert.equal(await resolveThumbnail('https://cdn.example.com/x.png', io), '')
  })

  test('oversized bodies yield no data URL', async () => {
    const big = Buffer.concat([PNG, Buffer.alloc(PREVIEW_IMAGE_MAX_BYTES, 0x41)])
    const io = binaryIo({ 'https://cdn.example.com/huge.png': { status: 200, body: big } })

    assert.equal(await resolveThumbnail('https://cdn.example.com/huge.png', io), '')
  })
})

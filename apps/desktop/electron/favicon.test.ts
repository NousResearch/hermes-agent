import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import {
  fallbackIconCandidates,
  type FaviconIo,
  iconCandidatesFromHtml,
  iconCandidatesFromManifest,
  imageMime,
  isPublicHttpUrl,
  largestDeclaredSize,
  manifestUrlFromHtml,
  rankCandidates,
  resolveFavicon,
  sniffImageMime
} from './favicon'

const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, ...new Array(60).fill(0)])
const HTML_BYTES = new Uint8Array([...Buffer.from('<!doctype html><html>nope</html>'), ...new Array(40).fill(0x20)])

/** An IO that serves fixed text per URL and images only for listed URLs,
 *  recording what was asked for and in what order. */
function fakeIo(options: {
  images?: Record<string, { bytes: Uint8Array; mime: string }>
  text?: Record<string, string>
}): { asked: string[]; io: FaviconIo } {
  const asked: string[] = []

  return {
    asked,
    io: {
      fetchImage: async url => {
        asked.push(url)

        return options.images?.[url] ?? null
      },
      fetchText: async url => options.text?.[url] ?? ''
    }
  }
}

describe('which hosts we will ask at all', () => {
  test('a public https host is fair game', () => {
    assert.equal(isPublicHttpUrl('https://linear.app'), true)
  })

  test.each([
    ['loopback by name', 'http://localhost:8000/mcp'],
    ['loopback by address', 'http://127.0.0.1:3000'],
    ['RFC1918 class A', 'http://10.1.2.3'],
    ['RFC1918 class B', 'http://172.16.0.9'],
    ['RFC1918 class C', 'http://192.168.1.5'],
    ['link-local', 'http://169.254.1.1'],
    ['mDNS', 'http://nas.local'],
    ['a bare hostname', 'http://buildbox'],
    ['a non-http scheme', 'file:///etc/passwd']
  ])('%s is refused', (_label, url) => {
    // A private endpoint has no logo out there to find, and asking would
    // announce an internal hostname.
    assert.equal(isPublicHttpUrl(url), false)
  })
})

describe('reading a page for the marks it declares', () => {
  test('declared icons come back absolute against the page', () => {
    const found = iconCandidatesFromHtml('<link rel="icon" href="/assets/mark.png">', 'https://acme.test/docs/start')

    assert.deepEqual(
      found.map(candidate => candidate.url),
      ['https://acme.test/assets/mark.png']
    )
  })

  test('a <base href> wins over the page URL, as a browser would resolve it', () => {
    const found = iconCandidatesFromHtml(
      '<base href="https://cdn.acme.test/"><link rel="icon" href="mark.png">',
      'https://acme.test/docs/'
    )

    assert.deepEqual(
      found.map(candidate => candidate.url),
      ['https://cdn.acme.test/mark.png']
    )
  })

  test('single-quoted and unquoted attributes parse too', () => {
    const found = iconCandidatesFromHtml(
      `<link rel='icon' href='/a.png'><link rel=icon href=/b.png>`,
      'https://acme.test'
    )

    assert.deepEqual(found.map(candidate => candidate.url).sort(), [
      'https://acme.test/a.png',
      'https://acme.test/b.png'
    ])
  })

  test('non-icon links are left alone', () => {
    const found = iconCandidatesFromHtml(
      '<link rel="stylesheet" href="/app.css"><link rel="canonical" href="/">',
      'https://acme.test'
    )

    assert.deepEqual(found, [])
  })

  test('a linked manifest is found', () => {
    assert.equal(
      manifestUrlFromHtml('<link rel="manifest" href="/site.webmanifest">', 'https://acme.test/x'),
      'https://acme.test/site.webmanifest'
    )
  })

  test('a page with no manifest says so rather than guessing one', () => {
    assert.equal(manifestUrlFromHtml('<link rel="icon" href="/a.png">', 'https://acme.test'), '')
  })
})

describe('ranking, so the best mark is fetched first', () => {
  test('the largest declared square wins', () => {
    assert.equal(largestDeclaredSize('32x32 180x180 16x16'), 180)
  })

  test('scalable outranks every raster size', () => {
    const [best] = rankCandidates(
      iconCandidatesFromHtml(
        '<link rel="icon" sizes="256x256" href="/big.png"><link rel="icon" type="image/svg+xml" href="/mark.svg">',
        'https://acme.test'
      )
    )

    assert.equal(best.url, 'https://acme.test/mark.svg')
  })

  test('an apple-touch link with no sizes still beats a guessed path', () => {
    const declared = iconCandidatesFromHtml('<link rel="apple-touch-icon" href="/touch.png">', 'https://acme.test')
    const ranked = rankCandidates([...fallbackIconCandidates('https://acme.test'), ...declared])

    assert.equal(ranked[0].url, 'https://acme.test/touch.png')
  })

  test('a manifest icon is ranked on its declared size', () => {
    const found = iconCandidatesFromManifest(
      JSON.stringify({
        icons: [
          { sizes: '512x512', src: '/pwa-512.png' },
          { sizes: '48x48', src: '/pwa-48.png' }
        ]
      }),
      'https://acme.test/site.webmanifest'
    )

    assert.equal(rankCandidates(found)[0].url, 'https://acme.test/pwa-512.png')
  })

  test('unparseable manifest JSON is not an error, just no candidates', () => {
    assert.deepEqual(iconCandidatesFromManifest('<!doctype html>', 'https://acme.test/m.json'), [])
  })

  test('the same URL declared twice is fetched once, at its best score', () => {
    const ranked = rankCandidates([
      { score: 32, url: 'https://acme.test/a.png' },
      { score: 180, url: 'https://acme.test/a.png' }
    ])

    assert.deepEqual(ranked, [{ score: 180, url: 'https://acme.test/a.png' }])
  })

  test('a page declaring many icons cannot turn one card into many requests', () => {
    const many = Array.from({ length: 40 }, (_unused, index) => ({
      score: index,
      url: `https://acme.test/${index}.png`
    }))

    assert.equal(rankCandidates(many).length, 6)
  })

  test('the apex is tried as well as the subdomain', () => {
    // Vendors routinely serve icons from example.com and nothing from
    // api.example.com.
    const urls = fallbackIconCandidates('https://mcp.acme.test/sse').map(candidate => candidate.url)

    assert.ok(urls.includes('https://mcp.acme.test/favicon.ico'))
    assert.ok(urls.includes('https://acme.test/favicon.ico'))
  })
})

describe('deciding whether bytes are actually an image', () => {
  test.each([
    ['png', [0x89, 0x50, 0x4e, 0x47], 'image/png'],
    ['jpeg', [0xff, 0xd8, 0xff], 'image/jpeg'],
    ['gif', [0x47, 0x49, 0x46, 0x38], 'image/gif'],
    ['ico', [0x00, 0x00, 0x01, 0x00], 'image/x-icon']
  ])('%s is recognised from its magic bytes', (_label, signature, expected) => {
    assert.equal(sniffImageMime(new Uint8Array([...signature, ...new Array(60).fill(0)])), expected)
  })

  test('webp needs both its RIFF header and its WEBP tag', () => {
    const riff = [0x52, 0x49, 0x46, 0x46]
    const webp = [0x57, 0x45, 0x42, 0x50]

    assert.equal(sniffImageMime(new Uint8Array([...riff, 0, 0, 0, 0, ...webp, ...new Array(48).fill(0)])), 'image/webp')
    assert.equal(sniffImageMime(new Uint8Array([...riff, ...new Array(60).fill(0)])), '')
  })

  test('the bytes overrule the server', () => {
    // A blocked request answers 200 with an HTML challenge page under
    // content-type: image/png often enough that believing the header is how
    // you end up rendering a broken-image box.
    assert.equal(imageMime('image/png', HTML_BYTES), '')
  })

  test('an SVG whose opening tag is past the sniff window is trusted on its header', () => {
    const padded = new Uint8Array([...Buffer.from(`<!--${'x'.repeat(2000)}--><svg/>`)])

    assert.equal(imageMime('image/svg+xml', padded), 'image/svg+xml')
  })

  test('a response too short to be an image is refused', () => {
    assert.equal(imageMime('image/png', new Uint8Array([0x89, 0x50])), '')
  })
})

describe('walking the ladder', () => {
  test('a private host is never fetched at all', async () => {
    const { asked, io } = fakeIo({})

    assert.equal(await resolveFavicon('http://127.0.0.1:8000/mcp', io), '')
    assert.deepEqual(asked, [])
  })

  test('a declared icon is preferred over the well-known path', async () => {
    const { io } = fakeIo({
      images: {
        'https://acme.test/declared.png': { bytes: PNG, mime: 'image/png' },
        'https://acme.test/favicon.ico': { bytes: PNG, mime: 'image/x-icon' }
      },
      text: { 'https://acme.test': '<link rel="icon" sizes="180x180" href="/declared.png">' }
    })

    const icon = await resolveFavicon('https://acme.test', io)

    assert.ok(icon.startsWith('data:image/png;base64,'))
  })

  test('a site that declares nothing still gets its guessed favicon', async () => {
    const { io } = fakeIo({ images: { 'https://acme.test/favicon.ico': { bytes: PNG, mime: '' } } })

    assert.ok((await resolveFavicon('https://acme.test', io)).startsWith('data:image/png;base64,'))
  })

  test('a candidate that answers with a challenge page is skipped for the next one', async () => {
    const { io } = fakeIo({
      images: {
        'https://acme.test/apple-touch-icon.png': { bytes: HTML_BYTES, mime: 'image/png' },
        'https://acme.test/favicon.ico': { bytes: PNG, mime: 'image/x-icon' }
      }
    })

    assert.ok((await resolveFavicon('https://acme.test', io)).startsWith('data:image/png;base64,'))
  })

  test('a site nobody can read keeps its monogram rather than asking a third party', async () => {
    const { asked, io } = fakeIo({})

    assert.equal(await resolveFavicon('https://walled.test', io), '')
    // Every attempt was against the site itself. No icon service, because
    // asking one means telling it which connector someone is wiring up.
    assert.ok(asked.length > 0)
    assert.ok(asked.every(url => url.includes('walled.test')))
  })

  test('a page that throws on read falls through to the guessed paths', async () => {
    const io: FaviconIo = {
      fetchImage: async url => (url.endsWith('/favicon.ico') ? { bytes: PNG, mime: '' } : null),
      fetchText: async () => {
        throw new Error('ECONNRESET')
      }
    }

    assert.ok((await resolveFavicon('https://acme.test', io)).startsWith('data:image/png;base64,'))
  })
})

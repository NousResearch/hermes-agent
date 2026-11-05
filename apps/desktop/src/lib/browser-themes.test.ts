import { strToU8, zipSync } from 'fflate'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createBrowserThemesBridge } from './browser-themes'
import { __testing } from './browser-themes'

const GALLERY_URL = 'https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery'

/** A realistic gallery response for one extension. */
function galleryResponse(extensions: unknown[]): unknown {
  return { results: [{ extensions }] }
}

/** The metadata the search maps onto a card. */
function extensionFixture(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    extensionName: 'night-owl',
    displayName: 'Night Owl',
    shortDescription: 'A theme',
    publisher: { publisherName: 'sdras', displayName: 'Sarah Drasner' },
    statistics: [{ statisticName: 'install', value: 1234.6 }],
    tags: ['theme'],
    ...overrides
  }
}

/** Build a real `.vsix` (a zip) with fflate, like the Marketplace serves. */
function buildVsix(files: Record<string, string>): Uint8Array {
  return zipSync(
    Object.fromEntries(Object.entries(files).map(([name, text]) => [name, strToU8(text)])),
    { level: 6 }
  )
}

const NIGHT_OWL_VSIX = () =>
  buildVsix({
    'extension/package.json': JSON.stringify({
      name: 'night-owl',
      displayName: 'Night Owl',
      contributes: {
        themes: [
          { label: 'Night Owl', uiTheme: 'vs-dark', path: './themes/night-owl-color-theme.json' },
          { label: 'Night Owl Light', uiTheme: 'vs', path: 'themes/night-owl-light-color-theme.json' },
          { label: 'Missing', uiTheme: 'vs-dark', path: 'themes/does-not-exist.json' }
        ]
      }
    }),
    'extension/themes/night-owl-color-theme.json': '{"colors": {"dark": true}}',
    'extension/themes/night-owl-light-color-theme.json': '{"colors": {"dark": false}}'
  })

describe('createBrowserThemesBridge (in-tab Marketplace)', () => {
  const fetchMock = vi.fn()

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    fetchMock.mockReset()
    vi.unstubAllGlobals()
  })

  it('queries the gallery directly and maps cards, filtering icon packs', async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(
        JSON.stringify(
          galleryResponse([
            extensionFixture(),
            extensionFixture({
              extensionName: 'file-icons',
              displayName: 'File Icons',
              tags: ['icon-theme', 'theme']
            }),
            extensionFixture({
              extensionName: 'pack',
              displayName: 'Product Icon Pack',
              shortDescription: 'a file icon pack'
            })
          ])
        ),
        { status: 200 }
      )
    )

    const bridge = createBrowserThemesBridge()
    const results = await bridge.themes.searchMarketplace('night')

    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe(GALLERY_URL)
    expect(init.method).toBe('POST')
    expect(init.headers['Accept']).toBe('application/json;api-version=3.0-preview.1')

    const payload = JSON.parse(init.body)
    expect(payload.flags).toBe(772)
    expect(payload.filters[0].criteria).toContainEqual({ filterType: 10, value: 'night' })

    // Icon-tagged and icon-looking extensions are filtered out before the page.
    expect(results).toHaveLength(1)
    expect(results[0]).toEqual({
      extensionId: 'sdras.night-owl',
      displayName: 'Night Owl',
      publisher: 'Sarah Drasner',
      description: 'A theme',
      installs: 1235
    })
  })

  it('resolves the extension, downloads the vsix in-tab, and extracts themes', async () => {
    const vsixUrl = 'https://gallerycdn.vsassets.io/extensions/sdras/night-owl/1.0.0/vspackage'
    fetchMock
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify(
            galleryResponse([
              {
                extensionName: 'night-owl',
                displayName: 'Night Owl',
                versions: [
                  { files: [{ assetType: 'Microsoft.VisualStudio.Services.VSIXPackage', source: vsixUrl }] }
                ]
              }
            ])
          ),
          { status: 200 }
        )
      )
      .mockResolvedValueOnce(new Response(NIGHT_OWL_VSIX() as unknown as BodyInit, { status: 200 }))

    const bridge = createBrowserThemesBridge()
    const result = await bridge.themes.fetchMarketplace('sdras.night-owl')

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[1][0]).toBe(vsixUrl)

    expect(result.extensionId).toBe('sdras.night-owl')
    expect(result.displayName).toBe('Night Owl')
    // The missing entry is skipped, the two real ones extracted, JSONC intact.
    expect(result.themes).toHaveLength(2)
    expect(result.themes[0]).toEqual({
      label: 'Night Owl',
      uiTheme: 'vs-dark',
      contents: '{"colors": {"dark": true}}'
    })
    expect(result.themes[1].label).toBe('Night Owl Light')
  })

  it('rejects ids that are not publisher.extension before any network call', async () => {
    const bridge = createBrowserThemesBridge()

    await expect(bridge.themes.fetchMarketplace('not a marketplace id')).rejects.toThrow(
      'Expected a Marketplace id'
    )
    await expect(bridge.themes.fetchMarketplace('../../etc/passwd')).rejects.toThrow(
      'Expected a Marketplace id'
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('refuses vsix downloads above the byte cap without buffering them', async () => {
    const vsixUrl = 'https://gallerycdn.vsassets.io/extensions/sdras/big/1.0.0/vspackage'
    fetchMock
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify(
            galleryResponse([
              {
                extensionName: 'big',
                versions: [{ files: [{ assetType: 'Microsoft.VisualStudio.Services.VSIXPackage', source: vsixUrl }] }]
              }
            ])
          ),
          { status: 200 }
        )
      )
      // A body whose first chunk already blows the cap.
      .mockImplementationOnce(async () => {
        const huge = new Uint8Array(41 * 1024 * 1024)

        return new Response(huge as unknown as BodyInit, { status: 200 })
      })

    const bridge = createBrowserThemesBridge()

    await expect(bridge.themes.fetchMarketplace('sdras.big')).rejects.toThrow('size limit')
  })
})

describe('extractThemes budget', () => {
  it('skips entries whose declared size overflows the whole-package budget', () => {
    // package.json declares one theme entry; unzipSync reports it huge.
    const vsix = zipSync({
      'extension/package.json': strToU8(
        JSON.stringify({ name: 'x', contributes: { themes: [{ path: 'themes/t.json' }] } })
      ),
      'extension/themes/t.json': new Uint8Array(17 * 1024 * 1024) // 17 MiB > 16 MiB budget
    })

    const themes = __testing.extractThemes(vsix)

    // Over budget: skipped, not inflated (memory never grows by the entry).
    expect(themes).toHaveLength(0)
  })

  it('keeps multiple small entries within one shared budget', () => {
    const vsix = zipSync({
      'extension/package.json': strToU8(
        JSON.stringify({
          name: 'x',
          contributes: { themes: [{ path: 'themes/a.json' }, { path: 'themes/b.json' }] }
        })
      ),
      'extension/themes/a.json': strToU8('{"a":1}'),
      'extension/themes/b.json': strToU8('{"b":2}')
    })

    const themes = __testing.extractThemes(vsix)

    expect(themes).toHaveLength(2)
    expect(themes.map(theme => theme.contents)).toEqual(['{"a":1}', '{"b":2}'])
  })
})

describe('themeEntryName', () => {
  it('normalizes leading ./ and / without stripping interior characters', () => {
    expect(__testing.themeEntryName('./themes/t.json')).toBe('extension/themes/t.json')
    expect(__testing.themeEntryName('/themes/t.json')).toBe('extension/themes/t.json')
    expect(__testing.themeEntryName('themes/t.json')).toBe('extension/themes/t.json')
  })
})
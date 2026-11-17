import { afterEach, describe, expect, it, vi } from 'vitest'

interface RadioModule {
  parseQuery: (text: string) => { name: string; country: string; genre: string }
  searchParams: (query?: Record<string, string | number>) => URLSearchParams
  searchStations: (
    text: string,
    signal: AbortSignal,
    options?: { offset?: number; limit?: number }
  ) => Promise<{ stations: Array<{ id: string; name: string; url: string }>; hls: number; hasMore: boolean; nextOffset: number }>
  popularStations: (
    signal: AbortSignal,
    options?: { offset?: number; limit?: number }
  ) => Promise<{ stations: Array<{ id: string; name: string; url: string }>; hls: number; hasMore: boolean; nextOffset: number }>
  rowsToStations: (rows: unknown[]) => { stations: Array<{ id: string; url: string }>; hls: number; count: number }
  transliterate: (text: string) => string
  fold: (text: string) => string
  searchTerms: (text: string) => string[]
  matchScore: (item: { name: string }, terms: string[]) => number
  shoutcastStations: (
    ctx: { rest: (path: string, options?: unknown) => Promise<unknown> },
    text: string,
    options?: { offset?: number; limit?: number }
  ) => Promise<{ stations: Array<{ id: string; name: string; provider: string; nowPlaying?: string }>; skipped: number; hasMore: boolean; nextOffset: number }>
  tuneinStations: (
    ctx: { rest: (path: string, options?: unknown) => Promise<unknown> },
    text: string,
    options?: { offset?: number; limit?: number }
  ) => Promise<{ stations: Array<{ id: string; name: string; provider: string }>; skipped: number; hasMore: boolean; nextOffset: number }>
}

const modules = import.meta.glob<RadioModule>('../plugins/radio/plugin.js', { eager: true })
const radio = Object.values(modules)[0]
const signal = () => new AbortController().signal

function json(data: unknown) {
  return { ok: true, json: async () => data }
}

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('Radio directory search', () => {
  it('parses country and genre tags, including quoted values', () => {
    expect(radio.parseQuery('country:bg')).toEqual({ name: '', country: 'bg', genre: '' })
    expect(radio.parseQuery('genre:jazz')).toEqual({ name: '', country: '', genre: 'jazz' })
    expect(radio.parseQuery('jazz country:bg')).toEqual({ name: 'jazz', country: 'bg', genre: '' })
    expect(radio.parseQuery('COUNTRY:BG')).toEqual({ name: '', country: 'BG', genre: '' })
    expect(radio.parseQuery('genre:"smooth jazz"')).toEqual({ name: '', country: '', genre: 'smooth jazz' })
    expect(radio.parseQuery('   ')).toEqual({ name: '', country: '', genre: '' })
  })

  it('maps aliases, two-letter codes, names, and tags to directory parameters', () => {
    const params = (query: string) => Object.fromEntries(radio.searchParams(radio.parseQuery(query)))
    expect(params('country:uk').countrycode).toBe('GB')
    expect(params('country:usa').countrycode).toBe('US')
    expect(params('country:bg').countrycode).toBe('BG')
    expect(params('country:bulgaria').country).toBe('bulgaria')
    expect(params('genre:jazz').tag).toBe('jazz')
    const both = params('jazz country:bg genre:smooth')
    expect(both).toMatchObject({ name: 'jazz', countrycode: 'BG', tag: 'smooth' })
    expect(Object.fromEntries(radio.searchParams({ name: 'jazz' }))).not.toHaveProperty('offset')
    expect(Object.fromEntries(radio.searchParams({ name: 'jazz', offset: 40 })).offset).toBe('40')
    expect(Object.fromEntries(radio.searchParams({ name: 'jazz', offset: 40 })).limit).toBe('40')
  })

  it('crosses Cyrillic and Latin names and ranks whole-word matches above lookalikes', () => {
    expect(radio.transliterate('Хоризонт')).toBe('horizont')
    expect(radio.searchTerms('Хоризонт')).toEqual(['Хоризонт', 'horizont'])
    expect(radio.searchTerms('Horizont')).toEqual(['Horizont'])
    expect(radio.fold('Хоризонт')).toBe(radio.fold('Horizont'))
    expect(radio.matchScore({ name: 'BNR Horizont' }, ['Horizont'])).toBeGreaterThan(
      radio.matchScore({ name: 'Horizonte 107.9 FM' }, ['Horizont'])
    )
    expect(radio.matchScore({ name: 'Smooth Jazz' }, ['Horizont'])).toBe(0)
  })

  it('hides HLS rows but counts them', () => {
    const { stations, hls, count } = radio.rowsToStations([
      { stationuuid: 'a', name: 'HLS', url_resolved: 'https://host/live.m3u8', hls: true },
      { stationuuid: 'b', name: 'Playable', url_resolved: 'https://host/live.mp3' },
      { stationuuid: 'c', name: 'Bad', url_resolved: 'ftp://host/live' }
    ])
    expect(stations.map(item => item.id)).toEqual(['b'])
    expect(hls).toBe(1)
    expect(count).toBe(3)
  })

  it('discovers servers, pages with an offset, and reports a full page', async () => {
    const paths: string[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      const path = new URL(url).pathname + new URL(url).search
      paths.push(path)
      if (path.endsWith('/json/servers')) return json([{ name: 'de1.api.radio-browser.info' }])
      const offset = Number(new URL(url).searchParams.get('offset') ?? 0)
      const start = offset || 0
      return json(Array.from({ length: 40 }, (_, index) => ({
        stationuuid: `s${start + index}`,
        name: `Station ${start + index}`,
        url_resolved: `https://stream.example/${start + index}.mp3`,
        tags: 'jazz,live',
        hls: false
      })))
    }))
    const first = await radio.searchStations('jazz', signal(), { limit: 20 })
    // The stub ignores limit, so the caller's page contract is checked on the
    // request and on nextOffset rather than on the stub's row count.
    expect(first.stations).toHaveLength(40)
    expect(first.hasMore).toBe(true)
    expect(first.nextOffset).toBe(20)
    expect(paths.some(path => path.includes('/json/servers'))).toBe(true)
    expect(paths.some(path => path.includes('limit=20'))).toBe(true)
    const second = await radio.searchStations('jazz', signal(), { limit: 20, offset: 20 })
    expect(second.stations[0].id).toBe('s20')
    expect(paths.some(path => path.includes('/json/stations/search'))).toBe(true)
  })

  it('maps Shoutcast and TuneIn route rows and reports skipped dead rows', async () => {
    const paths: string[] = []
    const ctx = {
      rest: async (path: string) => {
        paths.push(path)
        if (path.includes('/shoutcast/')) {
          return {
            stations: [
              { id: 'sc-1', name: 'Smooth Jazz UK', url: 'http://example.test/sc1', description: 'Smooth Jazz', nowPlaying: 'Rod Lucas', homepage: 'https://example.test' },
              { id: 'sc-2', name: 'No Stream', url: null }
            ],
            skipped: 2, count: 60, next: 30
          }
        }
        return { stations: [{ id: 'ti-1', name: 'BNR', url: 'https://example.test/bnr', description: '128 kbps' }], skipped: 1, count: 45, next: 30 }
      }
    }
    const shoutcast = await radio.shoutcastStations(ctx, 'genre:jazz')
    expect(shoutcast.stations).toHaveLength(1)
    expect(shoutcast.stations[0]).toMatchObject({ id: 'sc-1', provider: 'Shoutcast', nowPlaying: 'Rod Lucas' })
    expect(shoutcast.skipped).toBe(2)
    expect(shoutcast.hasMore).toBe(true)
    expect(shoutcast.nextOffset).toBe(30)
    expect(paths[0]).toContain('/shoutcast/search?q=jazz')
    expect(paths[0]).toContain('limit=30')

    const tunein = await radio.tuneinStations(ctx, 'jazz')
    expect(tunein.stations[0]).toMatchObject({ id: 'ti-1', provider: 'TuneIn' })
    expect(tunein.skipped).toBe(1)
    expect(paths[1]).toContain('/tunein/search?q=jazz')
  })

  it('a missing backend reports failure instead of silently succeeding', async () => {
    await expect(radio.shoutcastStations({ rest: undefined as never }, 'jazz')).rejects.toThrow(/not loaded/)
    await expect(radio.tuneinStations({ rest: undefined as never }, 'jazz')).rejects.toThrow(/not loaded/)
  })

  it('reads the top-vote directory through the same 40-row page contract', async () => {
    const paths: string[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      const parsed = new URL(url)
      paths.push(parsed.pathname)
      if (parsed.pathname.endsWith('/json/servers')) return json([{ name: 'de1.api.radio-browser.info' }])
      return json([{ stationuuid: 'top-1', name: 'Top', url_resolved: 'https://stream.example/top.mp3' }])
    }))
    const page = await radio.popularStations(signal())
    expect(page.stations.map(item => item.id)).toEqual(['top-1'])
    expect(page.hasMore).toBe(false)
    expect(paths.some(path => path.includes('/json/stations/topvote/40'))).toBe(true)
  })
})

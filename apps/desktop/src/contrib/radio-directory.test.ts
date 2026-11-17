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
      const path = new URL(url).pathname
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
    expect(first.stations).toHaveLength(20)
    expect(first.hasMore).toBe(true)
    expect(first.nextOffset).toBe(20)
    expect(paths.some(path => path.includes('/json/servers'))).toBe(true)
    const second = await radio.searchStations('jazz', signal(), { limit: 20, offset: 20 })
    expect(second.stations[0].id).toBe('s20')
    expect(paths.some(path => path.includes('/json/stations/search'))).toBe(true)
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

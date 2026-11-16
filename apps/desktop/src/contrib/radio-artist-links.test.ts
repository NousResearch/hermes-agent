import { afterEach, describe, expect, it, vi } from 'vitest'

import { createPluginContext } from './plugin'

interface ArtistModule {
  artistProfile: (relations: unknown[]) => string | null
  resolveArtist: (
    name: string,
    signal: AbortSignal,
    ctx: ReturnType<typeof createPluginContext>
  ) => Promise<string | null>
  trackCredit: (source: unknown) => { artist: string; title: string }
}

// Use the same plain-ESM delivery form as bundled discovery, without reading source text.
const modules = import.meta.glob<ArtistModule>('../plugins/radio/plugin.js', { eager: true })
const { artistProfile, resolveArtist, trackCredit } = Object.values(modules)[0]
const ctx = createPluginContext('radio-artist-test')
const signal = () => new AbortController().signal
const json = (data: unknown) => ({ ok: true, json: async () => data })

afterEach(() => {
  ctx.storage.remove('local.artistRequestAt')
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('Radio artist credits', () => {
  it('separates only a credited artist and accepts direct artist destinations, never track or search links', () => {
    expect(trackCredit({ title: 'AC-DC - Thunderstruck' })).toEqual({ artist: 'AC-DC', title: 'Thunderstruck' })
    expect(trackCredit({ artist: 'AC-DC', title: 'Live - Part Two' })).toEqual({
      artist: 'AC-DC',
      title: 'Live - Part Two'
    })
    expect(trackCredit({ title: 'Unstructured broadcast' }).artist).toBe('')
    expect(trackCredit(null).artist).toBe('')

    const spotify = 'https://open.spotify.com/artist/5V8x0GqH9GCBPl1fYeOhuB'
    const bandcamp = 'https://davdralleon.bandcamp.com/'
    const relation = (type: string, resource: string) => ({ type, url: { resource } })
    expect(artistProfile([relation('bandcamp', bandcamp), relation('free streaming', spotify)])).toBe(spotify)
    expect(
      artistProfile([
        relation('free streaming', 'https://open.spotify.com/track/5V8x0GqH9GCBPl1fYeOhuB'),
        relation('free streaming', 'https://open.spotify.com/search/Dav'),
        relation('bandcamp', `${bandcamp}track/song`),
        relation('official homepage', 'javascript:alert(1)')
      ])
    ).toBeNull()
    expect(artistProfile([relation('bandcamp', bandcamp)])).toBe(bandcamp)
  })

  it('resolves exact artist profiles, rejects ambiguity, and falls back after an unavailable provider', async () => {
    const apple = 'https://music.apple.com/us/artist/hello-meteor/1080606463'
    const row = { wrapperType: 'artist', artistName: 'Hello Meteor', artistLinkUrl: `${apple}?uo=4` }
    const fetch = vi.fn().mockResolvedValueOnce(json({ results: [row] }))
    vi.stubGlobal('fetch', fetch)
    expect(await resolveArtist('Hello Meteor', signal(), ctx)).toBe(apple)
    const request = new URL(fetch.mock.calls[0][0])
    expect(request.searchParams.get('entity')).toBe('musicArtist')
    expect(request.searchParams.get('term')).toBe('Hello Meteor')
    expect(fetch).toHaveBeenCalledTimes(1)

    fetch.mockResolvedValueOnce(
      json({ results: [row, { ...row, artistLinkUrl: 'https://music.apple.com/us/artist/hello-meteor/2' }] })
    )
    expect(await resolveArtist('Hello Meteor', signal(), ctx)).toBeNull()
    expect(fetch).toHaveBeenCalledTimes(2)

    const id = '08d6699f-ec17-434c-a922-3684531840d1'
    const spotify = 'https://open.spotify.com/artist/5V8x0GqH9GCBPl1fYeOhuB'
    vi.stubGlobal('navigator', {
      locks: {
        request: async (_name: string, _options: unknown, callback: () => Promise<unknown>) => callback()
      }
    })
    fetch
      .mockRejectedValueOnce(new Error('Apple unavailable'))
      .mockResolvedValueOnce(json({ artists: [{ name: 'Dav Dralleon', id, score: 100 }] }))
      .mockResolvedValueOnce(json({ relations: [{ type: 'free streaming', url: { resource: spotify } }] }))
    expect(await resolveArtist('Dav Dralleon', signal(), ctx)).toBe(spotify)
    expect(fetch.mock.calls.at(-1)?.[0]).toContain(`/artist/${id}?inc=url-rels`)
  })
})

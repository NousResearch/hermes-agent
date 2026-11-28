import { afterEach, describe, expect, it, vi } from 'vitest'

interface RadioModule {
  PRESETS: Array<{ id: string; name: string; url: string }>
  autoplayOnLaunch: (storage: { get: (key: string, fallback: unknown) => unknown }) => boolean
  shouldAutoplay: (
    storage: { get: (key: string, fallback: unknown) => unknown },
    station: unknown
  ) => boolean
  streamUrl: (value: unknown) => string | null
  createPlayer: (ctx: unknown) => RadioPlayer
}

interface RadioPlayer {
  station: { get: () => { id: string; name: string; url: string } }
  stations: { get: () => Array<{ id: string; name: string; url: string }> }
  favorites: { get: () => Array<{ id: string; name: string; url: string }> }
  status: { get: () => string }
  favorite: (station: { id: string }) => void
  updateStation: (station: { id: string }, changes: { name: string; url: string }) => boolean
  removeStation: (station: { id: string }) => void
  addStation: (input: { name: string; url: string }) => { station: { id: string } | null; added: boolean }
  restoreDefaults: () => number
  startIfRequested: () => boolean
}

const modules = import.meta.glob<RadioModule>('../plugins/radio/plugin.js', { eager: true })
const radio = Object.values(modules)[0]

function makeCtx(initial: Record<string, unknown> = {}) {
  const storage = { ...initial }
  return {
    storage: {
      get: (key: string, fallback: unknown) => (key in storage ? storage[key] : fallback),
      set: (key: string, value: unknown) => { storage[key] = value }
    },
    onDispose() {}
  }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('Radio editable station list', () => {
  it('accepts http and https streams without userinfo and rejects other schemes', () => {
    expect(radio.streamUrl('http://162.252.171.80:8002/stream')).toBe('http://162.252.171.80:8002/stream')
    expect(radio.streamUrl('https://stream.radioparadise.com/aac-128')).toBe('https://stream.radioparadise.com/aac-128')
    expect(radio.streamUrl('ftp://example.test/x')).toBeNull()
    expect(radio.streamUrl('file://../../victim')).toBeNull()
    expect(radio.streamUrl('http://user:pass@example.test/x')).toBeNull()
    expect(radio.streamUrl('not a url')).toBeNull()
  })

  it('seeds once, then keeps edits out of the shipped defaults', () => {
    const ctx = makeCtx()
    const player = radio.createPlayer(ctx)
    expect(player.stations.get().map(item => item.id)).toEqual(radio.PRESETS.map(item => item.id))
    expect(ctx.storage.get('local.stations', null)).toBeNull()
    const target = player.stations.get()[0]
    expect(player.updateStation(target, { name: '  Focus   Music  ', url: 'https://example.test/new.mp3' })).toBe(true)
    expect(player.stations.get()[0].name).toBe('Focus Music')
    expect(player.stations.get()[0].url).toBe('https://example.test/new.mp3')
    expect(player.updateStation(player.stations.get()[0], { name: 'Keep', url: 'not-a-url' })).toBe(false)
    expect(player.stations.get()[0].name).toBe('Focus Music')
  })

  it('keeps favorites and the current station in sync with edits and deletes', () => {
    const ctx = makeCtx()
    const player = radio.createPlayer(ctx)
    const target = player.stations.get()[1]
    player.favorite(target)
    expect(player.favorites.get()).toHaveLength(1)
    expect(player.updateStation(target, { name: 'Renamed', url: 'http://example.test/live' })).toBe(true)
    expect(player.favorites.get()[0]).toMatchObject({ name: 'Renamed', url: 'http://example.test/live' })
    player.removeStation(player.favorites.get()[0])
    expect(player.favorites.get()).toHaveLength(0)
    expect(player.stations.get().some(item => item.id === target.id)).toBe(false)
  })

  it('adds custom streams, rejects duplicates, and restores only missing defaults', async () => {
    const ctx = makeCtx()
    const player = radio.createPlayer(ctx)
    const added = player.addStation({ name: ' Shoutcast Tune ', url: 'http://example.test:8000/;' })
    expect(added.added).toBe(true)
    expect(added.station?.id).toMatch(/^custom-/)
    expect(player.stations.get()).toHaveLength(radio.PRESETS.length + 1)
    const duplicate = player.addStation({ name: 'Again', url: 'http://example.test:8000/;' })
    expect(duplicate.added).toBe(false)
    expect(duplicate.station?.id).toBe(added.station?.id)
    expect(player.addStation({ name: 'Bad', url: 'mailto:someone@example.test' }).station).toBeNull()

    player.removeStation(player.stations.get()[0])
    expect(player.restoreDefaults()).toBe(1)
    expect(player.stations.get().map(item => item.id)).toContain(radio.PRESETS[0].id)
    expect(player.restoreDefaults()).toBe(0)
  })

  it('starts the last station on launch only when the saved switch is on', () => {
    const off = makeCtx()
    expect(radio.autoplayOnLaunch(off.storage)).toBe(false)
    expect(radio.shouldAutoplay(off.storage, radio.PRESETS[0])).toBe(false)
    expect(radio.createPlayer(off).startIfRequested()).toBe(false)

    const last = radio.PRESETS[1]
    const on = makeCtx({ 'local.autoplay': true, 'local.station': last })
    expect(radio.autoplayOnLaunch(on.storage)).toBe(true)
    expect(radio.shouldAutoplay(on.storage, last)).toBe(true)
    expect(radio.createPlayer(on).startIfRequested()).toBe(true)
  })
})

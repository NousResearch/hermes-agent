import { isValidElement } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { discoverBundledPlugins } from './plugins'
import { $pluginDecisions, $pluginRecords, setPluginEnabled } from './plugins-store'
import { registry } from './registry'

vi.mock('./runtime-loader', () => ({ watchRuntimePlugins: vi.fn() }))

interface RadioPlayer {
  play: (station?: unknown) => Promise<void>
  stop: () => void
  toggle: () => void
  status: { get: () => string }
}

// jsdom has no decoder and no Web Audio, so the transport is exercised through
// the real element, its real events, and a stand-in audio graph that stays
// "running" (the metered path is the one with an analyser to keep alive).
class FakeAudioContext {
  state = 'running'
  sampleRate = 48000
  destination = {}

  createAnalyser() {
    return { fftSize: 0, connect() {}, disconnect() {}, getFloatTimeDomainData() {} }
  }

  createMediaElementSource() {
    return { connect() {}, disconnect() {} }
  }

  createGain() {
    return { gain: { value: 1 }, connect() {}, disconnect() {} }
  }

  resume() {
    return Promise.resolve()
  }

  suspend() {
    return Promise.resolve()
  }

  close() {
    return Promise.resolve()
  }
}

function player(): RadioPlayer {
  const contribution = registry.getArea('statusBar.right').find(item => item.source === 'plugin:radio')
  const element = contribution?.render?.()

  if (!isValidElement<{ player: RadioPlayer }>(element)) {
    throw new Error('Radio did not contribute its player through the SDK')
  }

  return element.props.player
}

afterEach(async () => {
  vi.useRealTimers()
  await setPluginEnabled('radio', false)
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('bundled Radio plugin', () => {
  it('inventories off by default and follows the ordinary live enable/disable lifecycle without autoplay', async () => {
    // Other bundled plugins are outside this integration test; Radio has no saved decision.
    $pluginDecisions.set({ accent: false, kanban: false, 'hermes-bots': false })
    const fetch = vi.fn()
    const audio = vi.fn()
    vi.stubGlobal('fetch', fetch)
    vi.stubGlobal('Audio', audio)
    const initialStyles = document.head.querySelectorAll('style').length

    discoverBundledPlugins()
    expect($pluginRecords.get().radio).toMatchObject({ kind: 'bundled', status: 'disabled' })
    expect(registry.getArea('statusBar.right').some(item => item.source === 'plugin:radio')).toBe(false)
    expect(document.head.querySelectorAll('style').length).toBe(initialStyles)

    await setPluginEnabled('radio', true)
    expect($pluginRecords.get().radio.status).toBe('loaded')
    expect(player().status.get()).toBe('paused')
    expect(audio).not.toHaveBeenCalled()
    expect(fetch).not.toHaveBeenCalled()

    await setPluginEnabled('radio', false)
    expect(registry.getArea('statusBar.right').some(item => item.source === 'plugin:radio')).toBe(false)
    expect(document.head.querySelectorAll('style').length).toBe(initialStyles)
    expect($pluginDecisions.get().radio).toBe(false)
  })

  it('releases the stream on disable and ignores late media events from the old player', async () => {
    $pluginDecisions.set({ accent: false, kanban: false, 'hermes-bots': false })
    discoverBundledPlugins()
    // jsdom has no decoder: the actual player and plugin lifecycle run against DOM media events.
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue()
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
    vi.stubGlobal('AudioContext', undefined)
    await setPluginEnabled('radio', true)
    const first = player()
    await first.play()
    const media = document.querySelector('audio')!
    media.dispatchEvent(new Event('playing'))
    expect(first.status.get()).toBe('live')

    await setPluginEnabled('radio', false)
    expect(document.querySelector('audio')).toBeNull()
    expect(media.hasAttribute('src')).toBe(false)
    media.dispatchEvent(new Event('playing'))
    expect(first.status.get()).toBe('paused')

    await setPluginEnabled('radio', true)
    expect(player()).not.toBe(first)
    expect(player().status.get()).toBe('paused')
    expect(document.querySelector('audio')).toBeNull()
  })
})

async function playStation() {
  $pluginDecisions.set({ accent: false, kanban: false, 'hermes-bots': false })
  const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
  vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
  vi.stubGlobal('AudioContext', FakeAudioContext)
  vi.useFakeTimers()
  discoverBundledPlugins()
  await setPluginEnabled('radio', true)
  const radio = player()
  await radio.play()
  const media = document.querySelector('audio')!

  return { radio, media, play, url: media.src }
}

describe('bundled Radio plugin uninterrupted playback', () => {
  it('resumes a stream paused outside the player, and keeps a pause the player itself made', async () => {
    const { radio, media, play } = await playStation()
    media.dispatchEvent(new Event('playing'))
    expect(radio.status.get()).toBe('live')

    // Media keys, an audio-device change and the OS all pause the element from
    // outside the widget. That must not read as the user pausing.
    const before = play.mock.calls.length
    media.dispatchEvent(new Event('pause'))
    expect(play.mock.calls.length).toBe(before + 1)
    expect(radio.status.get()).toBe('live')

    radio.toggle()
    expect(radio.status.get()).toBe('paused')
    const paused = play.mock.calls.length
    media.dispatchEvent(new Event('pause'))
    expect(play.mock.calls.length).toBe(paused)
    expect(radio.status.get()).toBe('paused')
  })

  it('reconnects a stream that stopped while playing instead of reporting a pause', async () => {
    const { radio, media, url } = await playStation()
    media.dispatchEvent(new Event('playing'))
    expect(radio.status.get()).toBe('live')

    // A live stream the broadcaster closes mid-playback: reconnect the same
    // station rather than parking a silent widget in a paused/error state.
    media.dispatchEvent(new Event('ended'))
    expect(radio.status.get()).toBe('connecting')
    expect(document.querySelector('audio')).toBeNull()

    await vi.advanceTimersByTimeAsync(1000)
    const reconnected = document.querySelector('audio')!
    expect(reconnected).not.toBe(media)
    expect(reconnected.src).toBe(url)
    reconnected.dispatchEvent(new Event('playing'))
    expect(radio.status.get()).toBe('live')
  })

  it('heals a stall while live in seconds, not after a long freeze', async () => {
    const { radio, media, url } = await playStation()
    media.dispatchEvent(new Event('playing'))
    expect(radio.status.get()).toBe('live')

    // Nothing else is fed to the element: jsdom reports no progress, which is
    // what a starved live stream looks like. The player must reconnect quickly —
    // a live stream rejoins at the live edge, so the listener hears a fraction
    // of a second instead of the whole stall.
    await vi.advanceTimersByTimeAsync(4000)
    expect(radio.status.get()).toBe('live')

    await vi.advanceTimersByTimeAsync(2000)
    expect(radio.status.get()).toBe('connecting')
    expect(document.querySelector('audio')).toBeNull()

    await vi.advanceTimersByTimeAsync(1000)
    const rejoined = document.querySelector('audio')!
    expect(rejoined.src).toBe(url)
    rejoined.dispatchEvent(new Event('playing'))
    expect(radio.status.get()).toBe('live')
  })

  it('still reports a station that never produced audio as unavailable', async () => {
    const { radio } = await playStation()

    // One metered attempt, one plain fallback, then the station is called
    // unavailable — recovery is for streams that were playing, not for
    // stations that never play at all.
    document.querySelector('audio')!.dispatchEvent(new Event('error'))
    document.querySelector('audio')!.dispatchEvent(new Event('error'))
    expect(radio.status.get()).toBe('error')
    expect(document.querySelector('audio')).toBeNull()
  })
})

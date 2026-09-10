import { isValidElement } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { discoverBundledPlugins } from './plugins'
import { $pluginDecisions, $pluginRecords, setPluginEnabled } from './plugins-store'
import { registry } from './registry'

vi.mock('./runtime-loader', () => ({ watchRuntimePlugins: vi.fn() }))

interface RadioPlayer {
  play: () => Promise<void>
  status: { get: () => string }
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

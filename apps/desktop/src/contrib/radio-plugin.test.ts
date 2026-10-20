import { cleanup, render, screen } from '@testing-library/react'
import { createElement, isValidElement } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { discoverBundledPlugins } from './plugins'
import { $pluginDecisions, $pluginRecords, setPluginEnabled } from './plugins-store'
import { registry } from './registry'

vi.mock('./runtime-loader', () => ({ watchRuntimePlugins: vi.fn() }))

interface RadioPlayer {
  play: () => Promise<void>
  status: { get: () => string }
  stop: () => void
  toggle: () => void | Promise<void>
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
  cleanup()
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
    vi.stubGlobal('matchMedia', () => ({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() }))
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

    const contribution = registry.getArea('statusBar.right').find(item => item.source === 'plugin:radio')
    render(createElement(I18nProvider, { children: contribution?.render?.(), configClient: null, initialLocale: 'ko' }))
    expect(screen.getByRole('button', { name: '라디오 재생' })).toBeTruthy()
    expect(screen.getByRole('button', { name: '다음 방송국' })).toBeTruthy()
    expect(audio).not.toHaveBeenCalled()
    expect(fetch).not.toHaveBeenCalled()
    cleanup()

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

  it('survives external pause without destroying the stream or blocking resume', async () => {
    $pluginDecisions.set({ accent: false, kanban: false, 'hermes-bots': false })
    discoverBundledPlugins()
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue()
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
    vi.stubGlobal('AudioContext', undefined)
    await setPluginEnabled('radio', true)
    const radio = player()
    await radio.play()
    const media = document.querySelector('audio')!
    media.dispatchEvent(new Event('playing'))
    expect(radio.status.get()).toBe('live')
    const originalSrc = media.getAttribute('src')

    // External controller (Fluid Voice, Whisper Flow, media keys) pauses the element.
    Object.defineProperty(media, 'paused', { value: true, configurable: true })
    Object.defineProperty(media, 'ended', { value: false, configurable: true })
    media.dispatchEvent(new Event('pause'))

    // Status should reflect pause, but the element and its src must survive.
    expect(radio.status.get()).toBe('paused')
    expect(document.querySelector('audio')).toBe(media)
    expect(media.getAttribute('src')).toBe(originalSrc)

    // Resume is then possible.
    Object.defineProperty(media, 'paused', { value: false })
    media.dispatchEvent(new Event('playing'))
    expect(radio.status.get()).toBe('live')
  })

  it('resumes the preserved stream when dictation text lands in an editable field (#108113)', async () => {
    $pluginDecisions.set({ accent: false, kanban: false, 'hermes-bots': false })
    discoverBundledPlugins()
    const playMock = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue()
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
    vi.stubGlobal('AudioContext', undefined)
    await setPluginEnabled('radio', true)
    const radio = player()
    await radio.play()
    const media = document.querySelector('audio')!
    media.dispatchEvent(new Event('playing'))
    expect(radio.status.get()).toBe('live')
    playMock.mockClear()

    // Fluid Voice pauses the stream, then inserts the transcript as a paste
    // into the focused field — no media resume event ever follows.
    Object.defineProperty(media, 'paused', { value: true, configurable: true })
    Object.defineProperty(media, 'ended', { value: false, configurable: true })
    media.dispatchEvent(new Event('pause'))
    expect(radio.status.get()).toBe('paused')

    const composer = document.createElement('textarea')
    document.body.append(composer)
    composer.dispatchEvent(new Event('paste', { bubbles: true }))

    await vi.waitFor(() => expect(radio.status.get()).toBe('live'))
    // Same element, same stream — no teardown and reconnect.
    expect(document.querySelector('audio')).toBe(media)
    expect(playMock).toHaveBeenCalled()
    composer.remove()
  })

  it('keeps the plugin Pause destructive: no paste resume, and Play reconnects (#108113)', async () => {
    $pluginDecisions.set({ accent: false, kanban: false, 'hermes-bots': false })
    discoverBundledPlugins()
    const playMock = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue()
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
    vi.stubGlobal('AudioContext', undefined)
    await setPluginEnabled('radio', true)
    const radio = player()
    await radio.play()
    const first = document.querySelector('audio')!
    first.dispatchEvent(new Event('playing'))
    expect(radio.status.get()).toBe('live')

    // The plugin's own Pause releases the stream by design.
    radio.toggle()
    expect(radio.status.get()).toBe('paused')
    expect(document.querySelector('audio')).toBeNull()
    expect(first.hasAttribute('src')).toBe(false)

    // A dictation paste must NOT resurrect a stream the user paused.
    const composer = document.createElement('textarea')
    document.body.append(composer)
    composer.dispatchEvent(new Event('paste', { bubbles: true }))
    expect(radio.status.get()).toBe('paused')
    expect(document.querySelector('audio')).toBeNull()
    composer.remove()

    // Play after an intentional pause is a full reconnect: a fresh element.
    playMock.mockClear()
    await radio.toggle()
    const second = document.querySelector('audio')!
    expect(second).not.toBe(first)
    second.dispatchEvent(new Event('playing'))
    expect(radio.status.get()).toBe('live')
  })
})

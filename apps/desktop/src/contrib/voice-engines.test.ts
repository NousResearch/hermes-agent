/**
 * The live-voice-engine contract: a plugin registers a `LiveVoiceEngine` on
 * `VOICE_ENGINE_AREA` through its ordinary `ctx.register`; the namespaced id
 * becomes the `voice.voice_chat_mode` value. These drive the REAL plugin
 * context and registry: what a plugin can register, and what core accepts.
 */
import { type LiveVoiceEngine, VOICE_ENGINE_AREA } from '@hermes/plugin-sdk'
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createPluginContext } from './plugin'
import { registry } from './registry'
import {
  findLiveVoiceEngine,
  liveVoiceEngineStatus,
  resolveLiveVoiceEngines,
  useLiveVoiceEngines
} from './voice-engines'

const loaded: Array<() => void> = []

function fakeEngine(overrides: Partial<LiveVoiceEngine> = {}): LiveVoiceEngine {
  return {
    label: 'Fake Voice',
    start: vi.fn(async () => ({ stop: vi.fn(async () => undefined) })),
    status: vi.fn(async () => ({ available: true })),
    ...overrides
  }
}

/** Load a plugin the way the loader does; returns its unload. */
function loadPlugin(pluginId: string, engines: Record<string, unknown>) {
  const disposers: Array<() => void> = []
  const ctx = createPluginContext(pluginId, dispose => disposers.push(dispose))

  for (const [id, data] of Object.entries(engines)) {
    ctx.register({ area: VOICE_ENGINE_AREA, data, id })
  }

  const unload = () => disposers.splice(0).forEach(dispose => dispose())
  loaded.push(unload)

  return unload
}

afterEach(() => {
  cleanup()
  loaded.splice(0).forEach(unload => unload())
  vi.restoreAllMocks()
})

describe('VOICE_ENGINE_AREA registration', () => {
  it('namespaces the engine id by plugin, which is the persisted mode value', () => {
    loadPlugin('hermes-xai', {
      grok: fakeEngine({
        description: 'full-duplex, delegates to Hermes',
        label: 'Grok Voice',
        settingsKeys: ['voice', 'instructions'],
        signIn: { provider: 'xai-oauth' }
      })
    })

    const [engine] = resolveLiveVoiceEngines(registry.getArea(VOICE_ENGINE_AREA))

    expect(engine).toMatchObject({
      description: 'full-duplex, delegates to Hermes',
      id: 'hermes-xai:grok',
      label: 'Grok Voice',
      pluginId: 'hermes-xai',
      settingsKeys: ['voice', 'instructions'],
      signIn: { provider: 'xai-oauth' }
    })
    expect(findLiveVoiceEngine('hermes-xai:grok')?.label).toBe('Grok Voice')
  })

  it('drops contributions core cannot drive, and ids that cannot be persisted', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    loadPlugin('demo', {
      'Bad Id': fakeEngine(),
      'no-label': fakeEngine({ label: '  ' }),
      'no-start': { label: 'Half', status: async () => ({ available: true }) },
      ok: fakeEngine({ settingsKeys: ['voice', 3 as unknown as string, ''], signIn: { provider: '' } })
    })

    const engines = resolveLiveVoiceEngines(registry.getArea(VOICE_ENGINE_AREA))

    expect(engines.map(engine => engine.id)).toEqual(['demo:ok'])
    expect(engines[0]).toMatchObject({ settingsKeys: ['voice'], signIn: null })
    expect(warn).toHaveBeenCalled()
  })

  it('lists engines in id order and follows plugins loading and unloading', () => {
    const { result } = renderHook(() => useLiveVoiceEngines())
    expect(result.current).toEqual([])

    let unloadB: () => void = () => undefined

    act(() => {
      unloadB = loadPlugin('zeta', { live: fakeEngine({ label: 'Zeta' }) })
      loadPlugin('alpha', { live: fakeEngine({ label: 'Alpha' }) })
    })

    expect(result.current.map(engine => engine.label)).toEqual(['Alpha', 'Zeta'])

    act(() => unloadB())

    expect(result.current.map(engine => engine.id)).toEqual(['alpha:live'])
    expect(findLiveVoiceEngine('zeta:live')).toBeNull()
  })

  it('resolves built-in and unknown modes to no engine', () => {
    loadPlugin('hermes-xai', { grok: fakeEngine() })

    expect(findLiveVoiceEngine('chained')).toBeNull()
    expect(findLiveVoiceEngine('gpt-live')).toBeNull()
    expect(findLiveVoiceEngine('hermes-xai:other')).toBeNull()
  })
})

describe('liveVoiceEngineStatus', () => {
  it('passes the engine verdict through', async () => {
    loadPlugin('p', { e: fakeEngine({ status: async () => ({ available: false, reason: 'Needs an xAI sign-in' }) }) })

    await expect(liveVoiceEngineStatus(findLiveVoiceEngine('p:e')!)).resolves.toEqual({
      available: false,
      reason: 'Needs an xAI sign-in'
    })
  })

  it('reads a throwing or hanging status() as unavailable instead of failing the picker', async () => {
    loadPlugin('p', {
      hangs: fakeEngine({ status: () => new Promise(() => undefined) }),
      throws: fakeEngine({ status: async () => Promise.reject(new Error('backend unreachable')) })
    })

    await expect(liveVoiceEngineStatus(findLiveVoiceEngine('p:throws')!)).resolves.toEqual({
      available: false,
      reason: 'backend unreachable'
    })
    await expect(liveVoiceEngineStatus(findLiveVoiceEngine('p:hangs')!, 10)).resolves.toEqual({
      available: false,
      reason: 'status check timed out'
    })
  })
})

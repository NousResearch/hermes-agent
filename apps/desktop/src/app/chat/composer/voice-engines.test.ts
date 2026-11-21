import { describe, expect, it } from 'vitest'

import {
  $pluginVoiceEngineChoice,
  choosePluginVoiceEngine,
  resolvePluginVoiceEngine
} from '@/store/plugin-voice-engine'

import { voiceEnginesFrom } from './contrib'

const start = () => undefined

describe('composer.voiceEngines', () => {
  it('keeps well-formed engines in registry order and drops malformed or duplicate payloads', () => {
    const engines = voiceEnginesFrom([
      { data: { label: 'Gemini Live', shortLabel: 'Gemini', start }, id: 'gemini-live.engine' },
      { data: { label: 'No start', shortLabel: 'x' }, id: 'broken.a' },
      { data: { label: '  ', shortLabel: 'x', start }, id: 'broken.b' },
      { data: null, id: 'broken.c' },
      { data: { label: 'Dup', shortLabel: 'Dup', start }, id: 'gemini-live.engine' },
      { data: { detail: '  needs a key ', label: 'Other', shortLabel: 'O', start }, id: 'other.engine' }
    ])

    expect(engines.map(e => e.id)).toEqual(['gemini-live.engine', 'other.engine'])
    expect(engines[1].detail).toBe('needs a key')
  })

  it('a choice is per scope, survives only while its engine is registered, and clears back to core', () => {
    const engines = voiceEnginesFrom([{ data: { label: 'Gemini Live', shortLabel: 'Gemini', start }, id: 'g' }])

    choosePluginVoiceEngine('g', 'conn::alice')
    expect(resolvePluginVoiceEngine(engines, $pluginVoiceEngineChoice.get(), 'conn::alice')?.id).toBe('g')
    // another profile on the same connection is untouched
    expect(resolvePluginVoiceEngine(engines, $pluginVoiceEngineChoice.get(), 'conn::bob')).toBeNull()
    // plugin uninstalled: the button falls back to the core engine
    expect(resolvePluginVoiceEngine([], $pluginVoiceEngineChoice.get(), 'conn::alice')).toBeNull()

    choosePluginVoiceEngine(null, 'conn::alice')
    expect($pluginVoiceEngineChoice.get()).toEqual({})
  })
})

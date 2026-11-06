import { describe, expect, it } from 'vitest'

import {
  createSettingsStore,
  DEFAULT_SETTINGS,
  MAX_CUSTOM_STYLE_CHARS,
  normalizeSettings,
  requestFor,
  SETTINGS_KEY
} from './settings'

function memoryStorage(initial: Record<string, unknown> = {}) {
  const data = new Map(Object.entries(initial))

  return {
    data,
    get: (key: string, fallback: unknown) => (data.has(key) ? data.get(key) : fallback),
    set: (key: string, value: unknown) => void data.set(key, value)
  }
}

describe('prompt refinement settings', () => {
  it('ships Off, with the depth chosen automatically (Deepen when fixed)', () => {
    expect(DEFAULT_SETTINGS.mode).toBe('off')
    expect(DEFAULT_SETTINGS.autoDepth).toBe(true)
    expect(DEFAULT_SETTINGS.depth).toBe('deepen')
    expect(normalizeSettings({ autoDepth: 'sure' }).autoDepth).toBe(true)
    expect(normalizeSettings({ autoDepth: false }).autoDepth).toBe(false)
    expect(createSettingsStore(memoryStorage()).get().mode).toBe('off')
  })

  it('falls back field by field, never to a more active mode', () => {
    expect(normalizeSettings({ mode: 'always', modelSource: 'gpt', minWords: 2.5 })).toEqual(DEFAULT_SETTINGS)
    expect(normalizeSettings('corrupt')).toEqual(DEFAULT_SETTINGS)
    expect(normalizeSettings({ mode: 'auto', minWords: 999 })).toMatchObject({ mode: 'auto', minWords: 50 })
    expect(normalizeSettings({ depth: 'extreme' }).depth).toBe('deepen')
    expect(normalizeSettings({ depth: 'polish' }).depth).toBe('polish')
    expect(normalizeSettings({ style: 'poetic', useChatContext: 'yes', customStyle: 7 })).toMatchObject({
      style: 'auto',
      useChatContext: false,
      customStyle: ''
    })
    expect(normalizeSettings({ customStyle: 'x'.repeat(MAX_CUSTOM_STYLE_CHARS + 50) }).customStyle).toHaveLength(
      MAX_CUSTOM_STYLE_CHARS
    )
  })

  it('ships with chat context off and the Auto style', () => {
    expect(DEFAULT_SETTINGS).toMatchObject({ style: 'auto', customStyle: '', useChatContext: false })
  })

  it('saves style, custom guidance and chat context, and reads them back after a restart', () => {
    const storage = memoryStorage()
    createSettingsStore(storage).update({ style: 'custom', customStyle: 'Bullet points.', useChatContext: true })

    expect(createSettingsStore(storage).get()).toMatchObject({
      style: 'custom',
      customStyle: 'Bullet points.',
      useChatContext: true
    })
  })
})

describe('requestFor — what both modes send', () => {
  const base = { ...DEFAULT_SETTINGS, mode: 'manual' as const, autoDepth: false }

  it('asks for Auto depth while it is on, and the fixed depth otherwise', () => {
    expect(requestFor({ ...base, autoDepth: true, depth: 'polish' }, 'go', []).depth).toBe('auto')
    expect(requestFor({ ...base, depth: 'polish' }, 'go', []).depth).toBe('polish')
  })

  it('sends custom guidance only with the Custom style', () => {
    expect(requestFor({ ...base, style: 'technical', customStyle: 'Be terse.' }, 'go', [])).toEqual({
      text: 'go',
      attachmentLabels: [],
      style: 'technical',
      depth: 'deepen'
    })
    expect(requestFor({ ...base, style: 'custom', customStyle: '  Be terse. ' }, 'go', []).customStyle).toBe(
      'Be terse.'
    )
    expect(requestFor({ ...base, style: 'custom', customStyle: '   ' }, 'go', [])).not.toHaveProperty('customStyle')
  })

  it('carries context only when there is some', () => {
    expect(requestFor(base, 'go', ['a.png'], '')).not.toHaveProperty('context')
    expect(requestFor(base, 'go', ['a.png'], 'User: hi')).toMatchObject({
      attachmentLabels: ['a.png'],
      context: 'User: hi'
    })
  })

  it('persists under its versioned, global-scoped key and follows another window', () => {
    const storage = memoryStorage()
    const store = createSettingsStore(storage)

    store.update({ mode: 'manual', modelSource: 'chat' })
    expect(storage.data.get(SETTINGS_KEY)).toMatchObject({ version: 1, mode: 'manual', modelSource: 'chat' })
    expect(SETTINGS_KEY).toBe('v1.global.settings')

    storage.set(SETTINGS_KEY, { ...store.get(), mode: 'auto' })
    store.reload()
    expect(store.get().mode).toBe('auto')

    storage.set(SETTINGS_KEY, { ...store.get(), depth: 'clarify' })
    store.reload()
    expect(store.get().depth).toBe('clarify')
  })
})

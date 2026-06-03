import { describe, expect, it } from 'vitest'

import type { HermesConfigRecord } from '@/types/hermes'

import { getNested, providerGroup, setNested } from './helpers'

describe('settings helpers', () => {
  it('reads and writes nested config paths', () => {
    const config: HermesConfigRecord = { display: { theme: 'mono' } }
    const next = setNested(config, 'display.theme', 'slate')

    expect(getNested(next, 'display.theme')).toBe('slate')
    expect(getNested(config, 'display.theme')).toBe('mono')
  })

  it('rejects prototype-polluting config paths', () => {
    const config: HermesConfigRecord = {}

    expect(() => setNested(config, '__proto__.polluted', true)).toThrow('Unsafe config path')
    expect(() => setNested(config, 'constructor.prototype.polluted', true)).toThrow('Unsafe config path')
    expect(({} as Record<string, unknown>).polluted).toBeUndefined()
  })

  describe('providerGroup', () => {
    it('maps a provider env var to its labeled group', () => {
      expect(providerGroup('XAI_API_KEY')).toBe('xAI')
      expect(providerGroup('NOUS_API_KEY')).toBe('Nous Portal')
      expect(providerGroup('OPENROUTER_API_KEY')).toBe('OpenRouter')
    })

    it('prefers the longest matching prefix so CN/regional buckets win', () => {
      // MINIMAX_CN_ must beat the generic MINIMAX_ prefix.
      expect(providerGroup('MINIMAX_CN_API_KEY')).toBe('MiniMax (China)')
      expect(providerGroup('MINIMAX_API_KEY')).toBe('MiniMax')
      // KIMI_CN_ likewise must beat KIMI_.
      expect(providerGroup('KIMI_CN_API_KEY')).toBe('Kimi (China)')
      expect(providerGroup('KIMI_API_KEY')).toBe('Kimi / Moonshot')
      // HERMES_QWEN_ and HERMES_GEMINI_ both share the HERMES_ stem.
      expect(providerGroup('HERMES_QWEN_BASE_URL')).toBe('DashScope (Qwen)')
      expect(providerGroup('HERMES_GEMINI_CLIENT_ID')).toBe('Gemini')
    })

    it('falls back to "Other" for un-grouped env vars', () => {
      expect(providerGroup('SOMETHING_RANDOM')).toBe('Other')
    })
  })
})

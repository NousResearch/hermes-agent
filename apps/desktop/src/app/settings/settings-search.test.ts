import { describe, expect, it } from 'vitest'

import { Settings2, Wrench } from '@/lib/icons'
import type { ConfigFieldSchema, HermesConfigRecord } from '@/types/hermes'

import {
  buildConfigSearchEntries,
  buildCredentialSearchEntries,
  credentialSettingsView,
  filterSettingsSearchEntries
} from './settings-search'
import { envVar } from './test-utils'

const searchCopy = {
  fieldDescriptions: {
    'display.personality': 'Choose how Hermes sounds in conversation.',
    'tts.edge.voice': 'Voice used by Edge TTS.'
  },
  fieldLabels: {
    'display.personality': 'Personality',
    'tts.edge.voice': 'Edge voice',
    'tts.openai.voice': 'OpenAI voice'
  },
  sections: {
    chat: 'Chat',
    voice: 'Voice'
  }
}

describe('settings search index', () => {
  it('keeps Unicode-equivalent queries and fields at the same rank without changing display text', () => {
    const entries = buildCredentialSearchEntries(
      { HANGUL_API_KEY: envVar('tool', { description: '한글 설명' }) },
      { settings: '설정', tools: '도구' },
      { settings: Settings2, tools: Wrench }
    )

    for (const form of ['NFC', 'NFD'] as const) {
      const label = '한글 검색'.normalize(form)
      const localized = { ...entries[0], context: '도구'.normalize(form), label }

      for (const query of ['한글 검색', '도구 한글', '한글 설명']) {
        for (const queryForm of ['NFC', 'NFD'] as const) {
          expect(filterSettingsSearchEntries([localized], query.normalize(queryForm))).toEqual([localized])
        }
      }

      expect(localized.label).toBe(label)
      expect(filterSettingsSearchEntries([localized], '한글 없는단어')).toEqual([])
    }
  })

  it('builds config results from renderable schema fields with exact deep links', () => {
    const schema: Record<string, ConfigFieldSchema> = {
      'display.personality': { type: 'select' },
      'tts.edge.voice': { type: 'string' },
      'tts.openai.voice': { type: 'string' }
    }

    const config = {
      display: { personality: 'default' },
      tts: { provider: 'edge', edge: { voice: '' }, openai: { voice: '' } }
    } as unknown as HermesConfigRecord

    const entries = buildConfigSearchEntries(schema, config, searchCopy)

    expect(entries.map(entry => entry.id)).toEqual([
      'config-field:display.personality',
      'config-field:tts.provider',
      'config-field:tts.edge.voice'
    ])
    expect(entries[0]).toMatchObject({
      context: 'Chat',
      description: 'Choose how Hermes sounds in conversation.',
      label: 'Personality',
      target: { field: 'display.personality', view: 'config:chat' }
    })
    expect(entries.some(entry => entry.id === 'config-field:tts.openai.voice')).toBe(false)
  })

  it('discovers future tool and setting entries entirely from backend metadata', () => {
    const vars = {
      FUTURE_CRAWLER_API_KEY: envVar('tool', {
        description: 'Fetch structured pages from a new crawler.',
        tools: ['future_crawl'],
        url: 'https://future.example/keys'
      }),
      FUTURE_GATEWAY_URL: envVar('setting', { description: 'Route gateway traffic.' }),
      TELEGRAM_BOT_TOKEN: envVar('messaging', { channel_managed: true }),
      MODEL_PROVIDER_API_KEY: envVar('provider')
    }

    const entries = buildCredentialSearchEntries(
      vars,
      { settings: 'Settings', tools: 'Tools' },
      { settings: Settings2, tools: Wrench }
    )

    expect(entries.map(entry => entry.id)).toEqual([
      'credential:FUTURE_CRAWLER_API_KEY',
      'credential:FUTURE_GATEWAY_URL'
    ])
    expect(entries[0]).toMatchObject({
      context: 'Tools',
      label: 'FUTURE CRAWLER',
      target: { key: 'FUTURE_CRAWLER_API_KEY', keysView: 'tools', view: 'keys' }
    })
    expect(filterSettingsSearchEntries(entries, 'structured crawler')).toHaveLength(1)
    expect(filterSettingsSearchEntries(entries, 'future_crawl')[0]?.id).toBe('credential:FUTURE_CRAWLER_API_KEY')
    expect(filterSettingsSearchEntries(entries, 'gateway traffic')[0]?.id).toBe('credential:FUTURE_GATEWAY_URL')
  })

  it('shares the Tools and Settings category boundary with the rendered page', () => {
    expect(credentialSettingsView(envVar('tool'))).toBe('tools')
    expect(credentialSettingsView(envVar('setting'))).toBe('settings')
    expect(credentialSettingsView(envVar('messaging'))).toBe('settings')
    expect(credentialSettingsView(envVar('messaging', { channel_managed: true }))).toBeNull()
    expect(credentialSettingsView(envVar('provider'))).toBeNull()
  })

  it('uses AND matching across labels, context, descriptions, and raw keys', () => {
    const entries = buildCredentialSearchEntries(
      {
        BRAVE_SEARCH_API_KEY: envVar('tool', { description: 'Search public web pages.' }),
        FIRECRAWL_API_KEY: envVar('tool', { description: 'Extract public web pages.' })
      },
      { settings: 'Settings', tools: 'Tools' },
      { settings: Settings2, tools: Wrench }
    )

    expect(filterSettingsSearchEntries(entries, 'brave tools')[0]?.id).toBe('credential:BRAVE_SEARCH_API_KEY')
    expect(filterSettingsSearchEntries(entries, 'firecrawl extract')[0]?.id).toBe('credential:FIRECRAWL_API_KEY')
    expect(filterSettingsSearchEntries(entries, 'brave extract')).toEqual([])
  })
})

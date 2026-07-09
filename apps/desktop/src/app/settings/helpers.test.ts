import { describe, expect, it } from 'vitest'

import type { HermesConfigRecord } from '@/types/hermes'

import { defineFieldCopy, fieldCopyForSchemaKey, schemaKeyToFieldCopyKey } from './field-copy'
import { enumOptionsFor, getNested, providerGroup, setNested, stripToolsetLabel, toolsetDisplayLabel } from './helpers'

describe('settings helpers', () => {
  it('lists Hindsight as a built-in desktop memory provider option', () => {
    const options = enumOptionsFor('memory.provider', '', {})

    expect(options).toContain('hindsight')
  })

  describe('defineFieldCopy', () => {
    it('flattens nested field copy paths', () => {
      const copy = defineFieldCopy({
        display: {
          personality: 'Personality'
        },
        stt: {
          elevenlabs: {
            language_code: 'Language'
          }
        }
      })

      expect(copy[['display', 'personality'].join('.')]).toBe('Personality')
      expect(copy[['stt', 'elevenlabs', 'language_code'].join('.')]).toBe('Language')
    })

    it('keeps top-level flat field keys', () => {
      expect(
        defineFieldCopy({
          model_context_length: 'Context Window',
          file_read_max_chars: 'File Read Limit'
        })
      ).toEqual({
        model_context_length: 'Context Window',
        file_read_max_chars: 'File Read Limit'
      })
    })

    it('maps schema keys to camelCase translation keys', () => {
      expect(schemaKeyToFieldCopyKey('model_context_length')).toBe('modelContextLength')
      expect(schemaKeyToFieldCopyKey('display.show_reasoning')).toBe('display.showReasoning')
      expect(schemaKeyToFieldCopyKey('tool_output.max_line_length')).toBe('toolOutput.maxLineLength')
      expect(schemaKeyToFieldCopyKey('updates.non_interactive_local_changes')).toBe(
        'updates.nonInteractiveLocalChanges'
      )
    })

    it('looks up camelCase field copy by schema key with legacy fallback', () => {
      const copy = defineFieldCopy({
        display: {
          showReasoning: 'Reasoning Blocks'
        },
        file_read_max_chars: 'Legacy File Read Limit',
        modelContextLength: 'Context Window',
        toolOutput: {
          maxLineLength: 'Line Length Limit'
        }
      })

      expect(fieldCopyForSchemaKey(copy, 'model_context_length')).toBe('Context Window')
      expect(fieldCopyForSchemaKey(copy, 'display.show_reasoning')).toBe('Reasoning Blocks')
      expect(fieldCopyForSchemaKey(copy, 'tool_output.max_line_length')).toBe('Line Length Limit')
      expect(fieldCopyForSchemaKey(copy, 'file_read_max_chars')).toBe('Legacy File Read Limit')
    })

    it('rejects duplicate flattened paths', () => {
      const duplicateKey = ['display', 'personality'].join('.')

      expect(() =>
        defineFieldCopy({
          display: {
            personality: 'Personality'
          },
          [duplicateKey]: 'Duplicate'
        })
      ).toThrow('Duplicate field copy key: display.personality')
    })
  })

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

  describe('stripToolsetLabel', () => {
    it('removes leading emoji prefixes from registry labels', () => {
      expect(stripToolsetLabel('⏰ Cron Jobs')).toBe('Cron Jobs')
      expect(stripToolsetLabel('⚡ Code Execution')).toBe('Code Execution')
      expect(stripToolsetLabel('❓ Clarifying Questions')).toBe('Clarifying Questions')
      expect(stripToolsetLabel('🌐 Browser Automation')).toBe('Browser Automation')
      expect(stripToolsetLabel('🎨 Image Generation')).toBe('Image Generation')
    })

    it('leaves plain titles unchanged', () => {
      expect(stripToolsetLabel('Terminal & Processes')).toBe('Terminal & Processes')
    })
  })

  describe('toolsetDisplayLabel', () => {
    it('strips emoji from toolset rows', () => {
      expect(toolsetDisplayLabel({ name: 'cronjob', label: '⏰ Cron Jobs' })).toBe('Cron Jobs')
    })
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
      // HERMES_QWEN_ shares the HERMES_ stem with other integrations.
      expect(providerGroup('HERMES_QWEN_BASE_URL')).toBe('DashScope (Qwen)')
      expect(providerGroup('GEMINI_API_KEY')).toBe('Gemini')
    })

    it('falls back to "Other" for un-grouped env vars', () => {
      expect(providerGroup('SOMETHING_RANDOM')).toBe('Other')
    })
  })

  describe('enumOptionsFor — backend selector dropdowns', () => {
    const config: HermesConfigRecord = {}

    it('renders a dropdown for the TTS provider including xAI (Grok)', () => {
      const opts = enumOptionsFor('tts.provider', 'edge', config)
      expect(opts).toBeDefined()
      expect(opts).toContain('xai')
      expect(opts).toContain('edge')
      expect(opts).toContain('elevenlabs')
    })

    it('renders a dropdown for the STT provider including xAI (Grok)', () => {
      const opts = enumOptionsFor('stt.provider', 'local', config)
      expect(opts).toEqual(['local', 'groq', 'openai', 'mistral', 'xai', 'elevenlabs'])
    })

    it('renders dropdowns for per-backend model/device sub-fields', () => {
      expect(enumOptionsFor('stt.openai.model', 'whisper-1', config)).toContain('gpt-4o-transcribe')
      expect(enumOptionsFor('tts.openai.model', 'gpt-4o-mini-tts', config)).toContain('tts-1-hd')
      expect(enumOptionsFor('tts.neutts.device', 'cpu', config)).toEqual(['cpu', 'cuda', 'mps'])
    })

    it('renders a dropdown for the terminal execution backend', () => {
      const opts = enumOptionsFor('terminal.backend', 'local', config)
      expect(opts).toEqual(['local', 'docker', 'singularity', 'modal', 'daytona', 'ssh'])
    })

    it('appends a hand-typed value not in the known list so it stays selected', () => {
      const opts = enumOptionsFor('tts.provider', 'my-custom-command-tts', config)
      expect(opts).toContain('my-custom-command-tts')
      expect(opts).toContain('xai')
    })
  })

  describe('enumOptionsFor — display.personality merges agent.personalities (#61312)', () => {
    it('returns the 14 built-in personalities when config has none', () => {
      const opts = enumOptionsFor('display.personality', '', {} as HermesConfigRecord)
      expect(opts).toEqual([
        '',
        'helpful',
        'concise',
        'technical',
        'creative',
        'teacher',
        'kawaii',
        'catgirl',
        'pirate',
        'shakespeare',
        'surfer',
        'noir',
        'uwu',
        'philosopher',
        'hype'
      ])
    })

    it('appends a single custom personality from agent.personalities', () => {
      const opts = enumOptionsFor('display.personality', '', {
        agent: {
          personalities: {
            bug_hunter: 'You are a bug bounty hunter. …'
          }
        }
      } as unknown as HermesConfigRecord)
      expect(opts).toContain('bug_hunter')
      expect(opts).toContain('helpful') // the built-ins remain
    })

    it('appends multiple custom personalities and de-duplicates', () => {
      const opts = enumOptionsFor('display.personality', '', {
        agent: {
          personalities: {
            bug_hunter: 'You are a bug bounty hunter. …',
            chef: 'You are a chef. …',
            pentester: 'pentest the world' // also appears hardcoded in JS bundles
          }
        }
      } as unknown as HermesConfigRecord)
      // Custom names appear once each.
      expect(opts?.filter((name) => name === 'bug_hunter')).toHaveLength(1)
      expect(opts?.filter((name) => name === 'chef')).toHaveLength(1)
      expect(opts?.filter((name) => name === 'pentester')).toHaveLength(1)
      // Built-ins remain present.
      expect(opts).toContain('helpful')
    })

    it('preserves the current personality even if neither built-in nor custom', () => {
      // The user hand-typed 'inherited-from-old-config' in a previous session;
      // the runtime must keep it visible so the dropdown doesn't go blank.
      const opts = enumOptionsFor('display.personality', 'inherited-from-old-config', {} as HermesConfigRecord)
      expect(opts).toContain('inherited-from-old-config')
      expect(opts).toContain('helpful')
    })

    it('treats agent.personalities as a non-object as if it were empty', () => {
      // A misconfigured config.yaml could surface a list/string instead of a
      // dict. The dropdown must not crash or surface [object Object].
      const opts = enumOptionsFor('display.personality', '', {
        agent: { personalities: ['not-a-dict'] as unknown }
      } as unknown as HermesConfigRecord)
      expect(opts).toContain('helpful')
    })
  })
})

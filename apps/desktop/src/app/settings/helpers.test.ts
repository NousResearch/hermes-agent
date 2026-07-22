import { describe, expect, it } from 'vitest'

import type { HermesConfigRecord } from '@/types/hermes'

import { FIELD_DESCRIPTIONS, FIELD_LABELS, SECTIONS } from './constants'
import { defineFieldCopy, fieldCopyForSchemaKey, schemaKeyToFieldCopyKey } from './field-copy'
import {
  enumOptionsFor,
  getNested,
  isExternalMemoryProvider,
  providerGroup,
  sectionFieldEntries,
  setNested,
  stripToolsetLabel,
  toolsetDisplayLabel
} from './helpers'

describe('settings helpers', () => {
  it('surfaces repository discovery config in Workspace with user-facing copy', () => {
    const workspace = SECTIONS.find(section => section.id === 'workspace')

    expect(workspace?.keys).toEqual(
      expect.arrayContaining([
        'desktop.repo_scan_enabled',
        'desktop.repo_scan_roots',
        'desktop.repo_scan_exclude_paths'
      ])
    )
    expect(fieldCopyForSchemaKey(FIELD_LABELS, 'desktop.repo_scan_enabled')).toBeTruthy()
    expect(fieldCopyForSchemaKey(FIELD_DESCRIPTIONS, 'desktop.repo_scan_exclude_paths')).toBeTruthy()
  })

  it('lists the desktop memory provider options in their declared order', () => {
    const options = enumOptionsFor('memory.provider', '', {})

    // Built-in memory is not a provider plugin; the empty sentinel is the
    // only built-in-shaped entry (#49513).
    expect(options).toEqual(['', 'honcho', 'hindsight'])
  })

  it('keeps a legacy literal builtin value visible as the current selection', () => {
    const options = enumOptionsFor('memory.provider', 'builtin', {})

    expect(options).toEqual(['', 'honcho', 'hindsight', 'builtin'])
  })

  it('prefers the discovered provider set over the static fallback for memory.provider', () => {
    // config-settings.tsx passes the live getMemoryStatus() providers as
    // dynamicOptions; user-installed/pip providers must appear, not just the
    // hardcoded honcho/hindsight subset.
    const discovered = ['', 'honcho', 'hindsight', 'mem0', 'supermemory']
    const options = enumOptionsFor('memory.provider', '', {}, discovered)

    expect(options).toEqual(discovered)
  })

  it('keeps the current memory.provider value visible even if discovery omits it', () => {
    // A provider selected in config but not (yet) returned by discovery must
    // still render as the current selection rather than silently vanishing.
    const discovered = ['', 'honcho']
    const options = enumOptionsFor('memory.provider', 'hindsight', {}, discovered)

    expect(options).toEqual(['', 'honcho', 'hindsight'])
  })

  describe('isExternalMemoryProvider', () => {
    it('treats only real plugin names as external providers', () => {
      expect(isExternalMemoryProvider('honcho')).toBe(true)
      expect(isExternalMemoryProvider('hindsight')).toBe(true)
    })

    it('treats built-in aliases and empty values as not external', () => {
      for (const value of ['', 'builtin', 'built-in', 'Builtin', 'none', '  ', undefined, null, 7]) {
        expect(isExternalMemoryProvider(value)).toBe(false)
      }
    })
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
      expect(providerGroup('FIREWORKS_API_KEY')).toBe('Fireworks AI')
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

    it('surfaces user-defined command-type TTS providers (canonical providers nesting + legacy)', () => {
      const withCustom: HermesConfigRecord = {
        tts: {
          provider: 'neutts',
          // canonical location the runtime resolves first: tts.providers.<name>
          providers: {
            higgs8: { type: 'command', command: 'curl …' },
            indextts2: { type: 'command', command: 'curl …' },
            // `type:` is optional at runtime — a bare command block still qualifies
            typeless: { command: 'curl …' },
            // misconfigured: type:command but no command → NOT a runtime provider
            noop: { type: 'command' }
          },
          // back-compat: a top-level tts.<name> command block still resolves at runtime
          mylegacy: { type: 'command', command: 'curl …' },
          // a non-command block (built-in config) must NOT be offered as a provider
          edge: { voice: 'en-US-JennyNeural' }
        }
      }

      const opts = enumOptionsFor('tts.provider', 'neutts', withCustom)
      expect(opts).toContain('higgs8') // canonical providers.<name>
      expect(opts).toContain('indextts2') // canonical providers.<name>
      expect(opts).toContain('typeless') // command block with no type: still surfaced
      expect(opts).toContain('mylegacy') // legacy top-level tts.<name>
      expect(opts).toContain('elevenlabs') // built-ins preserved
      expect(opts).not.toContain('noop') // type:command with no command is excluded
      // 'edge' appears once (the built-in), not duplicated by the config block
      expect(opts!.filter(o => o === 'edge')).toHaveLength(1)
      // the 'providers' container itself is never offered as a provider name
      expect(opts).not.toContain('providers')
    })

    it('surfaces command-type STT providers too (canonical providers nesting)', () => {
      const withCustom: HermesConfigRecord = {
        stt: {
          provider: 'local',
          providers: { myasr: { type: 'command', command: 'curl …' } }
        }
      }

      const opts = enumOptionsFor('stt.provider', 'local', withCustom)
      expect(opts).toContain('myasr')
      expect(opts).toContain('local')
      expect(opts).not.toContain('providers')
    })

    // The runtime rejects a built-in name as a command provider before any config
    // lookup, so such a block must never be offered — including the names the
    // display list omits (`deepinfra` for TTS; `deepinfra`/`local_command` for
    // STT), where filtering on ENUM_OPTIONS instead of the runtime's built-in set
    // would wrongly offer a provider that can never dispatch.
    it('never offers a built-in name as a command provider, even one absent from the dropdown list', () => {
      const shadowing: HermesConfigRecord = {
        tts: {
          provider: 'edge',
          providers: {
            // built-in and absent from ENUM_OPTIONS['tts.provider']
            deepinfra: { type: 'command', command: 'curl …' },
            // built-in guard is case-insensitive at runtime (provider.lower())
            EDGE: { type: 'command', command: 'curl …' },
            // a genuine custom provider alongside them still surfaces
            higgs8: { type: 'command', command: 'curl …' }
          }
        }
      }

      const opts = enumOptionsFor('tts.provider', 'edge', shadowing)
      expect(opts).not.toContain('deepinfra')
      expect(opts).not.toContain('EDGE')
      expect(opts).toContain('higgs8')
      expect(opts!.filter(o => o === 'edge')).toHaveLength(1)
    })

    it('never offers a built-in STT name absent from the dropdown list as a command provider', () => {
      const shadowing: HermesConfigRecord = {
        stt: {
          provider: 'local',
          providers: {
            // both are built-in STT names omitted from ENUM_OPTIONS['stt.provider']
            local_command: { type: 'command', command: 'curl …' },
            deepinfra: { type: 'command', command: 'curl …' },
            myasr: { type: 'command', command: 'curl …' }
          }
        }
      }

      const opts = enumOptionsFor('stt.provider', 'local', shadowing)
      expect(opts).not.toContain('local_command')
      expect(opts).not.toContain('deepinfra')
      expect(opts).toContain('myasr')
    })
  })

  describe('sectionFieldEntries', () => {
    it('renders memory.provider from config even when the backend schema omits it', () => {
      const schema = { 'memory.memory_enabled': { type: 'boolean' as const } }
      const config: HermesConfigRecord = { memory: { memory_enabled: true, provider: '' } }

      const memoryKeys = (sectionFieldEntries(schema, config).get('memory') ?? []).map(([key]) => key)

      expect(memoryKeys).toContain('memory.provider')
    })

    it('infers the field type from the config value when the schema omits the key', () => {
      const config: HermesConfigRecord = { memory: { provider: '', memory_enabled: true, memory_char_limit: 2200 } }

      const fields = new Map(sectionFieldEntries({}, config).get('memory') ?? [])

      expect(fields.get('memory.provider')?.type).toBe('string')
      expect(fields.get('memory.memory_enabled')?.type).toBe('boolean')
      expect(fields.get('memory.memory_char_limit')?.type).toBe('number')
    })

    it('prefers the backend schema entry over inference when both exist', () => {
      const schema = { 'memory.provider': { type: 'select' as const, options: ['honcho'] } }
      const config: HermesConfigRecord = { memory: { provider: 'honcho' } }

      const field = new Map(sectionFieldEntries(schema, config).get('memory') ?? []).get('memory.provider')

      expect(field?.type).toBe('select')
      expect(field?.options).toEqual(['honcho'])
    })

    it('hides declared keys absent from both schema and config', () => {
      expect(sectionFieldEntries({}, {}).get('memory') ?? []).toHaveLength(0)
    })
  })
})

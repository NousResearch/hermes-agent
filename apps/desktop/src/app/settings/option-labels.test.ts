import { describe, expect, it } from 'vitest'

import { optionLabelsForSchemaKey } from '@/app/settings/field-copy'
import { ar } from '@/i18n/ar'
import { en } from '@/i18n/en'
import { prettyName } from '@/lib/text'

// Mirrors the resolution order in config-field.tsx's closed-select branch:
//   dynamic (caller) ?? catalog (active locale) ?? prettyName (literal fallback).
// The dynamic map is absent here so we exercise catalog + fallback directly.
const resolve = (
  t: typeof en | typeof ar,
  schemaKey: string,
  option: string,
  dynamic?: Record<string, string>
) => {
  const catalog = optionLabelsForSchemaKey(t.settings.optionLabels, schemaKey)

  return dynamic?.[option] ?? catalog?.[option] ?? prettyName(option)
}

describe('option value localization', () => {
  it('(a) English renders prettyName as before (no map defined)', () => {
    expect(en.settings.optionLabels).toBeUndefined()
    expect(resolve(en, 'approvals.mode', 'off')).toBe('Off')
    expect(resolve(en, 'approvals.mode', 'smart')).toBe('Smart')
    expect(resolve(en, 'agent.image_input_mode', 'auto')).toBe('Auto')
    expect(resolve(en, 'display.personality', 'catgirl')).toBe('Catgirl')
  })

  it('(b) Arabic renders the label from the catalog map', () => {
    expect(resolve(ar, 'approvals.mode', 'off')).toBe('معطل')
    // raw schema key resolves against the camelCase catalog key via the middleware
    expect(resolve(ar, 'agent.image_input_mode', 'auto')).toBe('تلقائي')
    expect(resolve(ar, 'code_execution.mode', 'strict')).toBe('صارم')
    expect(resolve(ar, 'context.engine', 'compressor')).toBe('ضاغط')
    expect(resolve(ar, 'delegation.reasoning_effort', 'high')).toBe('عالٍ')
    expect(resolve(ar, 'display.personality', 'helpful')).toBe('متعاون')
    expect(resolve(ar, 'terminal.backend', 'local')).toBe('محلي')
  })

  it('(c) unknown option value falls through to prettyName', () => {
    // docker is intentionally absent from terminal.backend (brand) -> literal
    expect(resolve(ar, 'terminal.backend', 'docker')).toBe('Docker')
    // a field with no map at all (provider brands) -> literal even in Arabic
    expect(resolve(ar, 'tts.provider', 'elevenlabs')).toBe('Elevenlabs')
    // an unmapped personality value still prettifies
    expect(resolve(ar, 'display.personality', 'zzz')).toBe('Zzz')
  })

  it('dynamic caller labels win over catalog and prettyName', () => {
    const live = { xyz: 'صوتي المستنسخ' }
    expect(resolve(ar, 'tts.elevenlabs.voice_id', 'xyz', live)).toBe('صوتي المستنسخ')
  })
})

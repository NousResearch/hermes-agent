import { describe, expect, it } from 'vitest'

import { en } from './en'
import { ptBrOverrides } from './pt-br'

const BLUEPRINT_KEYS = [
  'bill-renewal-watch',
  'competitor-watch',
  'custom-reminder',
  'evening-winddown',
  'gratitude-journal',
  'habit-checkin',
  'hydration-move',
  'important-mail',
  'learn-daily',
  'meal-plan',
  'morning-brief',
  'news-digest',
  'on-this-day',
  'price-watch',
  'weekly-review',
  'workday-start'
]

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function expectCompleteCoverage(source: unknown, translation: unknown, path = 'root'): void {
  if (!isRecord(source)) {
    expect(translation, `Missing pt-BR translation for ${path}`).not.toBeUndefined()

    return
  }

  expect(isRecord(translation), `Missing pt-BR translation group for ${path}`).toBe(true)

  const sourceRecord = source as Record<string, unknown>
  const translationRecord = translation as Record<string, unknown>

  for (const key of Object.keys(translationRecord)) {
    // Blueprint metadata is backend-owned in English and localized only by
    // locales that provide a catalog overlay.
    if (path === 'root.cron.blueprints' && key === 'catalog') {
      continue
    }

    expect(sourceRecord, `Unexpected pt-BR translation key for ${path}.${key}`).toHaveProperty(key)
  }

  for (const [key, value] of Object.entries(source)) {
    expectCompleteCoverage(value, translationRecord[key], `${path}.${key}`)
  }
}

describe('pt-BR locale', () => {
  it('overrides every English translation key', () => {
    expectCompleteCoverage(en, ptBrOverrides)
  })

  it('localizes every backend automation blueprint currently supported by Desktop', () => {
    expect(Object.keys(ptBrOverrides.cron?.blueprints?.catalog ?? {}).sort()).toEqual(BLUEPRINT_KEYS)
  })
})

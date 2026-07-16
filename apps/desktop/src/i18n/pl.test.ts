import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

import { en } from './en'
import { pl } from './pl'

function ownLeafPaths(value: unknown, prefix = ''): string[] {
  if (
    typeof value === 'function' ||
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  ) {
    return [prefix]
  }

  if (Array.isArray(value)) {
    return value.flatMap((item, index) => ownLeafPaths(item, `${prefix}[${index}]`))
  }

  if (value && typeof value === 'object') {
    return Object.keys(value as Record<string, unknown>).flatMap(key =>
      ownLeafPaths((value as Record<string, unknown>)[key], prefix ? `${prefix}.${key}` : key)
    )
  }

  return [prefix]
}

describe('Polish desktop catalog', () => {
  it('has the same recursive own leaf paths as English', () => {
    expect(ownLeafPaths(pl).sort()).toEqual(ownLeafPaths(en).sort())
  })

  it('uses Polish for core visible actions', () => {
    expect(pl.common.save).toBe('Zapisz')
    expect(pl.common.cancel).toBe('Anuluj')
    expect(pl.common.delete).toBe('Usuń')
    expect(pl.settings.nav.providers).toBe('Dostawcy')
    expect(pl.boot.ready).toBe('Hermes Desktop jest gotowy')
    expect(pl.commandCenter.commandCenter).toBe('Centrum poleceń')
    expect(pl.commandCenter.generatePet.hatch).toBe('Wykluj')
    expect(pl.assistant.approval.run).toBe('Uruchom')
  })

  it('does not contain known literal-translation regressions', () => {
    const source = readFileSync(`${process.cwd()}/src/i18n/pl.ts`, 'utf8')

    const forbidden = [
      /modelk/i,
      /Bliźnięt/i,
      /\bBieganie\b/i,
      /\bBiegnij\b/i,
      /\bWłaz\b/i,
      /\bTarło\b/i,
      /Zremis/i,
      /\boddział/i,
      /żeton/i,
      /kompozytor/i,
      /zaplecz/i,
      /Pulpit Hermes/i,
      /Centrum dowodzenia/i,
      /\bbramk/i,
      /\bmonit(?:u|em|ach|ami|y|ów|owi|cie|owanie|owania)?\b/iu,
      /zachęt/i,
      /narzędzi\(a\)|serwera\(ów\)|umiejętność\(i\)/i
    ]

    for (const pattern of forbidden) {
      expect(source, `forbidden Polish localization pattern: ${pattern}`).not.toMatch(pattern)
    }
  })
})

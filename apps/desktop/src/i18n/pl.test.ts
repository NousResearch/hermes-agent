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
  })
})

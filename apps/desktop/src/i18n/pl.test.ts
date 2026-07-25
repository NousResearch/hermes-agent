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
    expect(pl.install.setupChoiceDesc).toBe(
      'Połącz tę aplikację z działającą bramą Hermesa albo zainstaluj Hermesa lokalnie na tym komputerze.'
    )
    expect(pl.install.connectExistingTitle).toBe('Połącz z istniejącą bramą Hermesa')
    expect(pl.install.connectExistingShort).toBe('Połącz z bramą')
    expect(pl.install.installLocalTitle).toBe('Zainstaluj Hermesa lokalnie')
    expect(pl.install.installLocalDesc).toBe(
      'Pobierz Hermesa, utwórz jego środowisko Python i uruchom backend na tym komputerze.'
    )
    expect(pl.install.remoteSetupTitle).toBe('Połącz z istniejącą bramą Hermesa')
  })
})

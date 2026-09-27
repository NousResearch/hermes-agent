import { expect, it } from 'vitest'

import { TRANSLATIONS } from './catalog'
import { ruOverrides } from './ru'

const leafPaths = (value: unknown, path = ''): string[] => {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return Object.entries(value).flatMap(([key, child]) => leafPaths(child, path ? `${path}.${key}` : key))
  }

  return [path]
}

it('keeps Russian overrides on every English desktop catalog leaf', () => {
  // Intro copy is shipped in JSONL, not authored in the English catalog.
  const english = new Set(leafPaths(TRANSLATIONS.en).filter(path => !path.startsWith('intro.')))
  const russian = new Set(leafPaths(ruOverrides))
  expect([...english].filter(path => !russian.has(path)).sort()).toEqual([])
})

it('keeps merged Russian plugin and fleet labels on the live nested paths', () => {
  for (const [russian, english] of [
    [TRANSLATIONS.ru.profiles.fleet.localDevice, TRANSLATIONS.en.profiles.fleet.localDevice],
    [TRANSLATIONS.ru.profiles.fleet.allOnGateway, TRANSLATIONS.en.profiles.fleet.allOnGateway],
    [TRANSLATIONS.ru.skills.plugins.pageBlurb, TRANSLATIONS.en.skills.plugins.pageBlurb],
    [TRANSLATIONS.ru.skills.plugins.agentTitle, TRANSLATIONS.en.skills.plugins.agentTitle]
  ]) {
    expect(russian).not.toBe(english)
  }
})

it('declines counts and retains interpolated plugin names and payment warnings', () => {
  const fact = TRANSLATIONS.ru.connectorsPage.card.fact.tools
  const reachable = TRANSLATIONS.ru.settings.customEndpoints.endpointReachableModels

  for (const [count, verb, tools, models] of [
    [1, 'Найдена', 'инструмент', 'модель'],
    [2, 'Найдено', 'инструмента', 'модели'],
    [5, 'Найдено', 'инструментов', 'моделей'],
    [11, 'Найдено', 'инструментов', 'моделей'],
    [12, 'Найдено', 'инструментов', 'моделей'],
    [13, 'Найдено', 'инструментов', 'моделей'],
    [14, 'Найдено', 'инструментов', 'моделей'],
    [22, 'Найдено', 'инструмента', 'модели'],
    [112, 'Найдено', 'инструментов', 'моделей'],
    [21, 'Найдена', 'инструмент', 'модель']
  ] as const) {
    expect(fact(count)).toBe(`${count} ${tools}`)
    expect(reachable('Доступно.', count)).toContain(`${verb} ${count} ${models}`)
  }

  const missingKey = TRANSLATIONS.ru.settings.plugins.installModal.missingEnv('Plugin Ω', 'VAR_SECRET')
  expect(missingKey).toContain('Plugin Ω')
  expect(missingKey).toContain('VAR_SECRET')
  expect(missingKey).toContain('не заработают')

  const warning = TRANSLATIONS.ru.settings.billing.charge.unconfirmedBody('Платёж обрабатывается.')
  expect(warning).toContain('Платёж обрабатывается.')
  expect(warning).toContain('прежде чем повторять попытку')
})

it('uses the instrumental case after the Russian starmap import preposition', () => {
  const importSuccess = TRANSLATIONS.ru.starmap.importSuccess

  for (const [count, noun] of [[1, 'узлом'], [2, 'узлами'], [5, 'узлами'], [11, 'узлами'], [12, 'узлами'], [13, 'узлами'], [14, 'узлами'], [21, 'узлом']] as const) {
    expect(importSuccess(count)).toContain(`с ${count} ${noun}`)
  }
})

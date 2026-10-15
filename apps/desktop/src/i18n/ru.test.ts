import { expect, it } from 'vitest'

import { TRANSLATIONS } from './catalog'

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

it('uses the instrumental case after the Russian starmap import preposition', () => {
  const importSuccess = TRANSLATIONS.ru.starmap.importSuccess
  for (const [count, noun] of [[1, 'узлом'], [2, 'узлами'], [5, 'узлами'], [11, 'узлами'], [21, 'узлом']] as const) {
    expect(importSuccess(count)).toContain(`с ${count} ${noun}`)
  }
})

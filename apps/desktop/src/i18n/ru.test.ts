import { expect, it } from 'vitest'

import { TRANSLATIONS } from './catalog'

it('uses the instrumental case after the Russian starmap import preposition', () => {
  const importSuccess = TRANSLATIONS.ru.starmap.importSuccess
  for (const [count, noun] of [[1, 'узлом'], [2, 'узлами'], [5, 'узлами'], [11, 'узлами'], [21, 'узлом']] as const) {
    expect(importSuccess(count)).toContain(`с ${count} ${noun}`)
  }
})

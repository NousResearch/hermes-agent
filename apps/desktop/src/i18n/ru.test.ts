import { describe, expect, it } from 'vitest'

import { ru } from './ru'

describe('russian locale catalog', () => {
  it('translates the live terminal-selection keybind and drops the retired key', () => {
    expect(ru.keybinds.actions['view.selectionToComposer']).toBe('Отправить выделенный текст в композер')
    expect(ru.keybinds.actions).not.toHaveProperty('view.terminalSelection')
  })

  it('translates the live auxiliary review task and drops the retired one', () => {
    expect(ru.settings.model.tasks.review).toEqual({ label: 'Ревью', hint: '/review — субагент-рецензент' })
    expect(ru.settings.model.tasks).not.toHaveProperty('web_extract')
  })

  it('declines the starmap node count in the instrumental case', () => {
    const importSuccess = ru.starmap.importSuccess

    expect(importSuccess(1)).toBe('Загружена карта с 1 узлом.')
    expect(importSuccess(2)).toBe('Загружена карта с 2 узлами.')
    expect(importSuccess(5)).toBe('Загружена карта с 5 узлами.')
    expect(importSuccess(11)).toBe('Загружена карта с 11 узлами.')
    expect(importSuccess(12)).toBe('Загружена карта с 12 узлами.')
    expect(importSuccess(21)).toBe('Загружена карта с 21 узлом.')
    expect(importSuccess(22)).toBe('Загружена карта с 22 узлами.')
  })

  it('picks the right plural form at 1/2/5/11/12/21/22 across count strings', () => {
    expect([1, 2, 5, 11, 12, 21, 22].map(ru.agents.agentsCount)).toEqual([
      '1 агент',
      '2 агента',
      '5 агентов',
      '11 агентов',
      '12 агентов',
      '21 агент',
      '22 агента'
    ])

    expect([1, 2, 5, 11, 12, 21, 22].map(ru.settings.about.minAgo)).toEqual([
      '1 минуту назад',
      '2 минуты назад',
      '5 минут назад',
      '11 минут назад',
      '12 минут назад',
      '21 минуту назад',
      '22 минуты назад'
    ])

    expect([1, 2, 5, 11, 12, 21, 22].map(ru.zones.tabCount)).toEqual([
      '1 вкладка',
      '2 вкладки',
      '5 вкладок',
      '11 вкладок',
      '12 вкладок',
      '21 вкладка',
      '22 вкладки'
    ])
  })
})

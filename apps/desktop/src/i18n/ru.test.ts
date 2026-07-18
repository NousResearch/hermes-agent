import { describe, expect, it } from 'vitest'

import { ru } from './ru'

describe('Russian locale', () => {
  it.each([
    [0, '0 профилей'],
    [1, '1 профиль'],
    [2, '2 профиля'],
    [5, '5 профилей'],
    [11, '11 профилей'],
    [21, '21 профиль'],
    [22, '22 профиля'],
    [25, '25 профилей']
  ])('declines profile counts for %i', (count, expected) => {
    expect(ru.profiles.count(count)).toBe(expected)
  })

  it('declines counts consistently across independent interface sections', () => {
    expect(ru.agents.workers(11)).toBe('11 исполнителей')
    expect(ru.agents.workers(21)).toBe('21 исполнитель')
    expect(ru.settings.sessions.messages(2)).toBe('2 сообщения')
    expect(ru.settings.sessions.messages(21)).toBe('21 сообщение')
    expect(ru.cron.count(22)).toBe('22 задачи')
    expect(ru.preview.console.sentMessage(1)).toBe('В поле ввода добавлено 1 запись журнала')
    expect(ru.preview.console.sentMessage(22)).toBe('В поле ввода добавлено 22 записи журнала')
  })
})

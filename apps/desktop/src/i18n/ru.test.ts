import { describe, expect, it } from 'vitest'

import { en } from './en'
import { ru, ruOverrides } from './ru'

function leafPaths(value: unknown, prefix = ''): string[] {
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => leafPaths(item, `${prefix}[${index}]`))
  }

  if (typeof value === 'object' && value !== null) {
    return Object.entries(value).flatMap(([key, item]) => leafPaths(item, prefix ? `${prefix}.${key}` : key))
  }

  return [prefix]
}

describe('Russian locale', () => {
  it('overrides every English catalog entry', () => {
    expect(leafPaths(ruOverrides).sort()).toEqual(leafPaths(en).sort())
  })

  it.each([
    [0, '0 профилей'],
    [1, '1 профиль'],
    [2, '2 профиля'],
    [5, '5 профилей'],
    [11, '11 профилей'],
    [12, '12 профилей'],
    [13, '13 профилей'],
    [14, '14 профилей'],
    [21, '21 профиль'],
    [22, '22 профиля'],
    [25, '25 профилей'],
    [111, '111 профилей'],
    [112, '112 профилей'],
    [113, '113 профилей'],
    [114, '114 профилей']
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
    expect(ru.desktop.skillCommandsAvailable(1)).toBe('Доступна 1 команда навыка.')
    expect(ru.desktop.skillCommandsAvailable(2)).toBe('Доступны 2 команды навыков.')
    expect(ru.desktop.skillCommandsAvailable(5)).toBe('Доступно 5 команд навыков.')
    expect(ru.desktop.skillCommandsAvailable(11)).toBe('Доступно 11 команд навыков.')
    expect(ru.desktop.skillCommandsAvailable(21)).toBe('Доступна 21 команда навыка.')
    expect(ru.artifactCard.generating(11)).toBe('Создание… 11 строк')
    expect(ru.artifactCard.generating(21)).toBe('Создание… 21 строка')
    expect(ru.artifactCard.versionBadge(12)).toBe('12 версий')
    expect(ru.artifactCard.versionBadge(22)).toBe('22 версии')
    expect(ru.messaging.pendingAria(1)).toBe('1 запрос на доступ ожидает подтверждения')
    expect(ru.messaging.pendingAria(2)).toBe('2 запроса на доступ ожидают подтверждения')
    expect(ru.messaging.pendingAria(11)).toBe('11 запросов на доступ ожидают подтверждения')
    expect(ru.messaging.pendingAria(21)).toBe('21 запрос на доступ ожидает подтверждения')
    expect(ru.assistant.thread.filesChanged(1)).toBe('1 файл изменён')
    expect(ru.assistant.thread.filesChanged(2)).toBe('2 файла изменены')
    expect(ru.assistant.thread.filesChanged(11)).toBe('11 файлов изменено')
    expect(ru.assistant.thread.filesChanged(21)).toBe('21 файл изменён')
  })

  it('keeps pending actions compatible with their rendered status', () => {
    for (const title of Object.values(ru.assistant.tool.titles)) {
      expect(title.pending).toContain(title.pendingAction)
    }
  })
})

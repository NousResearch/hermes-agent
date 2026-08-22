import { describe, expect, it } from 'vitest'

import { countRu, pluralRu } from './ru-plural'

const taskForms = {
  one: 'задача',
  few: 'задачи',
  many: 'задач',
  other: 'задачи'
}

describe('Russian plural forms', () => {
  it.each([
    [0, 'задач'],
    [1, 'задача'],
    [2, 'задачи'],
    [5, 'задач'],
    [11, 'задач'],
    [14, 'задач'],
    [21, 'задача'],
    [22, 'задачи'],
    [25, 'задач'],
    [101, 'задача'],
    [111, 'задач']
  ])('selects the form for %i', (count, expected) => {
    expect(pluralRu(count, taskForms)).toBe(expected)
  })

  it('returns a complete counter', () => {
    expect(countRu(22, taskForms)).toBe('22 задачи')
  })
})

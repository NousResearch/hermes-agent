import { describe, expect, it } from 'vitest'

import { TASKS_ROUTE } from '../routes'

import { hasBuiltinRoutePage } from './route-tile'

describe('route tiles', () => {
  it('supports Live Tasks as a built-in split page', () => {
    expect(hasBuiltinRoutePage(TASKS_ROUTE)).toBe(true)
  })
})

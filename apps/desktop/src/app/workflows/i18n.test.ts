import { describe, expect, it } from 'vitest'

import { WORKFLOW_COPY } from './i18n'

function keyShape(value: unknown): unknown {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return typeof value
  }

  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => [key, keyShape(child)])
  )
}

describe('workflow i18n copy', () => {
  it('keeps English and Chinese key coverage aligned', () => {
    expect(keyShape(WORKFLOW_COPY.zh)).toEqual(keyShape(WORKFLOW_COPY.en))
  })
})

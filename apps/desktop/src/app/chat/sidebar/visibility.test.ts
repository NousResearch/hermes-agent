import { describe, expect, it } from 'vitest'

import { shouldShowFlatSessionsWithProjects } from './visibility'

describe('sidebar section visibility', () => {
  it('keeps the flat sessions roster visible below the projects overview', () => {
    expect(shouldShowFlatSessionsWithProjects(true, false)).toBe(true)
  })

  it('does not duplicate the flat roster while viewing a single project', () => {
    expect(shouldShowFlatSessionsWithProjects(true, true)).toBe(false)
  })

  it('does not add a second roster when projects grouping is off', () => {
    expect(shouldShowFlatSessionsWithProjects(false, false)).toBe(false)
  })
})

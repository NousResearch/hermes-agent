import { describe, expect, it } from 'vitest'

import { parseListFieldDraft } from './config-field'

describe('parseListFieldDraft', () => {
  it('commits comma-separated list values after editing', () => {
    expect(parseListFieldDraft('C:/gitLibraries, C:/gitProjects')).toEqual(['C:/gitLibraries', 'C:/gitProjects'])
  })

  it('trims entries and drops empty segments', () => {
    expect(parseListFieldDraft(' one, , two,   ')).toEqual(['one', 'two'])
  })

  it('allows an unfinished trailing comma to remain draft-only until commit', () => {
    expect(parseListFieldDraft('first,')).toEqual(['first'])
  })
})

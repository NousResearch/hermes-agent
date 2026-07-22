import { describe, expect, it } from 'vitest'

import { resolveTitlebarEdgeTargets } from './titlebar-controls'

describe('resolveTitlebarEdgeTargets', () => {
  it('keeps the original physical mapping in unflipped left-to-right layouts', () => {
    expect(resolveTitlebarEdgeTargets(false, false)).toEqual({
      left: 'sessions',
      right: 'files'
    })
  })

  it('mirrors the physical mapping in flipped left-to-right layouts', () => {
    expect(resolveTitlebarEdgeTargets(false, true)).toEqual({
      left: 'files',
      right: 'sessions'
    })
  })

  it('mirrors the physical mapping in unflipped right-to-left layouts', () => {
    expect(resolveTitlebarEdgeTargets(true, false)).toEqual({
      left: 'files',
      right: 'sessions'
    })
  })

  it('restores the original physical mapping in flipped right-to-left layouts', () => {
    expect(resolveTitlebarEdgeTargets(true, true)).toEqual({
      left: 'sessions',
      right: 'files'
    })
  })
})

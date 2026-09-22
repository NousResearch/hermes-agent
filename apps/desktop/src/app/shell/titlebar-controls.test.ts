import { describe, expect, it } from 'vitest'

import { unreadBadgeForEdge } from './titlebar-controls'

describe('unreadBadgeForEdge', () => {
  it('does not put an unread badge on a visible sidebar hide control', () => {
    expect(unreadBadgeForEdge('left', false, true, 3, true, true)).toBeUndefined()
  })

  it('puts the count on the control that reveals a hidden Sessions sidebar', () => {
    expect(unreadBadgeForEdge('left', false, false, 3, true, true)).toBe(3)
    expect(unreadBadgeForEdge('right', false, false, 3, true, true)).toBeUndefined()
  })

  it('follows the Sessions sidebar when pane sides are flipped', () => {
    expect(unreadBadgeForEdge('left', true, false, 3, true, true)).toBeUndefined()
    expect(unreadBadgeForEdge('right', true, false, 3, true, true)).toBe(3)
  })

  it('hides empty counts', () => {
    expect(unreadBadgeForEdge('left', false, false, 0, true, true)).toBeUndefined()
  })

  it.each([
    { edgeOpen: true, flipped: false },
    { edgeOpen: false, flipped: false },
    { edgeOpen: false, flipped: true }
  ])('never badges a sidebar control in the Bots workspace', ({ edgeOpen, flipped }) => {
    expect(unreadBadgeForEdge(flipped ? 'right' : 'left', flipped, edgeOpen, 3, false, false)).toBeUndefined()
  })

  it.each([
    { edgeOpen: true, flipped: false },
    { edgeOpen: false, flipped: false },
    { edgeOpen: false, flipped: true }
  ])('never badges while Bots or Terminal owns the shared sidebar group', ({ edgeOpen, flipped }) => {
    expect(unreadBadgeForEdge(flipped ? 'right' : 'left', flipped, edgeOpen, 3, true, false)).toBeUndefined()
  })
})

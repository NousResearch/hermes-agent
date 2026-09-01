import { beforeEach, describe, expect, it, vi } from 'vitest'

const { revealTreePane } = vi.hoisted(() => ({ revealTreePane: vi.fn() }))

vi.mock('@/components/pane-shell/tree/store', () => ({ revealTreePane }))

import { $routeTiles, closeRouteTile, openRouteTile } from './route-tiles'

describe('route tile lifecycle', () => {
  beforeEach(() => {
    $routeTiles.set([])
    revealTreePane.mockClear()
  })

  it('reveals one stable tile when a route is activated repeatedly', () => {
    openRouteTile('/kanban')
    openRouteTile('/kanban', 'left')

    expect($routeTiles.get()).toEqual([{ dir: 'right', path: '/kanban' }])
    expect(revealTreePane).toHaveBeenCalledTimes(2)
    expect(revealTreePane).toHaveBeenLastCalledWith('route-tile:/kanban')
  })

  it('canonicalizes query and hash before storing or revealing a route tile', () => {
    openRouteTile('/kanban?view=board#today')

    expect($routeTiles.get()).toEqual([{ dir: 'right', path: '/kanban' }])
    expect(revealTreePane).toHaveBeenLastCalledWith('route-tile:/kanban')
  })

  it('closes and reopens the same route without retaining a duplicate', () => {
    openRouteTile('/llm-usage')
    closeRouteTile('/llm-usage')
    openRouteTile('/llm-usage')

    expect($routeTiles.get()).toEqual([{ dir: 'right', path: '/llm-usage' }])
    expect(revealTreePane).toHaveBeenLastCalledWith('route-tile:/llm-usage')
  })
})

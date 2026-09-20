import { describe, expect, test } from 'vitest'

import {
  equalizeSplitWeights,
  equalizeSplitWeightsInTree,
  type GroupNode,
  type LayoutNode,
  type SplitNode
} from './model'

const group = (id: string): GroupNode => ({ type: 'group', id, panes: [id], active: id })

const split = (id: string, orientation: 'column' | 'row', children: LayoutNode[], weights: number[]): SplitNode => ({
  type: 'split',
  id,
  orientation,
  children,
  weights
})

describe('pane distribution', () => {
  test('equalizes selected direct tracks while preserving other tracks', () => {
    const node = split('root', 'row', [group('fixed'), group('left'), group('right')], [1, 1, 2])

    const result = equalizeSplitWeights(node, new Set(['left', 'right']))

    expect(result.weights).toEqual([1, 1.5, 1.5])
  })

  test('uses the eligible weight total instead of the whole split total', () => {
    const node = split('root', 'row', [group('fixed'), group('left'), group('new')], [1, 0.5, 1])

    const result = equalizeSplitWeights(node, new Set(['left', 'new']))

    expect(result.weights).toEqual([1, 0.75, 0.75])
  })

  test('equalizes every selected nested split independently', () => {
    const nested = split('nested', 'column', [group('top-left'), group('top-right')], [1, 3])
    const root = split('root', 'row', [nested, group('bottom')], [2, 1])

    const result = equalizeSplitWeightsInTree(
      root,
      new Map([
        ['root', new Set(['nested', 'bottom'])],
        ['nested', new Set(['top-left', 'top-right'])]
      ])
    )

    expect(result.type).toBe('split')

    if (result.type !== 'split') {
      return
    }

    expect(result.weights).toEqual([1.5, 1.5])
    expect(result.children[0].type).toBe('split')

    if (result.children[0].type !== 'split') {
      return
    }

    expect(result.children[0].weights).toEqual([2, 2])
  })

  test('leaves a split unchanged when fewer than two tracks are eligible', () => {
    const node = split('root', 'row', [group('only'), group('other')], [0.25, 0.75])

    expect(equalizeSplitWeights(node, new Set(['only']))).toBe(node)
    expect(equalizeSplitWeights(node, new Set())).toBe(node)
  })
})

import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import { group, split } from '../model'
import { $collapsedTreeSides, $hiddenTreePanes } from '../store'

import { TreeSplit } from './tree-split'

const disposers: (() => void)[] = []

beforeAll(() => {
  class TestResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }

  globalThis.ResizeObserver ??= TestResizeObserver as unknown as typeof ResizeObserver
})

beforeEach(() => {
  $collapsedTreeSides.set(new Set())
  $hiddenTreePanes.set(new Set())

  disposers.push(
    registry.register({
      area: 'panes',
      data: { placement: 'left', maxWidth: '18rem', minWidth: '15rem', width: '15rem' },
      id: 'sessions-motion',
      render: () => null,
      title: 'Sessions'
    }),
    registry.register({
      area: 'panes',
      data: { placement: 'main', uncloseable: true },
      id: 'workspace-motion',
      render: () => null,
      title: 'Workspace'
    })
  )
})

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  $collapsedTreeSides.set(new Set())
  $hiddenTreePanes.set(new Set())
})

describe('TreeSplit sidebar motion', () => {
  it('keeps a collapsed sidebar mounted while its flex track contracts', () => {
    const node = split(
      'row',
      [
        group(['sessions-motion'], { active: 'sessions-motion', id: 'sessions-track' }),
        group(['workspace-motion'], { active: 'workspace-motion', id: 'workspace-track' })
      ],
      [1, 1],
      'motion-root'
    )

    const { container } = render(<TreeSplit node={node} root rootRow />)
    const track = container.querySelector<HTMLElement>('[data-tree-track="sessions-track"]')!

    expect(track.className).toContain('transition-[flex-basis,flex-grow,opacity]')
    expect(track.style.display).toBe('')

    act(() => {
      $collapsedTreeSides.set(new Set(['left']))
    })

    expect(track.dataset.collapsed).toBe('true')
    expect(track.style.flex).toBe('0 0 0px')
    expect(track.style.opacity).toBe('0')
    expect(track.style.display).toBe('')
    expect(track.hasAttribute('inert')).toBe(true)

    act(() => {
      $collapsedTreeSides.set(new Set())
    })

    expect(track.dataset.collapsed).toBeUndefined()
    expect(track.style.flex).not.toBe('0 0 0px')
    expect(track.hasAttribute('inert')).toBe(false)
  })
})

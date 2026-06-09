import { cleanup, render } from '@testing-library/react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { PaneNode } from '@/lib/terminal-store'

const mounts = new Map<string, number>()
const unmounts = new Map<string, number>()

vi.mock('./terminal-pane', () => ({
  TerminalPane: ({ leafId }: { leafId: string }) => {
    useEffect(() => {
      mounts.set(leafId, (mounts.get(leafId) ?? 0) + 1)

      return () => {
        unmounts.set(leafId, (unmounts.get(leafId) ?? 0) + 1)
      }
    }, [leafId])

    return <div data-testid={leafId} />
  }
}))

import { SplitLayout } from './split-layout'

const noop = () => undefined

function renderLayout(node: PaneNode) {
  return (
    <SplitLayout
      cwd="/workspace"
      focusedLeafId="first"
      node={node}
      onAddSelectionToChat={noop}
      onClosePane={noop}
      onFocusPane={noop}
      onResize={noop}
    />
  )
}

describe('SplitLayout terminal lifecycle', () => {
  afterEach(() => {
    cleanup()
    mounts.clear()
    unmounts.clear()
  })

  it('keeps an existing terminal mounted when its pane is split', () => {
    const rendered = render(renderLayout({ id: 'first', type: 'leaf' }))

    rendered.rerender(
      renderLayout({
        direction: 'horizontal',
        first: { id: 'first', type: 'leaf' },
        ratio: 0.5,
        second: { id: 'second', type: 'leaf' },
        type: 'split'
      })
    )

    expect(mounts.get('first')).toBe(1)
    expect(unmounts.get('first')).toBeUndefined()
    expect(mounts.get('second')).toBe(1)
  })
})

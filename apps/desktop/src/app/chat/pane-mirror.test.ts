import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/components/pane-shell/tree/parked-panes', () => ({ setTreePaneParked: vi.fn() }))
vi.mock('@/components/pane-shell/tree/store', () => ({
  adoptContributedPanes: vi.fn(),
  registerPaneCloser: vi.fn(),
  removeTreePane: vi.fn(),
  treePanesWithPrefix: vi.fn(() => [])
}))
vi.mock('@/contrib/registry', () => ({ registry: { register: vi.fn(() => () => undefined) } }))

import { registry } from '@/contrib/registry'

import { paneMirror } from './pane-mirror'

type Register = { mock: { calls: Array<[{ id: string; render: () => unknown; title: string }]> } }

const calls = () => (registry.register as unknown as Register).mock.calls.map(([c]) => c)

describe('paneMirror', () => {
  beforeEach(() => vi.mocked(registry.register).mockClear())

  it('keeps the same render fn when a title refresh re-registers the tile', () => {
    // ContribRender mounts `render` as a component type: a new fn identity
    // remounts the tile's whole body (composer, live voice call, caret).
    const source = atom([{ id: 'a' }])
    const titles = atom<Record<string, string>>({ a: 'New session' })

    paneMirror<{ id: string }>({
      also: [titles],
      close: () => undefined,
      key: t => t.id,
      minWidth: '20rem',
      prefix: 'tile',
      render: key => key,
      source,
      title: key => titles.get()[key]
    })()

    titles.set({ a: 'Renamed' })

    const [first, second] = calls()

    expect(second.title).toBe('Renamed')
    expect(second.render).toBe(first.render)
    expect(second.render()).toBe('a')
  })

  it('starts a fresh render fn after the tile is disposed and reopened', () => {
    const source = atom([{ id: 'a' }])

    paneMirror<{ id: string }>({
      close: () => undefined,
      key: t => t.id,
      minWidth: '20rem',
      prefix: 'tile',
      render: key => key,
      source,
      title: () => 'A'
    })()

    source.set([])
    source.set([{ id: 'a' }])

    const [first, reopened] = calls()

    expect(reopened.render).not.toBe(first.render)
  })
})

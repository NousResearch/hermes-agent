import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  register: vi.fn(),
  registerPaneCloser: vi.fn(),
  removeTreePane: vi.fn(),
  treePanesWithPrefix: vi.fn((): string[] => [])
}))

vi.mock('@/contrib/registry', () => ({ registry: { register: mocks.register } }))
vi.mock('@/components/pane-shell/tree/store', () => ({
  registerPaneCloser: mocks.registerPaneCloser,
  removeTreePane: mocks.removeTreePane,
  treePanesWithPrefix: mocks.treePanesWithPrefix
}))

import { paneMirror } from './pane-mirror'

describe('paneMirror lifecycle', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.treePanesWithPrefix.mockReturnValue([])
  })

  it('registers dynamic panes and disposes plus prunes them when the source disappears', () => {
    const source = atom<{ id: string; title: string }[]>([])
    const dispose = vi.fn()
    const close = vi.fn()
    mocks.register.mockReturnValue(dispose)

    const watch = paneMirror({
      source,
      key: item => item.id,
      prefix: 'generated-view',
      minWidth: '20rem',
      title: id => source.get().find(item => item.id === id)?.title ?? id,
      render: id => <div>{id}</div>,
      close
    })

    watch()
    source.set([{ id: 'usage', title: 'Usage' }])

    expect(mocks.register).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'generated-view:usage', area: 'panes', title: 'Usage' })
    )
    expect(mocks.registerPaneCloser).toHaveBeenCalledWith('generated-view:usage', expect.any(Function))

    source.set([])

    expect(dispose).toHaveBeenCalledTimes(1)
    expect(mocks.removeTreePane).toHaveBeenCalledWith('generated-view:usage')
  })

  it('prunes a stale persisted pane that was never registered this session', () => {
    mocks.treePanesWithPrefix.mockReturnValueOnce(['generated-view:deleted'])
    const source = atom<{ id: string }[]>([])

    paneMirror({
      source,
      key: item => item.id,
      prefix: 'generated-view',
      minWidth: '20rem',
      title: id => id,
      render: id => <div>{id}</div>,
      close: vi.fn()
    })()

    expect(mocks.removeTreePane).toHaveBeenCalledWith('generated-view:deleted')
  })
})

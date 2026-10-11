import { cleanup, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { StarmapView } from './index'

vi.mock('@/store/starmap', async () => {
  const { atom } = await import('nanostores')

  return {
    $starmapError: atom<null | string>(null),
    $starmapGraph: atom(null),
    $starmapLoading: atom(false),
    loadStarmapGraph: vi.fn()
  }
})
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: { starmap: { close: 'Close', emptyDesc: '', emptyTitle: '', loadFailed: '', loading: '' } }
  })
}))
vi.mock('../overlays/panel', () => ({
  Panel: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  PanelEmpty: () => null
}))
vi.mock('@/components/page-loader', () => ({ PageLoader: () => null }))
vi.mock('./star-map', () => ({ StarMap: () => null }))

import { loadStarmapGraph } from '@/store/starmap'

afterEach(cleanup)

describe('StarmapView', () => {
  it('force-refetches the graph when the panel opens', () => {
    render(<StarmapView onClose={() => {}} />)

    // The store cache only resets on a profile switch; without `force` a
    // reopened panel keeps showing data from the renderer's first open.
    expect(vi.mocked(loadStarmapGraph)).toHaveBeenCalledWith(true)
  })
})

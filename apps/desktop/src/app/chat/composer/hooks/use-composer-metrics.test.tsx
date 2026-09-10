import { renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { useComposerMetrics } from './use-composer-metrics'

vi.mock('@assistant-ui/react', () => ({
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) => selector({ composer: { text: '' } })
}))

const emptyRef = { current: null }

describe('useComposerMetrics — local draft edges', () => {
  it('stacks a multiline local draft even when the shared runtime is empty', () => {
    const { result } = renderHook(() =>
      useComposerMetrics({
        composerDockRef: emptyRef,
        composerRef: emptyRef,
        composerSurfaceRef: emptyRef,
        editorRef: emptyRef,
        hasHardNewline: true,
        isEmpty: false,
        poppedOut: false
      })
    )

    expect(result.current.stacked).toBe(true)
  })
})

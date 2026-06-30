import { renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { useComposerDrop } from './use-composer-drop'

const fileList = (file: File) => ({
  item: (index: number) => (index === 0 ? file : null),
  length: 1
})

describe('useComposerDrop', () => {
  it('claims composer form drops so the outer chat drop zone does not attach the same file again', () => {
    const onAttachDroppedItems = vi.fn()
    const stopPropagation = vi.fn()
    const event = {
      dataTransfer: {
        files: fileList(new File(['png'], 'screenshot.png', { type: 'image/png' })),
        getData: () => '',
        items: undefined
      },
      preventDefault: vi.fn(),
      stopPropagation
    }

    const { result } = renderHook(() =>
      useComposerDrop({
        cwd: '/repo',
        insertInlineRefs: () => false,
        onAttachDroppedItems,
        requestMainFocus: vi.fn()
      })
    )

    result.current.handleDrop(event as never)

    expect(stopPropagation).toHaveBeenCalledOnce()
    expect(onAttachDroppedItems).toHaveBeenCalledOnce()
  })
})

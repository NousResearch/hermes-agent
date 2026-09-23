import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { useFileDropZone } from './use-file-drop-zone'

function dropEvent(files: File[]) {
  return {
    dataTransfer: {
      files: {
        item: (index: number) => files[index] ?? null,
        length: files.length
      },
      items: [],
      types: []
    },
    preventDefault: vi.fn()
  }
}

describe('useFileDropZone drop handler', () => {
  it('dispatches OS files that are only exposed in DataTransfer.files at drop time', () => {
    const onDropFiles = vi.fn()
    const { result } = renderHook(() => useFileDropZone({ onDropFiles }))
    const file = new File(['report'], 'report.txt', { type: 'text/plain' })
    const event = dropEvent([file])

    act(() => result.current.dropHandlers.onDrop(event as never))

    expect(event.preventDefault).toHaveBeenCalledOnce()
    expect(onDropFiles).toHaveBeenCalledWith([{ file, path: '' }])
  })

  it('ignores a transfer with no files or attachment metadata', () => {
    const onDropFiles = vi.fn()
    const { result } = renderHook(() => useFileDropZone({ onDropFiles }))
    const event = dropEvent([])

    act(() => result.current.dropHandlers.onDrop(event as never))

    expect(event.preventDefault).not.toHaveBeenCalled()
    expect(onDropFiles).not.toHaveBeenCalled()
  })
})

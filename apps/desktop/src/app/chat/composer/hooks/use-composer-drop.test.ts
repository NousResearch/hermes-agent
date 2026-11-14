import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { type DroppedFile, HERMES_PATHS_MIME } from '../../hooks/use-composer-actions'

import { useComposerDrop } from './use-composer-drop'

// #128165: some cross-app drags present no attachment metadata in
// DataTransfer.types at drop time (no HERMES_PATHS_MIME, no 'Files') yet
// still expose files through DataTransfer.files/items. The input drop
// handler used to gate on the type metadata and silently ignore the drop
// while the dropzone overlay highlighted it. Acceptance must be decided by
// the extracted payload instead.
function metadatalessDropEvent(file: File) {
  return {
    dataTransfer: {
      files: {
        item: (index: number) => (index === 0 ? file : null),
        length: 1
      },
      getData: (type: string) => (type === 'text/plain' ? 'pasted text' : ''),
      items: [{ kind: 'file', getAsFile: () => file, webkitGetAsEntry: () => null }],
      types: ['text/plain']
    },
    preventDefault: vi.fn(),
    stopPropagation: vi.fn()
  }
}

function renderComposerDrop(onAttachDroppedItems: (files: DroppedFile[]) => boolean | void | Promise<boolean | void>) {
  return renderHook(() =>
    useComposerDrop({
      cwd: '/repo',
      insertInlineRefs: vi.fn(() => false),
      onAttachDroppedItems,
      recordUndoPoint: vi.fn(),
      requestMainFocus: vi.fn(),
      sessionId: null
    })
  )
}

describe('useComposerDrop input drop acceptance', () => {
  it('attaches a file whose drag metadata under-reports (no custom MIME, no Files type)', () => {
    const onAttachDroppedItems = vi.fn()
    const { result } = renderComposerDrop(onAttachDroppedItems)
    const file = new File(['x'], 'photo.png', { type: 'image/png' })
    const event = metadatalessDropEvent(file)

    act(() => result.current.handleInputDrop(event as never))

    // OS drops route through the upload pipeline: partitionDroppedFiles puts a
    // File-bearing, path-less drop into osDrops.
    expect(onAttachDroppedItems).toHaveBeenCalledTimes(1)
    expect(onAttachDroppedItems.mock.calls[0][0]).toEqual([{ file, path: '' }])
    expect(event.preventDefault).toHaveBeenCalled()
  })

  it('still ignores a text-only drag and banks the undo snapshot', () => {
    const onAttachDroppedItems = vi.fn()
    const recordUndoPoint = vi.fn()

    const { result } = renderHook(() =>
      useComposerDrop({
        cwd: '/repo',
        insertInlineRefs: vi.fn(() => false),
        onAttachDroppedItems,
        recordUndoPoint,
        requestMainFocus: vi.fn(),
        sessionId: null
      })
    )

    const event = {
      dataTransfer: {
        files: { item: () => null, length: 0 },
        getData: () => 'text',
        items: [{ kind: 'string', getAsFile: () => null, webkitGetAsEntry: () => null }],
        types: ['text/plain']
      },
      preventDefault: vi.fn(),
      stopPropagation: vi.fn()
    }

    act(() => result.current.handleInputDrop(event as never))

    expect(onAttachDroppedItems).not.toHaveBeenCalled()
    expect(event.preventDefault).not.toHaveBeenCalled()
    expect(recordUndoPoint).toHaveBeenCalled()
  })

  it('attaches an in-app project-tree drag that carries the custom MIME', () => {
    const onAttachDroppedItems = vi.fn()
    const insertInlineRefs = vi.fn(() => true)

    const { result } = renderHook(() =>
      useComposerDrop({
        cwd: '/repo',
        insertInlineRefs,
        onAttachDroppedItems,
        recordUndoPoint: vi.fn(),
        requestMainFocus: vi.fn(),
        sessionId: null
      })
    )

    const event = {
      dataTransfer: {
        files: { item: () => null, length: 0 },
        getData: (type: string) =>
          type === HERMES_PATHS_MIME ? JSON.stringify([{ path: 'src/app.ts' }]) : '',
        items: [],
        types: [HERMES_PATHS_MIME, 'text/plain']
      },
      preventDefault: vi.fn(),
      stopPropagation: vi.fn()
    }

    act(() => result.current.handleInputDrop(event as never))

    expect(insertInlineRefs).toHaveBeenCalled()
    expect(event.preventDefault).toHaveBeenCalled()
  })
})

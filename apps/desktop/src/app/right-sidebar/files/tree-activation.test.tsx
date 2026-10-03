import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('react-arborist', async () => {
  const { forwardRef, useImperativeHandle } = await import('react')

  return {
    Tree: forwardRef(function MockTree(props: { onActivate?: (node: unknown) => void }, ref) {
      const file = {
        data: { id: 'C:/repo/file.txt', isDirectory: false },
        isOpen: false,
        select: vi.fn()
      }

      useImperativeHandle(ref, () => ({ focusedNode: file, selectedNodes: [file] }))

      return (
        <button data-testid="activate-file" onClick={() => props.onActivate?.(file)}>
          activate
        </button>
      )
    })
  }
})

vi.mock('@/hooks/use-resize-observer', () => ({
  useResizeObserver: (callback: (entries: readonly ResizeObserverEntry[]) => void) => {
    useEffect(() => callback([]), [callback])
  }
}))

vi.mock('../file-actions', () => ({
  FileEntryContextMenu: ({ children }: { children: unknown }) => children,
  InlineRenameInput: () => null,
  isRenameShortcut: () => false
}))

vi.mock('./dnd-manager', () => ({ getFileTreeDndManager: () => ({}) }))

import { ProjectTree } from './tree'

afterEach(cleanup)

describe('Files panel preview activation contract (#93970)', () => {
  it('does not open preview on a single Arborist activation', async () => {
    const preview = vi.fn()
    const original = HTMLElement.prototype.getBoundingClientRect
    HTMLElement.prototype.getBoundingClientRect = () => ({ height: 200, width: 300 }) as DOMRect

    try {
      await act(async () => {
        render(
          <ProjectTree
            collapseNonce={0}
            cwd="C:/repo"
            data={[]}
            onActivateFile={vi.fn()}
            onActivateFolder={vi.fn()}
            onLoadChildren={vi.fn()}
            onNodeOpenChange={vi.fn()}
            onPreviewFile={preview}
            openState={{}}
          />
        )
      })

      fireEvent.click(screen.getByTestId('activate-file'))
      expect(preview).not.toHaveBeenCalled()
    } finally {
      HTMLElement.prototype.getBoundingClientRect = original
    }
  })

  it('keeps Space as the keyboard preview action', async () => {
    const preview = vi.fn()
    const original = HTMLElement.prototype.getBoundingClientRect
    HTMLElement.prototype.getBoundingClientRect = () => ({ height: 200, width: 300 }) as DOMRect

    try {
      await act(async () => {
        render(
          <ProjectTree
            collapseNonce={0}
            cwd="C:/repo"
            data={[]}
            onActivateFile={vi.fn()}
            onActivateFolder={vi.fn()}
            onLoadChildren={vi.fn()}
            onNodeOpenChange={vi.fn()}
            onPreviewFile={preview}
            openState={{}}
          />
        )
      })

      fireEvent.keyDown(screen.getByTestId('activate-file'), { key: ' ' })
      expect(preview).toHaveBeenCalledOnce()
      expect(preview).toHaveBeenCalledWith('C:/repo/file.txt')
    } finally {
      HTMLElement.prototype.getBoundingClientRect = original
    }
  })
})

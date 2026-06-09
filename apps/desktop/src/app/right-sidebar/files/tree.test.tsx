import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ProjectTree } from './tree'

class ResizeObserverMock {
  private callback: ResizeObserverCallback

  constructor(callback: ResizeObserverCallback) {
    this.callback = callback
  }

  observe(target: Element) {
    this.callback([{ target, contentRect: { height: 300, width: 400 } } as ResizeObserverEntry], this)
  }

  disconnect() {}
  unobserve() {}
}

describe('ProjectTree', () => {
  const originalResizeObserver = window.ResizeObserver
  const originalGetBoundingClientRect = Element.prototype.getBoundingClientRect

  beforeEach(() => {
    window.ResizeObserver = ResizeObserverMock as unknown as typeof ResizeObserver
    Element.prototype.getBoundingClientRect = vi.fn(() => ({
      bottom: 300,
      height: 300,
      left: 0,
      right: 400,
      top: 0,
      width: 400,
      x: 0,
      y: 0,
      toJSON: () => ({})
    }))
  })

  afterEach(() => {
    cleanup()
    window.ResizeObserver = originalResizeObserver
    Element.prototype.getBoundingClientRect = originalGetBoundingClientRect
  })

  it('shows a Preview context menu action for files', async () => {
    const onPreviewFile = vi.fn()
    const onActivateFile = vi.fn()

    render(
      <ProjectTree
        collapseNonce={0}
        cwd="/project"
        data={[{ id: '/project/README.md', name: 'README.md', isDirectory: false }]}
        onActivateFile={onActivateFile}
        onActivateFolder={vi.fn()}
        onLoadChildren={vi.fn()}
        onNodeOpenChange={vi.fn()}
        onPreviewFile={onPreviewFile}
        openState={{}}
      />
    )

    fireEvent.contextMenu(await screen.findByText('README.md'))
    fireEvent.click(await screen.findByText('Preview'))

    await waitFor(() => expect(onPreviewFile).toHaveBeenCalledWith('/project/README.md'))
    expect(onActivateFile).not.toHaveBeenCalled()
  })

  it('keeps regular click as selection and shift-click as attach-file', async () => {
    const onPreviewFile = vi.fn()
    const onActivateFile = vi.fn()

    render(
      <ProjectTree
        collapseNonce={0}
        cwd="/project"
        data={[{ id: '/project/README.md', name: 'README.md', isDirectory: false }]}
        onActivateFile={onActivateFile}
        onActivateFolder={vi.fn()}
        onLoadChildren={vi.fn()}
        onNodeOpenChange={vi.fn()}
        onPreviewFile={onPreviewFile}
        openState={{}}
      />
    )

    fireEvent.click(await screen.findByText('README.md'))
    expect(onPreviewFile).not.toHaveBeenCalled()
    expect(onActivateFile).not.toHaveBeenCalled()

    fireEvent.click(await screen.findByText('README.md'), { shiftKey: true })

    await waitFor(() => expect(onActivateFile).toHaveBeenCalledWith('/project/README.md'))
    expect(onPreviewFile).not.toHaveBeenCalled()
  })
})

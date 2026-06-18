import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import { getPaneStateSnapshot, setPaneOpen } from '@/store/panes'
import { clearSessionPreviewRegistry } from '@/store/preview'
import { $activeSessionId, $connection, setCurrentCwd } from '@/store/session'

import { resetProjectTreeState } from './files/use-project-tree'

import { RightSidebarPane } from './index'

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()
const selectPaths = vi.fn()

vi.mock('@/lib/local-preview', () => ({
  normalizeOrLocalPreviewTarget: vi.fn(async (path: string) => ({
    kind: 'file',
    language: 'markdown',
    previewKind: 'text',
    source: path,
    title: 'README.md',
    url: path
  }))
}))

function ok(entries: { name: string; path: string; isDirectory: boolean }[]): HermesReadDirResult {
  return { entries }
}

function installBridge() {
  ;(
    window as unknown as {
      hermesDesktop: {
        readDir: typeof readDir
        selectPaths: typeof selectPaths
      }
    }
  ).hermesDesktop = { readDir, selectPaths }
}

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

describe('RightSidebarPane', () => {
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
    $connection.set(null)
    $activeSessionId.set('session-1')
    resetProjectTreeState()
    setCurrentCwd('/repo')
    clearSessionPreviewRegistry()
    setPaneOpen('preview', false)
    readDir.mockReset()
    selectPaths.mockReset()
    readDir.mockResolvedValue(ok([{ name: 'README.md', path: '/repo/README.md', isDirectory: false }]))
    selectPaths.mockResolvedValue(['/repo-next'])
    installBridge()
  })

  afterEach(() => {
    cleanup()
    window.ResizeObserver = originalResizeObserver
    Element.prototype.getBoundingClientRect = originalGetBoundingClientRect
    $connection.set(null)
    $activeSessionId.set(null)
    setCurrentCwd('')
    clearSessionPreviewRegistry()
    resetProjectTreeState()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('refreshes the current tree without opening the folder picker', async () => {
    const onChangeCwd = vi.fn()

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} onChangeCwd={onChangeCwd} />)

    await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh tree' }).hasAttribute('disabled')).toBe(false))

    readDir.mockClear()

    fireEvent.click(screen.getByRole('button', { name: 'Refresh tree' }))

    await waitFor(() => expect(readDir).toHaveBeenCalledWith('/repo'))
    expect(selectPaths).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Open folder' }))

    await waitFor(() =>
      expect(selectPaths).toHaveBeenCalledWith({
        defaultPath: '/repo',
        directories: true,
        multiple: false,
        title: 'Change working directory'
      })
    )
    await waitFor(() => expect(onChangeCwd).toHaveBeenCalledWith('/repo-next'))
  })

  it('opens the preview pane when a file is previewed from the file tree', async () => {
    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} onChangeCwd={vi.fn()} />)

    fireEvent.doubleClick(await screen.findByText('README.md'))

    await waitFor(() => expect(getPaneStateSnapshot('preview')?.open).toBe(true))
  })
})

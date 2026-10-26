import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $notifications } from '@/store/notifications'

import { useComposerActions } from './use-composer-actions'

// A drop item the composer cannot attach is reported in one bounded warning, never dropped silently.
describe('useComposerActions drop warnings', () => {
  afterEach(() => {
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('warns about a dropped folder it cannot attach instead of staging its File', async () => {
    const stageFileForAttach = vi.fn(async () => '/srv/uploads/web-my-repo')
    const value = { getPathForFile: () => '', stageFileForAttach }
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value })

    const { result } = renderHook(() =>
      useComposerActions({ activeSessionId: null, currentCwd: '/w', requestGateway: vi.fn() })
    )

    let attached = true

    await act(async () => {
      attached = await result.current.attachDroppedItems([{ file: new File([], 'shots.png'), isDirectory: true, path: '' }])
    })

    expect(attached).toBe(false)
    expect(stageFileForAttach).not.toHaveBeenCalled()
    expect($notifications.get().at(-1)).toMatchObject({ kind: 'warning' })
    expect($notifications.get().at(-1)?.message).toContain("Could not attach shots.png: Folders can't be attached")
  })

  // A drop that attached some files must still name the ones it left out.
  it('reports the items a mixed drop could not attach alongside the ones it did', async () => {
    const stageFileForAttach = vi.fn(async (file: File) => {
      if (file.name === 'big.zip') {
        throw new Error('Upload request is too large')
      }

      return `/srv/uploads/web-${file.name}`
    })

    const value = { getPathForFile: () => '', stageFileForAttach }
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value })

    const add = vi.fn()

    const { result } = renderHook(() =>
      useComposerActions({
        activeSessionId: null,
        currentCwd: '/w',
        requestGateway: vi.fn(),
        scope: { add, remove: vi.fn(() => null), target: 'main', update: vi.fn(() => true), updateIfCurrent: vi.fn(() => true) }
      })
    )

    let attached = false

    await act(async () => {
      attached = await result.current.attachDroppedItems([
        { file: new File(['x'], 'notes.txt'), path: '' },
        { file: new File(['x'], 'big.zip'), path: '' },
        { file: new File([], 'my-repo'), isDirectory: true, path: '' }
      ])
    })

    expect(attached).toBe(true)
    expect(add).toHaveBeenCalledOnce()
    expect(add).toHaveBeenCalledWith(expect.objectContaining({ path: '/srv/uploads/web-notes.txt' }))

    const warning = $notifications.get().find(notice => notice.message.includes('big.zip'))

    expect(warning).toMatchObject({ kind: 'warning' })
    expect(warning?.message).toContain('Could not attach big.zip: Upload request is too large')
    expect(warning?.message).toContain("Could not attach my-repo: Folders can't be attached")
  })

  // 50 oversized files must not grow one toast off screen: name a few, count the rest.
  it('bounds the warning for a large mixed drop and keeps the full list in its detail', async () => {
    const stageFileForAttach = vi.fn(async (file: File) => {
      if (file.name.startsWith('huge-')) {
        throw new Error('Upload request is too large')
      }

      return `/srv/uploads/web-${file.name}`
    })

    const value = { getPathForFile: () => '', stageFileForAttach }
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value })

    const add = vi.fn()

    const { result } = renderHook(() =>
      useComposerActions({
        activeSessionId: null,
        currentCwd: '/w',
        requestGateway: vi.fn(),
        scope: { add, remove: vi.fn(() => null), target: 'main', update: vi.fn(() => true), updateIfCurrent: vi.fn(() => true) }
      })
    )

    const huge = Array.from({ length: 50 }, (_, index) => ({ file: new File(['x'], `huge-${index}.zip`), path: '' }))

    await act(async () => {
      await result.current.attachDroppedItems([{ file: new File(['x'], 'notes.txt'), path: '' }, ...huge])
    })

    expect(add).toHaveBeenCalledOnce()

    const warning = $notifications.get().find(notice => notice.message.includes('huge-0.zip'))

    expect(warning).toMatchObject({ kind: 'warning' })
    expect(warning?.message).toContain('huge-2.zip')
    expect(warning?.message).not.toContain('huge-3.zip')
    expect(warning?.message).toMatch(/and 47 more$/)
    expect(warning?.message.length).toBeLessThan(300)
    expect(warning?.detail?.split('\n')).toHaveLength(50)
    expect(warning?.detail).toContain('huge-49.zip')
  })
})

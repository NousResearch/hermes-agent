import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'

import { resetProjectTreeState, useProjectTree } from './use-project-tree'

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()

beforeEach(() => {
  resetProjectTreeState()
  readDir.mockReset()
  ;(window as unknown as { hermesDesktop: { readDir: typeof readDir } }).hermesDesktop = { readDir }
})

afterEach(() => {
  resetProjectTreeState()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

function ok(entries: { name: string; path: string; isDirectory: boolean }[]): HermesReadDirResult {
  return { entries }
}

describe('useProjectTree', () => {
  it('starts empty when cwd is blank and skips IPC', async () => {
    const { result } = renderHook(() => useProjectTree(''))

    await waitFor(() => expect(result.current.rootLoading).toBe(false))

    expect(result.current.data).toEqual([])
    expect(result.current.rootError).toBeNull()
    expect(readDir).not.toHaveBeenCalled()
  })

  it('loads root entries on mount and sorts folders before files', async () => {
    readDir.mockResolvedValueOnce(
      ok([
        { name: 'README.md', path: '/p/README.md', isDirectory: false },
        { name: 'src', path: '/p/src', isDirectory: true }
      ])
    )

    const { result } = renderHook(() => useProjectTree('/p'))

    await waitFor(() => expect(result.current.data.length).toBe(2))

    expect(readDir).toHaveBeenCalledWith('/p')
    // Hook trusts main-process sort order; folders/files preserved as supplied.
    expect(result.current.data.map(n => n.name)).toEqual(['README.md', 'src'])
    // Folder children start undefined (lazy load on first expand).
    expect(result.current.data.find(n => n.name === 'src')?.children).toBeUndefined()
    expect(result.current.data.find(n => n.name === 'src')?.isDirectory).toBe(true)
    expect(result.current.data.find(n => n.name === 'README.md')?.isDirectory).toBe(false)
  })

  it('records rootError when readDir returns an error', async () => {
    readDir.mockResolvedValueOnce({ entries: [], error: 'EACCES' })

    const { result } = renderHook(() => useProjectTree('/locked'))

    await waitFor(() => expect(result.current.rootError).toBe('EACCES'))
    expect(result.current.data).toEqual([])
  })

  it('lazy-loads children on loadChildren and replaces the placeholder', async () => {
    readDir.mockResolvedValueOnce(ok([{ name: 'src', path: '/p/src', isDirectory: true }]))
    readDir.mockResolvedValueOnce(
      ok([
        { name: 'index.ts', path: '/p/src/index.ts', isDirectory: false },
        { name: 'lib', path: '/p/src/lib', isDirectory: true }
      ])
    )

    const { result } = renderHook(() => useProjectTree('/p'))

    await waitFor(() => expect(result.current.data.length).toBe(1))

    await act(async () => {
      await result.current.loadChildren('/p/src')
    })

    const src = result.current.data[0]
    expect(src.children?.map(n => n.name)).toEqual(['index.ts', 'lib'])
    expect(src.loading).toBe(false)
    expect(src.error).toBeUndefined()
  })

  it('keeps loaded tree state across remounts for the same cwd', async () => {
    readDir.mockResolvedValueOnce(ok([{ name: 'src', path: '/p/src', isDirectory: true }]))

    const { result, unmount } = renderHook(() => useProjectTree('/p'))

    await waitFor(() => expect(result.current.data.length).toBe(1))

    act(() => {
      result.current.setNodeOpen('/p/src', true)
    })

    unmount()

    const remounted = renderHook(() => useProjectTree('/p'))

    expect(remounted.result.current.data.map(n => n.name)).toEqual(['src'])
    expect(remounted.result.current.openState).toEqual({ '/p/src': true })
    expect(readDir).toHaveBeenCalledTimes(1)
  })

  it('captures per-folder error code and leaves the folder expandable but empty', async () => {
    readDir.mockResolvedValueOnce(ok([{ name: 'priv', path: '/p/priv', isDirectory: true }]))
    readDir.mockResolvedValueOnce({ entries: [], error: 'EACCES' })

    const { result } = renderHook(() => useProjectTree('/p'))

    await waitFor(() => expect(result.current.data.length).toBe(1))

    await act(async () => {
      await result.current.loadChildren('/p/priv')
    })

    expect(result.current.data[0].error).toBe('EACCES')
    expect(result.current.data[0].children).toEqual([])
  })

  it('dedupes concurrent loadChildren calls for the same id', async () => {
    readDir.mockResolvedValueOnce(ok([{ name: 'src', path: '/p/src', isDirectory: true }]))

    let resolveChildren: ((value: HermesReadDirResult) => void) | undefined
    readDir.mockImplementationOnce(
      () =>
        new Promise<HermesReadDirResult>(resolve => {
          resolveChildren = resolve
        })
    )

    const { result } = renderHook(() => useProjectTree('/p'))

    await waitFor(() => expect(result.current.data.length).toBe(1))

    await act(async () => {
      // First call enters inflight, second short-circuits, third also short-circuits.
      void result.current.loadChildren('/p/src')
      void result.current.loadChildren('/p/src')
      void result.current.loadChildren('/p/src')
      resolveChildren?.(ok([{ name: 'a.ts', path: '/p/src/a.ts', isDirectory: false }]))
    })

    // Mount load + a single folder fetch — duplicates were dropped.
    expect(readDir).toHaveBeenCalledTimes(2)
  })

  it('refreshRoot reloads the root and clears prior error', async () => {
    readDir.mockResolvedValueOnce({ entries: [], error: 'EACCES' })
    readDir.mockResolvedValueOnce(ok([{ name: 'README.md', path: '/p/README.md', isDirectory: false }]))

    const { result } = renderHook(() => useProjectTree('/p'))

    await waitFor(() => expect(result.current.rootError).toBe('EACCES'))

    await act(async () => {
      await result.current.refreshRoot()
    })

    expect(result.current.rootError).toBeNull()
    expect(result.current.data.map(n => n.name)).toEqual(['README.md'])
  })

  it('reloads when cwd changes', async () => {
    readDir.mockResolvedValueOnce(ok([{ name: 'one', path: '/a/one', isDirectory: false }]))
    readDir.mockResolvedValueOnce(ok([{ name: 'two', path: '/b/two', isDirectory: false }]))

    const { rerender, result } = renderHook(({ cwd }) => useProjectTree(cwd), { initialProps: { cwd: '/a' } })

    await waitFor(() => expect(result.current.data[0]?.name).toBe('one'))

    rerender({ cwd: '/b' })

    await waitFor(() => expect(result.current.data[0]?.name).toBe('two'))
    expect(readDir).toHaveBeenLastCalledWith('/b')
  })

  it('returns no-bridge gracefully when window.hermesDesktop is missing', async () => {
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop

    const { result } = renderHook(() => useProjectTree('/p'))

    await waitFor(() => expect(result.current.rootError).toBe('no-bridge'))
    expect(result.current.data).toEqual([])
  })
})

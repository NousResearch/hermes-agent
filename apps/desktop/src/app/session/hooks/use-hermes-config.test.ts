import { act, renderHook } from '@testing-library/react'
// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { getHermesConfig } from '@/hermes'
import { persistString } from '@/lib/storage'
import { $currentCwd, setCurrentCwd } from '@/store/session'

import { useHermesConfig } from './use-hermes-config'

vi.mock('@/hermes', () => ({
  getHermesConfig: vi.fn(),
  getHermesConfigDefaults: vi.fn().mockResolvedValue({}),
}))

const WORKSPACE_CWD_KEY = 'hermes.desktop.workspace-cwd'

describe('useHermesConfig refreshHermesConfig', () => {
  beforeEach(() => {
    // Reset atoms and localStorage between tests
    setCurrentCwd('')
    persistString(WORKSPACE_CWD_KEY, null)
  })

  it('applies terminal.cwd from config even when localStorage has a stale value', async () => {
    // Simulate a stale remembered workspace cwd
    persistString(WORKSPACE_CWD_KEY, '/Users/old/stale-project')
    setCurrentCwd('/Users/old/stale-project')

    vi.mocked(getHermesConfig).mockResolvedValue({
      terminal: { cwd: '/Users/example/new-workspace' },
    } as any)

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: null },
        refreshProjectBranch: vi.fn().mockResolvedValue(undefined),
      }),
    )

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    // The configured terminal.cwd must override the stale localStorage value
    expect($currentCwd.get()).toBe('/Users/example/new-workspace')
  })

  it('uses empty string when terminal.cwd is not set and localStorage is empty', async () => {
    vi.mocked(getHermesConfig).mockResolvedValue({} as any)

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: null },
        refreshProjectBranch: vi.fn().mockResolvedValue(undefined),
      }),
    )

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    expect($currentCwd.get()).toBe('')
  })

  it('ignores terminal.cwd when it is "."', async () => {
    vi.mocked(getHermesConfig).mockResolvedValue({
      terminal: { cwd: '.' },
    } as any)

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: null },
        refreshProjectBranch: vi.fn().mockResolvedValue(undefined),
      }),
    )

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    expect($currentCwd.get()).toBe('')
  })

  it('calls refreshProjectBranch with the configured cwd', async () => {
    const refreshProjectBranch = vi.fn().mockResolvedValue(undefined)
    setCurrentCwd('')

    vi.mocked(getHermesConfig).mockResolvedValue({
      terminal: { cwd: '/workspace/project-a' },
    } as any)

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: null },
        refreshProjectBranch,
      }),
    )

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    expect(refreshProjectBranch).toHaveBeenCalledWith('/workspace/project-a')
  })
})
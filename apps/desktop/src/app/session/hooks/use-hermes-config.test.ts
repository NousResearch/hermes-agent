// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { getHermesConfig } from '@/hermes'
import { persistString } from '@/lib/storage'
import {
  $currentCwd,
  $currentFastMode,
  $currentReasoningEffort,
  markComposerSelectionManual,
  setCurrentCwd,
  setCurrentFastMode,
  setCurrentModelSource,
  setCurrentReasoningEffort
} from '@/store/session'

import { useHermesConfig } from './use-hermes-config'

vi.mock('@/hermes', () => ({
  getHermesConfig: vi.fn(),
  getHermesConfigDefaults: vi.fn().mockResolvedValue({})
}))

const WORKSPACE_CWD_KEY = 'hermes.desktop.workspace-cwd'

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

const mockConfig = (config: Record<string, unknown>) =>
  vi.mocked(getHermesConfig).mockResolvedValue(config as Awaited<ReturnType<typeof getHermesConfig>>)

describe('useHermesConfig refreshHermesConfig', () => {
  beforeEach(() => {
    // Reset atoms and localStorage between tests
    setCurrentCwd('')
    setCurrentFastMode(false)
    setCurrentModelSource('')
    setCurrentReasoningEffort('')
    persistString(WORKSPACE_CWD_KEY, null)
  })

  it('applies terminal.cwd from config even when localStorage has a stale value', async () => {
    // Simulate a stale remembered workspace cwd
    persistString(WORKSPACE_CWD_KEY, '/Users/old/stale-project')
    setCurrentCwd('/Users/old/stale-project')

    mockConfig({ terminal: { cwd: '/Users/example/new-workspace' } })

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: null },
        refreshProjectBranch: vi.fn().mockResolvedValue(undefined)
      })
    )

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    // The configured terminal.cwd must override the stale localStorage value
    expect($currentCwd.get()).toBe('/Users/example/new-workspace')
  })

  it('keeps the active session workspace when a session is running', async () => {
    setCurrentCwd('/workspace/attached-project')

    mockConfig({ terminal: { cwd: '/Users/example/new-workspace' } })

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: 'session-1' },
        refreshProjectBranch: vi.fn().mockResolvedValue(undefined)
      })
    )

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    // Config refreshes mid-session must not yank the workspace out from
    // under the attached session.
    expect($currentCwd.get()).toBe('/workspace/attached-project')
  })

  it('uses empty string when terminal.cwd is not set and localStorage is empty', async () => {
    mockConfig({})

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: null },
        refreshProjectBranch: vi.fn().mockResolvedValue(undefined)
      })
    )

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    expect($currentCwd.get()).toBe('')
  })

  it('ignores terminal.cwd when it is "."', async () => {
    mockConfig({ terminal: { cwd: '.' } })

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: null },
        refreshProjectBranch: vi.fn().mockResolvedValue(undefined)
      })
    )

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    expect($currentCwd.get()).toBe('')
  })

  it('calls refreshProjectBranch with the configured cwd', async () => {
    const refreshProjectBranch = vi.fn().mockResolvedValue(undefined)
    setCurrentCwd('')

    mockConfig({ terminal: { cwd: '/workspace/project-a' } })

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: null },
        refreshProjectBranch
      })
    )

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    expect(refreshProjectBranch).toHaveBeenCalledWith('/workspace/project-a')
  })

  it('refreshes the branch for the session cwd (not config) when a session is active', async () => {
    const refreshProjectBranch = vi.fn().mockResolvedValue(undefined)
    setCurrentCwd('/workspace/attached-project')

    mockConfig({ terminal: { cwd: '/Users/example/new-workspace' } })

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: 'session-1' },
        refreshProjectBranch
      })
    )

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    expect(refreshProjectBranch).toHaveBeenCalledWith('/workspace/attached-project')
  })

  it('does not let a stale forced config refresh overwrite newer draft selector intent', async () => {
    const profileConfig = deferred<Awaited<ReturnType<typeof getHermesConfig>>>()
    vi.mocked(getHermesConfig).mockReturnValueOnce(profileConfig.promise)

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: null },
        refreshProjectBranch: vi.fn().mockResolvedValue(undefined)
      })
    )

    let pendingRefresh!: Promise<void>
    act(() => {
      pendingRefresh = result.current.refreshHermesConfig(true)
    })
    expect(getHermesConfig).toHaveBeenCalled()

    // The user turns Fast off and chooses a different effort while the profile
    // defaults are still loading. That newer picker intent owns the composer.
    markComposerSelectionManual()
    setCurrentReasoningEffort('high')
    setCurrentFastMode(false)
    profileConfig.resolve({
      agent: { reasoning_effort: 'low', service_tier: 'priority' }
    } as Awaited<ReturnType<typeof getHermesConfig>>)

    await act(async () => {
      await pendingRefresh
    })

    expect($currentReasoningEffort.get()).toBe('high')
    expect($currentFastMode.get()).toBe(false)
  })

  it('does not let an older profile config overwrite a newer profile', async () => {
    const profileB = deferred<Awaited<ReturnType<typeof getHermesConfig>>>()
    const profileC = deferred<Awaited<ReturnType<typeof getHermesConfig>>>()
    vi.mocked(getHermesConfig).mockReturnValueOnce(profileB.promise).mockReturnValueOnce(profileC.promise)

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: null },
        refreshProjectBranch: vi.fn().mockResolvedValue(undefined)
      })
    )

    let refreshB!: Promise<void>
    let refreshC!: Promise<void>
    act(() => {
      refreshB = result.current.refreshHermesConfig(true)
      refreshC = result.current.refreshHermesConfig(true)
    })

    profileC.resolve({ agent: { reasoning_effort: 'low', service_tier: 'normal' } })
    await act(async () => {
      await refreshC
    })
    profileB.resolve({ agent: { reasoning_effort: 'high', service_tier: 'priority' } })
    await act(async () => {
      await refreshB
    })

    expect($currentReasoningEffort.get()).toBe('low')
    expect($currentFastMode.get()).toBe(false)
  })
})

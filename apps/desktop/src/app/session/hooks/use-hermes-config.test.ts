// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getEffectivePreventSleepConfig, getHermesConfig } from '@/hermes'
import { persistString } from '@/lib/storage'
import { $currentCwd, setCurrentCwd } from '@/store/session'

import { useHermesConfig } from './use-hermes-config'

vi.mock('@/hermes', () => ({
  getEffectivePreventSleepConfig: vi.fn(),
  getHermesConfig: vi.fn(),
  getHermesConfigDefaults: vi.fn().mockResolvedValue({})
}))

const WORKSPACE_CWD_KEY = 'hermes.desktop.workspace-cwd'
const refreshPowerSaveBlocker = vi.fn()

const mockConfig = (config: Record<string, unknown>) =>
  vi.mocked(getHermesConfig).mockResolvedValue(config as Awaited<ReturnType<typeof getHermesConfig>>)

describe('useHermesConfig refreshHermesConfig', () => {
  beforeEach(() => {
    // Reset atoms and localStorage between tests
    setCurrentCwd('')
    persistString(WORKSPACE_CWD_KEY, null)
    vi.mocked(getEffectivePreventSleepConfig)
      .mockReset()
      .mockResolvedValue({
        enabled: false,
        mode: 'system',
        surfaces: ['desktop', 'gateway']
      })
    vi.mocked(getHermesConfig).mockReset()
    refreshPowerSaveBlocker.mockReset().mockResolvedValue({
      active: false,
      mode: 'system',
      ok: true,
      surfaces: ['desktop', 'gateway']
    })
    vi.stubGlobal('hermesDesktop', { refreshPowerSaveBlocker })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('refreshes Electron sleep prevention from the canonical backend decision', async () => {
    const block = { enabled: true, mode: 'display' as const, surfaces: ['desktop'] }

    mockConfig({})
    vi.mocked(getEffectivePreventSleepConfig).mockResolvedValue(block)

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: null },
        refreshProjectBranch: vi.fn().mockResolvedValue(undefined)
      })
    )

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    expect(refreshPowerSaveBlocker).toHaveBeenCalledWith(block)
  })

  it('discards a stale config response after a newer profile refresh wins', async () => {
    type Config = Awaited<ReturnType<typeof getHermesConfig>>

    let resolveFirst!: (config: Config) => void

    const firstResponse = new Promise<Config>(resolve => {
      resolveFirst = resolve
    })

    const firstBlock = { enabled: true, mode: 'display', surfaces: ['desktop'] }
    const latestBlock = { enabled: false, mode: 'system', surfaces: ['desktop'] }

    vi.mocked(getHermesConfig)
      .mockImplementationOnce(() => firstResponse)
      .mockResolvedValueOnce({} as Config)
    vi.mocked(getEffectivePreventSleepConfig)
      .mockResolvedValueOnce(firstBlock as Awaited<ReturnType<typeof getEffectivePreventSleepConfig>>)
      .mockResolvedValueOnce(latestBlock as Awaited<ReturnType<typeof getEffectivePreventSleepConfig>>)

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: null },
        refreshProjectBranch: vi.fn().mockResolvedValue(undefined)
      })
    )

    const firstRefresh = result.current.refreshHermesConfig()

    await act(async () => {
      await result.current.refreshHermesConfig()
    })
    await act(async () => {
      resolveFirst({} as Config)
      await firstRefresh
    })

    expect(refreshPowerSaveBlocker).toHaveBeenCalledTimes(1)
    expect(refreshPowerSaveBlocker).toHaveBeenCalledWith(latestBlock)
  })

  it('preserves config refreshes when an older backend lacks the power endpoint', async () => {
    vi.mocked(getEffectivePreventSleepConfig).mockRejectedValue(new Error('not found'))
    mockConfig({ terminal: { cwd: '/workspace/from-config' } })

    const { result } = renderHook(() =>
      useHermesConfig({
        activeSessionIdRef: { current: null },
        refreshProjectBranch: vi.fn().mockResolvedValue(undefined)
      })
    )

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    expect($currentCwd.get()).toBe('/workspace/from-config')
    expect(refreshPowerSaveBlocker).not.toHaveBeenCalled()
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
})

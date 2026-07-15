// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { getHermesConfig } from '@/hermes'
import { persistString } from '@/lib/storage'
import { $currentCwd, setCurrentCwd } from '@/store/session'

import { isFileBrowserHoverRevealEnabled, useHermesConfig } from './use-hermes-config'

vi.mock('@/hermes', () => ({
  getHermesConfig: vi.fn(),
  getHermesConfigDefaults: vi.fn().mockResolvedValue({})
}))

const WORKSPACE_CWD_KEY = 'hermes.desktop.workspace-cwd'

const mockConfig = (config: Record<string, unknown>) =>
  vi.mocked(getHermesConfig).mockResolvedValue(config as Awaited<ReturnType<typeof getHermesConfig>>)

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

describe('isFileBrowserHoverRevealEnabled', () => {
  it('defaults to enabled for existing configs', () => {
    expect(isFileBrowserHoverRevealEnabled({})).toBe(true)
    expect(isFileBrowserHoverRevealEnabled({ display: {} })).toBe(true)
  })

  it('honors an explicit false value', () => {
    expect(isFileBrowserHoverRevealEnabled({ display: { hover_reveal_file_browser: false } })).toBe(false)
  })

  it('keeps hover reveal enabled when explicitly true', () => {
    expect(isFileBrowserHoverRevealEnabled({ display: { hover_reveal_file_browser: true } })).toBe(true)
  })
})

// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $terminalFontFamily, setTerminalFontFamilyFromConfig } from '@/app/right-sidebar/terminal/terminal-font'
import { getHermesConfig } from '@/hermes'
import { persistString } from '@/lib/storage'
import { $showReasoning, setShowReasoningFromConfig } from '@/store/reasoning-disclosure'
import {
  $currentCwd,
  $currentFastMode,
  $currentReasoningEffort,
  $currentServiceTier,
  $defaultReasoningEffort,
  markComposerSelectionManual,
  setCurrentCwd,
  setCurrentFastMode,
  setCurrentModelSource,
  setCurrentReasoningEffort,
  setDefaultReasoningEffort
} from '@/store/session'
import { $showToolActivity, setShowToolActivityFromConfig } from '@/store/tool-activity'

import { deferred } from '../../../test/deferred'

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
    setShowReasoningFromConfig(undefined)
    setShowToolActivityFromConfig(undefined)
    setCurrentCwd('')
    setCurrentFastMode(false)
    setCurrentModelSource('')
    setCurrentReasoningEffort('')
    setDefaultReasoningEffort('')
    setTerminalFontFamilyFromConfig('')
    persistString(WORKSPACE_CWD_KEY, null)
  })

  // #49664: the Reasoning Blocks toggle wrote config but the renderer never
  // read it. A refresh mirrors the key (quoted "false" included) and a
  // missing key falls back to the DEFAULT_CONFIG default (on).
  it('mirrors display.show_reasoning and resets a missing key to the default', async () => {
    mockConfig({ display: { show_reasoning: 'false' } })
    const { result } = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: null } }))

    await act(async () => {
      await result.current.refreshHermesConfig()
    })
    expect($showReasoning.get()).toBe(false)

    mockConfig({})
    await act(async () => {
      await result.current.refreshHermesConfig()
    })
    expect($showReasoning.get()).toBe(true)
  })

  it('mirrors display.tool_progress independently of show_reasoning', async () => {
    mockConfig({ display: { show_reasoning: false, tool_progress: 'off' } })
    const { result } = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: null } }))

    await act(async () => {
      await result.current.refreshHermesConfig()
    })
    expect($showToolActivity.get()).toBe(false)

    mockConfig({ display: { show_reasoning: false } })
    await act(async () => {
      await result.current.refreshHermesConfig()
    })
    expect($showToolActivity.get()).toBe(true)
  })

  // Regression: the composer keeps a manual model pick sticky, which skips the
  // composer reseed. The profile default must still be published, because the
  // model picker resolves "the default effort" from it when applying a model's
  // preset — otherwise selecting a model silently downgrades a configured
  // `agent.reasoning_effort: high` to Hermes' built-in medium.
  it('publishes the profile default effort even when a manual pick blocks the composer reseed', async () => {
    setCurrentModelSource('manual')
    setCurrentReasoningEffort('low')

    mockConfig({ agent: { reasoning_effort: 'high' } })
    const { result } = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: null } }))

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    expect($defaultReasoningEffort.get()).toBe('high')
    // The manual pick itself is still respected.
    expect($currentReasoningEffort.get()).toBe('low')
  })

  it('does not let terminal.cwd replace an inactive selected workspace', async () => {
    setCurrentCwd('/Users/example/repo/.worktrees/feature')

    mockConfig({ terminal: { cwd: '/Users/example/new-workspace' } })
    const { result } = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: null } }))

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    expect($currentCwd.get()).toBe('/Users/example/repo/.worktrees/feature')
  })

  it('does not let terminal.cwd replace an active session workspace', async () => {
    setCurrentCwd('/Users/example/repo/.worktrees/attached')

    mockConfig({ terminal: { cwd: '/Users/example/new-workspace' } })
    const { result } = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: 'session-1' } }))

    await act(async () => {
      await result.current.refreshHermesConfig()
    })

    expect($currentCwd.get()).toBe('/Users/example/repo/.worktrees/attached')
  })

  it('does not let a stale forced config refresh overwrite newer draft selector intent', async () => {
    const profileConfig = deferred<Awaited<ReturnType<typeof getHermesConfig>>>()
    vi.mocked(getHermesConfig).mockReturnValueOnce(profileConfig.promise)

    const { result } = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: null } }))

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

  it('does not publish config after its switch loses ownership', async () => {
    const staleConfig = deferred<Awaited<ReturnType<typeof getHermesConfig>>>()
    vi.mocked(getHermesConfig).mockReturnValueOnce(staleConfig.promise)
    const { result } = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: null } }))
    let ownsSwitch = true

    let refresh!: Promise<void>
    act(() => {
      refresh = result.current.refreshHermesConfig(false, () => ownsSwitch)
    })

    ownsSwitch = false
    staleConfig.resolve({
      display: { show_reasoning: false },
      agent: { reasoning_effort: 'high', service_tier: 'priority' },
      terminal: { font_family: 'MesloLGS NF' }
    } as Awaited<ReturnType<typeof getHermesConfig>>)

    await act(async () => {
      await refresh
    })

    expect($defaultReasoningEffort.get()).toBe('')
    expect($currentReasoningEffort.get()).toBe('')
    expect($currentFastMode.get()).toBe(false)
    expect($terminalFontFamily.get()).toBe('')
    expect($showReasoning.get()).toBe(true)
  })

  it('does not let an older profile config overwrite a newer profile', async () => {
    const profileB = deferred<Awaited<ReturnType<typeof getHermesConfig>>>()
    const profileC = deferred<Awaited<ReturnType<typeof getHermesConfig>>>()
    vi.mocked(getHermesConfig).mockReturnValueOnce(profileB.promise).mockReturnValueOnce(profileC.promise)

    const { result } = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: null } }))

    let refreshB!: Promise<void>
    let refreshC!: Promise<void>
    act(() => {
      refreshB = result.current.refreshHermesConfig(true)
      refreshC = result.current.refreshHermesConfig(true)
    })

    // A raw OpenAI tier word the composer cannot send (create would 4002) seeds Standard.
    profileC.resolve({ agent: { reasoning_effort: 'low', service_tier: 'flex' } })
    await act(async () => {
      await refreshC
    })
    profileB.resolve({ agent: { reasoning_effort: 'high', service_tier: 'priority' } })
    await act(async () => {
      await refreshB
    })

    expect($currentReasoningEffort.get()).toBe('low')
    expect($currentFastMode.get()).toBe(false)
    expect($currentServiceTier.get()).toBe('normal')
  })

  it('does not let an older profile response restore its terminal font', async () => {
    const profileB = deferred<Awaited<ReturnType<typeof getHermesConfig>>>()
    const profileC = deferred<Awaited<ReturnType<typeof getHermesConfig>>>()
    vi.mocked(getHermesConfig).mockReturnValueOnce(profileB.promise).mockReturnValueOnce(profileC.promise)
    const { result } = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: null } }))

    let refreshB!: Promise<void>
    let refreshC!: Promise<void>
    act(() => {
      refreshB = result.current.refreshHermesConfig(true)
      refreshC = result.current.refreshHermesConfig(true)
    })

    profileC.resolve({ terminal: { font_family: 'Hack Nerd Font' } })
    await act(async () => {
      await refreshC
    })
    profileB.resolve({ terminal: { font_family: 'MesloLGS NF' } })
    await act(async () => {
      await refreshB
    })

    expect($terminalFontFamily.get()).toBe('Hack Nerd Font')
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

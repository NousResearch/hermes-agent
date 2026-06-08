import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $currentCwd, setCurrentCwd } from '@/store/session'

import { useHermesConfig } from './use-hermes-config'

vi.mock('@/hermes', () => ({
  getHermesConfig: vi.fn(),
  getHermesConfigDefaults: vi.fn()
}))

import { getHermesConfig, getHermesConfigDefaults } from '@/hermes'

function Probe({ refreshProjectBranch }: { refreshProjectBranch: (cwd: string) => Promise<void> }) {
  const { refreshHermesConfig } = useHermesConfig({
    activeSessionIdRef: { current: null },
    refreshProjectBranch
  })

  return <button onClick={() => void refreshHermesConfig()}>refresh</button>
}

describe('useHermesConfig workspace defaults', () => {
  beforeEach(() => {
    window.localStorage.clear()
    setCurrentCwd('')
    vi.mocked(getHermesConfig).mockResolvedValue({})
    vi.mocked(getHermesConfigDefaults).mockResolvedValue({})
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        settings: {
          getDefaultProjectDir: vi.fn().mockResolvedValue({
            defaultLabel: '/Users/test/fallback',
            dir: '/Users/test/sywork'
          })
        }
      }
    })
  })

  afterEach(() => {
    cleanup()
    window.localStorage.clear()
    setCurrentCwd('')
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.clearAllMocks()
  })

  it('seeds new chats from the desktop default project directory when config cwd is empty', async () => {
    const refreshProjectBranch = vi.fn().mockResolvedValue(undefined)
    const { getByRole } = render(<Probe refreshProjectBranch={refreshProjectBranch} />)

    await act(async () => {
      getByRole('button', { name: 'refresh' }).click()
    })

    await waitFor(() => expect($currentCwd.get()).toBe('/Users/test/sywork'))
    expect(refreshProjectBranch).toHaveBeenCalledWith('/Users/test/sywork')
  })
})

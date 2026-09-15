import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { DesktopUninstallSummary } from '@/global'

import { UninstallSection } from './uninstall-section'

function summary(allowed: boolean): DesktopUninstallSummary {
  return {
    code_removal_allowed: allowed,
    hermes_home: '/test/home',
    agent_installed: true,
    gui_installed: true,
    source_built_artifacts: [],
    packaged_app_paths: [],
    userdata_dir: '/test/desktop',
    userdata_exists: true,
    platform: 'linux'
  }
}

afterEach((): void => {
  cleanup()
  vi.unstubAllGlobals()
})

it.each(['external', 'missing-policy', 'probe-failed', 'loading'] as const)(
  'does not offer uninstall actions when ownership is %s',
  async (state: 'external' | 'missing-policy' | 'probe-failed' | 'loading'): Promise<void> => {
    const getSummary: ReturnType<typeof vi.fn> = vi.fn(async (): Promise<DesktopUninstallSummary> => {
      if (state === 'probe-failed') { throw new Error('IPC unavailable') }

      if (state === 'loading') { return new Promise<DesktopUninstallSummary>((): void => {}) }

      if (state === 'missing-policy') {
        const { code_removal_allowed: _ignored, ...oldSummary }: ReturnType<typeof summary> = summary(true)

        return oldSummary as DesktopUninstallSummary
      }

      return summary(false)
    })

    const run: ReturnType<typeof vi.fn> = vi.fn()

    vi.stubGlobal('hermesDesktop', { uninstall: { summary: getSummary, run } })
    await act(async (): Promise<void> => { render(<UninstallSection />) })
    expect(getSummary).toHaveBeenCalledOnce()
    expect(screen.queryByText('Uninstall Hermes')).toBeNull()
    expect(screen.queryByRole('button', { name: /Uninstall/ })).toBeNull()
    expect(screen.queryByText('Danger zone')).toBeNull()
    expect(run).not.toHaveBeenCalled()
  }
)

it('keeps owned-install removal modes and confirms the selected mode', async (): Promise<void> => {
  const run: ReturnType<typeof vi.fn> = vi.fn().mockResolvedValue({ ok: true })
  vi.stubGlobal('hermesDesktop', {
    uninstall: { summary: async (): Promise<DesktopUninstallSummary> => summary(true), run }
  })
  render(<UninstallSection />)
  fireEvent.click(await screen.findByRole('button', { name: /Uninstall Chat GUI only/ }))
  expect(run).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Yes, uninstall' }))
  await waitFor((): void => { expect(run).toHaveBeenCalledWith('gui') })
})

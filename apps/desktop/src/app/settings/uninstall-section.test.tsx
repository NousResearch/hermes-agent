import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { DesktopUninstallMode, DesktopUninstallResult, DesktopUninstallSummary } from '@/global'
import { I18nProvider } from '@/i18n'
import { ko } from '@/i18n/ko'
import { settingsRiskCopyKo } from '@/i18n/settings-risk-copy'

import { UninstallSection } from './uninstall-section'

const copy = settingsRiskCopyKo.uninstallSection

const summary: DesktopUninstallSummary = {
  agent_installed: true,
  gui_installed: true,
  hermes_home: '/isolated/hermes',
  packaged_app_paths: [],
  source_built_artifacts: [],
  userdata_dir: '/isolated/desktop',
  userdata_exists: true,
  platform: 'win32',
  running_app_path: '/isolated/앱'
}

function mount(probe: Promise<DesktopUninstallSummary>, run = vi.fn<() => Promise<DesktopUninstallResult>>()) {
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = { uninstall: { summary: () => probe, run } }
  render(
    <I18nProvider configClient={null} initialLocale="ko">
      <UninstallSection />
    </I18nProvider>
  )

  return run
}

afterEach(() => {
  cleanup()
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

it('requires scope-specific confirmation, keeps cancel inert, and preserves a failed start after returning to choices', async () => {
  for (const mode of ['gui', 'lite', 'full'] as DesktopUninstallMode[]) {
    const run = mount(Promise.resolve(summary), vi.fn().mockResolvedValue({ ok: false, error: 'EACCES: audit-denied' }))
    const option = await screen.findByRole('button', { name: new RegExp(copy.options[mode].title) })
    fireEvent.click(option)
    expect(screen.getByText(copy.confirmRemoval(copy.options[mode].consequence))).toBeTruthy()
    expect(screen.getByText(copy.appPath(summary.running_app_path!))).toBeTruthy()
    expect(run).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: ko.common.cancel }))
    expect(run).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: new RegExp(copy.options[mode].title) }))
    fireEvent.click(screen.getByRole('button', { name: copy.confirmAction }))
    await waitFor(() => expect(run).toHaveBeenCalledExactlyOnceWith(mode))
    expect((await screen.findByRole('alert')).textContent).toContain(copy.startFailed)
    expect(screen.getByRole('alert').textContent).toContain('EACCES: audit-denied')
    expect(screen.getByRole('button', { name: new RegExp(copy.options[mode].title) })).toBeTruthy()
    cleanup()
  }
})

it('offers only app removal when the agent is absent or the probe fails, and keeps thrown native errors visible', async () => {
  for (const failedProbe of [false, true]) {
    const probe = failedProbe
      ? Promise.reject(new Error('probe unavailable'))
      : Promise.resolve({ ...summary, agent_installed: false })

    mount(probe, vi.fn().mockRejectedValue(new Error('native IPC unavailable')))
    fireEvent.click(await screen.findByRole('button', { name: new RegExp(copy.options.gui.title) }))
    expect(screen.queryByRole('button', { name: new RegExp(copy.options.full.title) })).toBeNull()
    expect(screen.queryByRole('button', { name: new RegExp(copy.options.lite.title) })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: copy.confirmAction }))
    expect((await screen.findByRole('alert')).textContent).toContain('native IPC unavailable')
    cleanup()
  }
})

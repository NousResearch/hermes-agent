import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $desktopVersion, $updateApply, $updateChecking, $updateStatus } from '@/store/updates'

import { AboutSettings } from './about-settings'

function renderAbout(
  locale: 'en' | 'pl' | 'zh' = 'en',
  versionState = {
    appVersion: '0.17.2',
    backendVersion: '0.17.2',
    electronVersion: '40.10.2',
    hermesRoot: '/tmp/hermes-agent',
    nodeVersion: '24.11.0',
    platform: 'linux',
    bundleOutOfSync: false,
    bundleSwapPending: false
  }
) {
  const openExternal = vi.fn()

  window.hermesDesktop = {
    ...window.hermesDesktop,
    getVersion: vi.fn().mockResolvedValue(versionState),
    openExternal
  } as unknown as Window['hermesDesktop']

  render(
    <I18nProvider configClient={null} initialLocale={locale}>
      <AboutSettings />
    </I18nProvider>
  )

  return { openExternal }
}

afterEach(() => {
  cleanup()
  $desktopVersion.set(null)
  $updateApply.set({ applying: false, command: null, error: null, log: [], message: '', percent: null, stage: 'idle' })
  $updateChecking.set(false)
  $updateStatus.set(null)
  vi.restoreAllMocks()
})

describe('AboutSettings', () => {
  it('shows product identity, runtime version, attribution, and MIT license in English', async () => {
    renderAbout('en')

    expect(await screen.findByRole('heading', { name: 'AI Evolution Jarvis' })).not.toBeNull()
    await waitFor(() => expect(screen.getByText('Version 0.17.2')).not.toBeNull())
    expect(screen.getByText('Powered by Hermes Agent — Nous Research')).not.toBeNull()
    expect(screen.getByRole<HTMLAnchorElement>('link', { name: /Hermes Agent repository/i }).href).toBe(
      'https://github.com/NousResearch/hermes-agent'
    )
    expect(screen.getByRole<HTMLAnchorElement>('link', { name: /Nous Research/i }).href).toBe(
      'https://nousresearch.com/'
    )
    expect(screen.getByText('Hermes Agent is distributed under the MIT License.')).not.toBeNull()
    expect(screen.getByRole<HTMLAnchorElement>('link', { name: /MIT license/i }).href).toBe(
      'https://github.com/NousResearch/hermes-agent/blob/main/LICENSE'
    )
    expect(screen.getByText('Updates')).not.toBeNull()
    expect(screen.getByText('Automatic updates')).not.toBeNull()
  })

  it('routes product release and installer actions to AI Evolution distribution URLs', async () => {
    const { openExternal } = renderAbout('en')

    await screen.findByRole('heading', { name: 'AI Evolution Jarvis' })

    const releaseNotes = screen.getByRole<HTMLAnchorElement>('link', { name: /Release notes/i })
    expect(releaseNotes.href).toBe('https://github.com/aievolutionpl/hermes-agent/releases')
    releaseNotes.click()
    expect(openExternal).toHaveBeenCalledWith('https://github.com/aievolutionpl/hermes-agent/releases')

    expect(openExternal).not.toHaveBeenCalledWith('https://github.com/NousResearch/hermes-agent/releases')
    expect(openExternal).not.toHaveBeenCalledWith('https://hermes-agent.nousresearch.com/')
  })

  it('uses the AI Evolution release page for out-of-sync installer recovery', async () => {
    const versionState = {
      appVersion: '0.17.2',
      backendVersion: '0.17.3',
      electronVersion: '40.10.2',
      hermesRoot: '/tmp/hermes-agent',
      nodeVersion: '24.11.0',
      platform: 'linux',
      bundleOutOfSync: true,
      bundleSwapPending: false
    }

    $desktopVersion.set(versionState)
    const { openExternal } = renderAbout('en', versionState)

    const installer = await screen.findByRole<HTMLAnchorElement>('link', { name: /Get the installer/i })
    expect(installer.href).toBe('https://github.com/aievolutionpl/hermes-agent/releases/latest')
    installer.click()
    expect(openExternal).toHaveBeenCalledWith('https://github.com/aievolutionpl/hermes-agent/releases/latest')
    expect(openExternal).not.toHaveBeenCalledWith('https://hermes-agent.nousresearch.com/')
  })

  it('uses typed Polish and Chinese About copy for product attribution', async () => {
    renderAbout('pl')
    expect(await screen.findByRole('heading', { name: 'AI Evolution Jarvis' })).not.toBeNull()
    expect(screen.getByText('Hermes Agent jest rozpowszechniany na licencji MIT.')).not.toBeNull()

    cleanup()
    renderAbout('zh')
    expect(await screen.findByRole('heading', { name: 'AI Evolution Jarvis' })).not.toBeNull()
    expect(screen.getByText('Hermes Agent 基于 MIT 许可证分发。')).not.toBeNull()
  })
})

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, type Mock, vi } from 'vitest'

import type { DesktopVersionInfo } from '@/global'
import { I18nProvider } from '@/i18n'
import { $previewTabs, closeRightRail } from '@/store/preview'
import { $updateStatus } from '@/store/updates'

import { VersionDetails } from './version-details'

afterEach((): void => {
  cleanup()
  $updateStatus.set(null)
  closeRightRail()
  vi.unstubAllGlobals()
})

const baseVersion: DesktopVersionInfo = {
  appVersion: '0.19.0',
  electronVersion: '37.0.0',
  hermesRoot: '/tmp/hermes',
  nodeVersion: '22.0.0',
  platform: 'linux'
}

describe('VersionDetails', () => {
  interface VersionCase {
    version: Partial<DesktopVersionInfo>
    visible: string[]
    absent?: string[]
  }

  const cases: VersionCase[] = [
    { version: { source: 'ci', branch: null }, visible: ['Build Origin', 'CI'] },
    { version: { source: 'ci', branch: 'unknown' }, visible: ['CI (unknown)'] },
    { version: { source: 'nix', distribution: 'nix' }, visible: ['Build Origin', 'Nix', 'Distribution'] },
    { version: { source: 'ci', distribution: 'docker' }, visible: ['CI', 'Distribution', 'Docker'] },
    {
      version: { distribution: 'desktop-app', hermesRuntime: { type: 'embedded' } },
      visible: ['Runtime', 'Embedded runtime']
    },
    {
      version: { hermesRuntime: { type: 'external', source: { type: 'git', root: '/home/u/.hermes/hermes-agent' } } },
      visible: ['Runtime', 'git (/home/u/.hermes/hermes-agent)'],
      absent: ['External (uses the machine runtime)']
    },
    { version: { hermesRuntime: { type: 'external' } }, visible: ['Runtime', 'External (uses the machine runtime)'] },
    {
      version: { distribution: 'desktop-app', updateMechanism: 'microsoft-store' },
      visible: ['Distribution', 'Microsoft Store'],
      absent: ['Desktop app (MSIX)']
    },
    {
      version: { distribution: 'desktop-app', updateMechanism: 'app-installer', payload: 'bundled' },
      visible: ['Desktop app (MSIX)'],
      absent: ['Microsoft Store']
    },
    {
      version: { distribution: 'desktop-app', updateMechanism: 'electron-updater', payload: 'bundled' },
      visible: ['Desktop app'],
      absent: ['Desktop app (MSIX)']
    },
    // Old-style installer shell (bootstrap artifact over a managed checkout —
    // e.g. iris's v0.17.6 .app): named as the installer, never as MSIX.
    {
      version: { distribution: 'desktop-app', updateMechanism: 'self', payload: 'bootstrap' },
      visible: ['Desktop app (installer)'],
      absent: ['Desktop app (MSIX)', 'Microsoft Store']
    },
    // `hermes desktop` packs the same bootstrap payload from a source checkout;
    // a locally built stamp names the source install, never the installer.
    {
      version: {
        distribution: 'desktop-app',
        updateMechanism: 'self',
        payload: 'bootstrap',
        source: 'local',
        installedByScript: true
      },
      visible: ['Source (install script) + hermes desktop'],
      absent: ['Desktop app (installer)']
    },
    {
      version: { distribution: 'desktop-app', updateMechanism: 'self', payload: 'bootstrap', source: 'local' },
      visible: ['Source + hermes desktop'],
      absent: ['Desktop app (installer)']
    },
    // install.sh / install.ps1 checkout (receipt present) vs a manual git
    // clone (live provenance, no receipt): both honestly say Source.
    { version: { installedByScript: true }, visible: ['Source (install script)'] },
    { version: { source: 'git' }, visible: ['Distribution', 'Source'], absent: ['Source (install script)'] },
    { version: {}, visible: ['Version'], absent: ['Distribution'] }
  ]

  it.each(cases)('renders $version', ({ version, visible, absent = [] }: VersionCase): void => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VersionDetails version={{ ...baseVersion, ...version }} />
      </I18nProvider>
    )

    for (const text of visible) {
      expect(screen.getAllByText(text).length).toBeGreaterThan(0)
    }

    for (const text of absent) {
      expect(screen.queryByText(text)).toBeNull()
    }

    if (version.source === 'nix') {
      expect(screen.getAllByText('Nix')).toHaveLength(2)
    }
  })

  it('opens the commit URL via the system-browser bridge without opening a preview tab', async () => {
    const openExternal: Mock<Window['hermesDesktop']['openExternal']> = vi
      .fn<Window['hermesDesktop']['openExternal']>()
      .mockResolvedValue(undefined)

    vi.stubGlobal('hermesDesktop', { openExternal } satisfies Pick<Window['hermesDesktop'], 'openExternal'>)

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VersionDetails version={{ ...baseVersion, commit: 'd233b6d7a9c5b79288e48dfb3b29e2ead106ac73' }} />
      </I18nProvider>
    )

    fireEvent.click(screen.getByText('d233b6d7a9c5b7'))

    await waitFor(() => {
      expect(openExternal).toHaveBeenCalledWith(
        'https://github.com/NousResearch/hermes-agent/commit/d233b6d7a9c5b79288e48dfb3b29e2ead106ac73'
      )
    })
    expect($previewTabs.get()).toHaveLength(0)
  })

  it('names the source channel, and offers the change action only where the caller passes one', () => {
    const render_ = (action?: boolean) =>
      render(
        <I18nProvider>
          <VersionDetails channelAction={action ? <button type="button">Change</button> : undefined} version={baseVersion} />
        </I18nProvider>
      )

    $updateStatus.set({ supported: true, mechanism: 'posix-handoff', channel: 'stable' })
    render_()
    expect(screen.getByText('Channel')).toBeTruthy()
    expect(screen.getByText('Stable releases')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Change' })).toBeNull()
    cleanup()

    $updateStatus.set({ supported: true, mechanism: 'windows-handoff', branch: 'main' })
    render_(true)
    expect(screen.getByText(/Every commit \(main\)/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Change' })).toBeTruthy()
    cleanup()

    $updateStatus.set({ supported: true, mechanism: 'posix-handoff', branch: 'feature/gui' })
    render_(true)
    expect(screen.getByText(/Branch: feature\/gui/)).toBeTruthy()
    cleanup()

    // A packaged install's channel is the package's, not a source track: no row.
    $updateStatus.set({ supported: true, mechanism: 'electron-updater', channel: 'stable' })
    render_(true)
    expect(screen.queryByText('Channel')).toBeNull()
  })
})

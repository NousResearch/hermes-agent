import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, type Mock, vi } from 'vitest'

import type { DesktopVersionInfo } from '@/global'
import { I18nProvider } from '@/i18n'
import { $previewTabs, closeRightRail } from '@/store/preview'

import { VersionDetails } from './version-details'

afterEach((): void => {
  cleanup()
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
  interface VersionCase { version: Partial<DesktopVersionInfo>; visible: string[]; absent?: string[] }

  const cases: VersionCase[] = [
    { version: { source: 'ci', branch: null }, visible: ['Build Origin', 'CI'] },
    { version: { source: 'ci', branch: 'unknown' }, visible: ['CI (unknown)'] },
    { version: { source: 'nix', distribution: 'nix' }, visible: ['Build Origin', 'Nix', 'Distribution'] },
    { version: { source: 'ci', distribution: 'docker' }, visible: ['CI', 'Distribution', 'Docker'] },
    { version: { distribution: 'desktop-app', hermesRuntime: { type: 'embedded' } }, visible: ['Runtime', 'Embedded runtime'] },
    { version: { hermesRuntime: { type: 'external', source: { type: 'git', root: '/home/u/.hermes/hermes-agent' } } },
      visible: ['Runtime', 'git (/home/u/.hermes/hermes-agent)'], absent: ['External (uses the machine runtime)'] },
    { version: { hermesRuntime: { type: 'external' } }, visible: ['Runtime', 'External (uses the machine runtime)'] },
    { version: { distribution: 'desktop-app', updateMechanism: 'microsoft-store' },
      visible: ['Distribution', 'Microsoft Store'], absent: ['Desktop app (MSIX)'] },
    { version: { distribution: 'desktop-app', updateMechanism: 'app-installer' },
      visible: ['Desktop app (MSIX)'], absent: ['Microsoft Store'] }
  ]

  it.each(cases)('renders $version', ({ version, visible, absent = [] }: VersionCase): void => {
    render(<I18nProvider configClient={null} initialLocale="en"><VersionDetails version={{ ...baseVersion, ...version }} /></I18nProvider>)

    for (const text of visible) { expect(screen.getAllByText(text).length).toBeGreaterThan(0) }

    for (const text of absent) { expect(screen.queryByText(text)).toBeNull() }

    if (version.source === 'nix') { expect(screen.getAllByText('Nix')).toHaveLength(2) }
  })

  it('opens the commit URL via the system-browser bridge without opening a preview tab', async () => {
    const openExternal: Mock<Window['hermesDesktop']['openExternal']> = vi.fn<Window['hermesDesktop']['openExternal']>().mockResolvedValue(undefined)
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
})

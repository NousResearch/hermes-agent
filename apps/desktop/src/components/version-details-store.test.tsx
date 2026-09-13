import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { DesktopVersionInfo } from '@/global'
import { I18nProvider } from '@/i18n'

import { VersionDetails } from './version-details'

const baseVersion: DesktopVersionInfo = {
  appVersion: '0.19.0',
  electronVersion: '37.0.0',
  hermesRoot: '/tmp/hermes',
  nodeVersion: '22.0.0',
  platform: 'linux'
}

afterEach(cleanup)

describe('VersionDetails Store identity label', () => {
  it('labels a Store-identity build from the stamp mechanism, not windowsStore', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VersionDetails version={{ ...baseVersion, distribution: 'desktop-app', updateMechanism: 'microsoft-store' }} />
      </I18nProvider>
    )

    expect(screen.getByText('Distribution')).toBeTruthy()
    expect(screen.getByText('Microsoft Store')).toBeTruthy()
    expect(screen.queryByText('Desktop app (MSIX)')).toBeNull()
  })

  it('keeps the MSIX label for the out-of-store app-installer build', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VersionDetails version={{ ...baseVersion, distribution: 'desktop-app', updateMechanism: 'app-installer' }} />
      </I18nProvider>
    )

    expect(screen.getByText('Desktop app (MSIX)')).toBeTruthy()
    expect(screen.queryByText('Microsoft Store')).toBeNull()
  })
})

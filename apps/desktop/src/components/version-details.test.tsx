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

describe('VersionDetails', () => {
  it('labels a missing branch explicitly', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VersionDetails version={{ ...baseVersion, branch: null }} />
      </I18nProvider>
    )

    expect(screen.getByText('Branch')).toBeTruthy()
    expect(screen.getByText('No branch information')).toBeTruthy()
  })

  it('preserves a literal branch named unknown', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VersionDetails version={{ ...baseVersion, branch: 'unknown' }} />
      </I18nProvider>
    )

    expect(screen.getByText('unknown')).toBeTruthy()
    expect(screen.queryByText('No branch information')).toBeNull()
  })

  it('shows the Nix source and distribution from the stamp', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VersionDetails version={{ ...baseVersion, source: 'nix', distribution: 'nix' }} />
      </I18nProvider>
    )

    expect(screen.getByText('Source')).toBeTruthy()
    expect(screen.getAllByText('Nix')).toHaveLength(2)
    expect(screen.getByText('Distribution')).toBeTruthy()
  })

  it('distinguishes CI provenance from the Docker distribution', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VersionDetails version={{ ...baseVersion, source: 'ci', distribution: 'docker' }} />
      </I18nProvider>
    )

    expect(screen.getByText('CI')).toBeTruthy()
    expect(screen.getByText('Distribution')).toBeTruthy()
    expect(screen.getByText('Docker')).toBeTruthy()
  })
})

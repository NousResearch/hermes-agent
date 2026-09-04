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
  it('omits the branch suffix when no branch is present', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VersionDetails version={{ ...baseVersion, source: 'ci', branch: null }} />
      </I18nProvider>
    )

    expect(screen.getByText('Build Origin')).toBeTruthy()
    expect(screen.getByText('CI')).toBeTruthy()
  })

  it('shows a literal branch named unknown inline', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VersionDetails version={{ ...baseVersion, source: 'ci', branch: 'unknown' }} />
      </I18nProvider>
    )

    expect(screen.getByText('CI (unknown)')).toBeTruthy()
  })

  it('shows the Nix source and distribution from the stamp', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VersionDetails version={{ ...baseVersion, source: 'nix', distribution: 'nix' }} />
      </I18nProvider>
    )

    expect(screen.getByText('Build Origin')).toBeTruthy()
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

  it('shows the runtime row for an embedded build running its payload', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VersionDetails
          version={{
            ...baseVersion,
            distribution: 'desktop-app',
            hermesRuntime: { type: 'embedded' }
          }}
        />
      </I18nProvider>
    )

    expect(screen.getByText('Runtime')).toBeTruthy()
    expect(screen.getByText('Embedded runtime')).toBeTruthy()
  })

  it('shows the runtime source with its location when an external build runs a machine runtime', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VersionDetails
          version={{
            ...baseVersion,
            hermesRuntime: { type: 'external', source: { type: 'git', root: '/home/u/.hermes/hermes-agent' } }
          }}
        />
      </I18nProvider>
    )

    expect(screen.getByText('Runtime')).toBeTruthy()
    expect(screen.getByText('git (/home/u/.hermes/hermes-agent)')).toBeTruthy()
    expect(screen.queryByText('External (uses the machine runtime)')).toBeNull()
  })

  it('shows the generic external label before the first backend spawn', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <VersionDetails version={{ ...baseVersion, hermesRuntime: { type: 'external' } }} />
      </I18nProvider>
    )

    expect(screen.getByText('Runtime')).toBeTruthy()
    expect(screen.getByText('External (uses the machine runtime)')).toBeTruthy()
  })
})

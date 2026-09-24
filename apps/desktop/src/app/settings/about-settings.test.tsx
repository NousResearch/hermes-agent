// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { AboutSettings } from './about-settings'

const mocks = vi.hoisted(() => ({
  startActiveUpdate: vi.fn()
}))

vi.mock('@nanostores/react', () => ({
  useStore: (store: { value: unknown }) => store.value
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        about: {
          automaticUpdates: 'Automatic updates',
          automaticUpdatesDesc: '',
          branchCommit: () => '',
          cantReach: '',
          cantUpdate: '',
          checkNow: 'Check now',
          checking: 'Checking',
          daysAgo: () => '',
          heading: 'About',
          hoursAgo: () => '',
          installing: 'Installing',
          justNow: 'just now',
          justNowSuffix: '',
          lastChecked: () => '',
          minAgo: () => '',
          never: 'never',
          onLatest: 'Up to date',
          releaseNotes: 'Release notes',
          seeWhatsNew: "See what's new",
          tapCheck: 'Check for updates',
          updateNow: 'Update now',
          updateReady: () => '',
          updateReadyUnknown: '',
          version: () => '',
          versionUnavailable: '',
          bundleOutOfSync: '',
          bundleOutOfSyncAction: '',
          bundleOutOfSyncDesc: '',
          bundleSwapPending: '',
          bundleSwapPendingAction: '',
          bundleSwapPendingDesc: ''
        }
      }
    }
  })
}))

vi.mock('@/store/updates', () => ({
  $desktopVersion: { value: null },
  $updateApply: { value: { applying: false, stage: 'idle' } },
  $updateChecking: { value: false },
  $updateStatus: { value: { behind: 1, supported: true } },
  checkUpdates: vi.fn(),
  openUpdatesWindow: vi.fn(),
  refreshDesktopVersion: vi.fn(),
  startActiveUpdate: (...args: unknown[]) => mocks.startActiveUpdate(...args)
}))

vi.mock('@/components/brand-mark', () => ({ BrandMark: () => null }))
vi.mock('@/components/ui/button', () => ({
  Button: ({
    asChild: _asChild,
    children,
    ...props
  }: React.ButtonHTMLAttributes<HTMLButtonElement> & { asChild?: boolean }) => <button {...props}>{children}</button>
}))
vi.mock('@/components/ui/codicon', () => ({ Codicon: () => null }))
vi.mock('@/lib/icons', () => ({
  AlertTriangle: () => null,
  CheckCircle2: () => null,
  ExternalLink: () => null,
  Loader2: () => null,
  RefreshCw: () => null
}))
vi.mock('@/lib/utils', () => ({ cn: () => '' }))
vi.mock('./primitives', () => ({
  ListRow: () => null,
  SectionHeading: () => null,
  SettingsContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
}))
vi.mock('./settings-manifest', () => ({
  SETTING_IDS: { about: { automaticUpdates: 'automatic-updates' } },
  settingElementId: (id: string) => id
}))
vi.mock('./uninstall-section', () => ({ UninstallSection: () => null }))
vi.mock('./use-setting-deep-link', () => ({ useSettingDeepLink: () => undefined }))

describe('AboutSettings', () => {
  beforeEach(() => mocks.startActiveUpdate.mockReset())

  it('applies the local Desktop update without using the fleet update default', () => {
    render(<AboutSettings />)

    fireEvent.click(screen.getByRole('button', { name: 'Update now' }))

    expect(mocks.startActiveUpdate).toHaveBeenCalledWith('client')
  })
})

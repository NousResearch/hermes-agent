// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { RepoScanSetting } from './repo-scan-setting'

const mocks = vi.hoisted(() => ({
  loadedConfig: {} as Record<string, unknown> | undefined,
  scan: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        repoScan: {
          title: 'Repository Discovery Scan',
          hint: 'The same scan the sidebar runs.',
          scanNow: 'Scan now',
          scanning: 'Scanning...',
          found: (count: number) => `Found ${count} repositories.`,
          foundOne: 'Found 1 repository.',
          disabled: 'Turn on Automatic Repository Discovery to scan.',
          noRoots: 'Nothing to scan: set a Working Directory above, or add a discovery root.',
          noBridge: 'This build cannot read the local disk, so there is nothing to scan.',
          rejected: 'The backend refused the scan: its discovery settings differ from this page.',
          failed: 'The scan failed. Try again.',
          failedWith: (detail: string) => `The scan failed: ${detail}`
        }
      }
    }
  })
}))

vi.mock('@/store/projects', () => ({
  $reposScanning: atom(false),
  scanAndRecordRepos: (force?: boolean) => mocks.scan(force)
}))

vi.mock('../hooks/use-config-record', () => ({
  useHermesConfigRecord: () => ({ data: mocks.loadedConfig })
}))

beforeEach(() => {
  mocks.loadedConfig = { desktop: { repo_scan_enabled: true } }
  mocks.scan.mockResolvedValue({ found: 2, reason: 'ok' })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('RepoScanSetting', () => {
  it('runs a forced scan and reports what it found', async () => {
    render(<RepoScanSetting />)

    expect(screen.getByText('Repository Discovery Scan')).toBeTruthy()
    expect(screen.getByText('The same scan the sidebar runs.')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Scan now' }))

    await waitFor(() => expect(screen.getByText('Found 2 repositories.')).toBeTruthy())
    expect(mocks.scan).toHaveBeenCalledWith(true)
  })

  it('says why the scan came up empty', async () => {
    mocks.scan.mockResolvedValue({ reason: 'no-roots' })

    render(<RepoScanSetting />)
    fireEvent.click(screen.getByRole('button', { name: 'Scan now' }))

    await waitFor(() =>
      expect(screen.getByText('Nothing to scan: set a Working Directory above, or add a discovery root.')).toBeTruthy()
    )
  })

  it('quotes the error when the scan throws', async () => {
    mocks.scan.mockResolvedValue({
      detail: 'invalid params for projects.record_repos: discovery_policy.nestedd: Extra inputs are not permitted',
      reason: 'failed'
    })

    render(<RepoScanSetting />)
    fireEvent.click(screen.getByRole('button', { name: 'Scan now' }))

    await waitFor(() =>
      expect(
        screen.getByText(
          'The scan failed: invalid params for projects.record_repos: discovery_policy.nestedd: Extra inputs are not permitted'
        )
      ).toBeTruthy()
    )
  })

  it('is unavailable while discovery is off', () => {
    mocks.loadedConfig = { desktop: { repo_scan_enabled: false } }

    render(<RepoScanSetting />)

    expect((screen.getByRole('button', { name: 'Scan now' }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByText('Turn on Automatic Repository Discovery to scan.')).toBeTruthy()
  })
})

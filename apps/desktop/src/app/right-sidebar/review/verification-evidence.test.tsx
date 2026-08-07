import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import {
  $activeSessionId,
  $busy,
  $currentCwd,
  $selectedStoredSessionId,
  setActiveSessionId,
  setCurrentCwd
} from '@/store/session'
import { $workspaceChangeTick } from '@/store/workspace-events'

import { VerificationEvidencePanel } from './verification-evidence'

const mocks = vi.hoisted(() => ({
  requestComposerSubmit: vi.fn(),
  requestGateway: vi.fn()
}))

vi.mock('@/app/chat/composer/focus', () => ({ requestComposerSubmit: mocks.requestComposerSubmit }))
vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: mocks.requestGateway })
}))

const renderPanel = () =>
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <VerificationEvidencePanel />
    </I18nProvider>
  )

describe('VerificationEvidencePanel', () => {
  beforeEach(() => {
    mocks.requestComposerSubmit.mockReset()
    mocks.requestGateway.mockReset()
    setCurrentCwd('/repo')
    setActiveSessionId('session-1')
    $selectedStoredSessionId.set(null)
    $busy.set(false)
    $workspaceChangeTick.set(0)
  })

  afterEach(() => {
    setCurrentCwd('')
    setActiveSessionId(null)
    $selectedStoredSessionId.set(null)
    $busy.set(false)
    $workspaceChangeTick.set(0)
  })

  it('reads and renders fresh verification evidence through the real gateway contract', async () => {
    mocks.requestGateway.mockResolvedValue({
      verification: {
        changed_paths: [],
        evidence: {
          canonical_command: 'npm test',
          created_at: new Date(Date.now() - 60_000).toISOString(),
          exit_code: 0,
          kind: 'test',
          scope: 'full'
        },
        root: '/repo',
        session_id: 'session-1',
        status: 'passed'
      }
    })

    renderPanel()

    expect(await screen.findByText('Passed')).toBeTruthy()
    expect(screen.getByText('npm test')).toBeTruthy()
    expect(screen.getByText(/Full/)).toBeTruthy()
    expect(mocks.requestGateway).toHaveBeenCalledWith('verification.status', {
      cwd: '/repo',
      session_id: 'session-1'
    })

    fireEvent.click(screen.getByRole('button', { name: 'Verify again' }))
    expect(mocks.requestComposerSubmit).toHaveBeenCalledWith(expect.stringContaining('verification checks'), {
      target: 'main'
    })
  })

  it('requests with the stored session key and falls back to the runtime id', async () => {
    mocks.requestGateway.mockResolvedValue({ verification: { evidence: null, status: 'unverified' } })
    $selectedStoredSessionId.set('20260720_101530_abc123')

    renderPanel()

    // Both atoms set: the durable stored key (the gateway ledger's key) wins.
    await waitFor(() =>
      expect(mocks.requestGateway).toHaveBeenCalledWith('verification.status', {
        cwd: '/repo',
        session_id: '20260720_101530_abc123'
      })
    )

    mocks.requestGateway.mockClear()
    $selectedStoredSessionId.set(null)

    // Stored key empty: fall back to the runtime session id.
    await waitFor(() =>
      expect(mocks.requestGateway).toHaveBeenCalledWith('verification.status', {
        cwd: '/repo',
        session_id: 'session-1'
      })
    )
  })

  it('makes stale and failed proof explicit without rendering raw command output', async () => {
    mocks.requestGateway.mockResolvedValue({
      verification: {
        changed_paths: ['src/a.ts', 'src/b.ts'],
        evidence: {
          canonical_command: 'npm run typecheck',
          created_at: new Date(Date.now() - 120_000).toISOString(),
          exit_code: 1,
          kind: 'typecheck',
          output_summary: 'SECRET OUTPUT MUST NOT RENDER',
          scope: 'targeted'
        },
        status: 'stale'
      }
    })

    renderPanel()

    expect(await screen.findByText('Stale')).toBeTruthy()
    expect(screen.getByText('2 paths changed since this check')).toBeTruthy()
    expect(screen.queryByText('SECRET OUTPUT MUST NOT RENDER')).toBeNull()

    mocks.requestGateway.mockResolvedValue({
      verification: {
        changed_paths: [],
        evidence: {
          canonical_command: 'npm run typecheck',
          created_at: new Date().toISOString(),
          exit_code: 1,
          kind: 'typecheck',
          scope: 'full'
        },
        status: 'failed'
      }
    })
    $workspaceChangeTick.set(1)

    expect(await screen.findByText('Failed')).toBeTruthy()
    expect(screen.getByText(/Exit 1/)).toBeTruthy()
  })

  it('refreshes when a turn settles and stays inert without a session or workspace', async () => {
    mocks.requestGateway.mockResolvedValue({ verification: { evidence: null, status: 'unverified' } })

    renderPanel()
    expect(await screen.findByText('Unverified')).toBeTruthy()
    expect(mocks.requestGateway).toHaveBeenCalledTimes(1)

    $busy.set(true)
    await Promise.resolve()
    expect(mocks.requestGateway).toHaveBeenCalledTimes(1)

    $busy.set(false)
    await waitFor(() => expect(mocks.requestGateway).toHaveBeenCalledTimes(2))

    mocks.requestGateway.mockClear()
    $activeSessionId.set(null)
    $currentCwd.set('')
    await Promise.resolve()
    expect(mocks.requestGateway).not.toHaveBeenCalled()
  })

  it('keeps the previous result visible during a background refresh', async () => {
    const evidence = {
      canonical_command: 'npm test',
      created_at: new Date().toISOString(),
      exit_code: 0,
      kind: 'test',
      scope: 'full'
    }

    mocks.requestGateway.mockResolvedValue({ verification: { changed_paths: [], evidence, status: 'passed' } })

    renderPanel()
    expect(await screen.findByText('Passed')).toBeTruthy()

    let resolveRefresh: (value: unknown) => void = () => {}

    mocks.requestGateway.mockImplementation(() => new Promise(resolve => (resolveRefresh = resolve)))
    $workspaceChangeTick.set(1)

    // Same cwd + session: no loading flash, the old result holds until the
    // in-flight response lands.
    await waitFor(() => expect(mocks.requestGateway).toHaveBeenCalledTimes(2))
    expect(screen.getByText('Passed')).toBeTruthy()
    expect(screen.queryByText('Checking')).toBeNull()

    resolveRefresh({ verification: { changed_paths: ['src/a.ts'], evidence, status: 'stale' } })
    expect(await screen.findByText('Stale')).toBeTruthy()
  })

  it('does not show the previous session evidence while a new session is busy', async () => {
    mocks.requestGateway.mockResolvedValue({
      verification: {
        changed_paths: [],
        evidence: {
          canonical_command: 'npm test',
          created_at: new Date().toISOString(),
          exit_code: 0,
          kind: 'test',
          scope: 'full'
        },
        status: 'passed'
      }
    })

    renderPanel()
    expect(await screen.findByText('Passed')).toBeTruthy()

    $busy.set(true)
    $activeSessionId.set('session-2')

    expect(await screen.findByText('Checking')).toBeTruthy()
    expect(screen.queryByText('Passed')).toBeNull()
    expect(mocks.requestGateway).toHaveBeenCalledTimes(1)
  })

  it('does not claim a passing status from malformed evidence', async () => {
    mocks.requestGateway.mockResolvedValue({
      verification: {
        evidence: { canonical_command: 'npm test' },
        status: 'passed'
      }
    })

    renderPanel()

    expect(await screen.findByText('Unavailable')).toBeTruthy()
    expect(screen.queryByText('Passed')).toBeNull()
  })
})

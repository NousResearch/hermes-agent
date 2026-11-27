import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopCloudAgent } from '@/global'
import type * as AgentTransfer from '@/lib/agent-transfer'
import type { TransferOptions } from '@/lib/agent-transfer'

import { BackupSettings } from './backup-settings'

const AGENT: DesktopCloudAgent = {
  dashboardGatewayState: 'active',
  dashboardUrl: 'https://agent-1.cloud.example',
  id: 'agent-1',
  name: 'Agent One',
  status: 'running'
}

const mocks = vi.hoisted(() => ({
  signedIn: true,
  transferAgent: vi.fn(),
  undoTransfer: vi.fn(async () => undefined)
}))

vi.mock('@/lib/hermes-cloud-agents', () => ({
  backup: vi.fn(),
  cloudLogin: vi.fn(),
  cloudStatus: async () => ({ portalBaseUrl: '', signedIn: mocks.signedIn }),
  connectCloudAgent: async () => 'cloud-conn',
  discoverCloudAgents: async () => ({ agents: [AGENT], needsOrgSelection: false, org: null, orgs: [] }),
  PORTAL_AGENTS_URL: 'https://portal.example/agents',
  transferAvailable: () => true,
  transferRequest: vi.fn()
}))

vi.mock('@/lib/agent-transfer', async importOriginal => ({
  ...(await importOriginal<typeof AgentTransfer>()),
  transferAgent: mocks.transferAgent,
  undoTransfer: mocks.undoTransfer
}))

afterEach(() => {
  cleanup()
  mocks.transferAgent.mockReset()
  mocks.undoTransfer.mockClear()
  mocks.signedIn = true
  window.document.documentElement.removeAttribute('data-hermes-desktop-host')
})

const renderPage = () =>
  render(
    <MemoryRouter initialEntries={['/settings?tab=backup']}>
      <BackupSettings />
    </MemoryRouter>
  )

async function startTransfer() {
  fireEvent.click(await screen.findByRole('button', { name: /Agent One/ }))
  fireEvent.click(screen.getByRole('checkbox'))
  fireEvent.click(screen.getByRole('button', { name: 'Replace Agent One' }))
}

describe('Transfer to Hermes Cloud', () => {
  it('Undo after a retried failed restore puts back the FIRST undo point, not the half-restored one', async () => {
    const { TransferRestoreError } = await vi.importActual<typeof AgentTransfer>('@/lib/agent-transfer')

    for (const archive of ['/cloud/first.zip', '/cloud/half-restored.zip']) {
      mocks.transferAgent.mockImplementationOnce(async (...args: unknown[]) => {
        ;(args[4] as TransferOptions).onUndoArchive?.(archive)

        throw new TransferRestoreError('Error: disk full', archive)
      })
    }

    renderPage()
    await startTransfer()
    await screen.findByText(/may be partly replaced/)
    expect(screen.getByRole('button', { name: 'Undo' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(mocks.transferAgent).toHaveBeenCalledTimes(2))
    fireEvent.click(await screen.findByRole('button', { name: 'Undo' }))

    await waitFor(() => expect(mocks.undoTransfer).toHaveBeenCalledTimes(1))
    expect(mocks.undoTransfer.mock.calls[0]).toEqual(
      expect.arrayContaining([{ connectionId: 'cloud-conn', profile: 'default' }, '/cloud/first.zip'])
    )
  })

  it('offers Cancel while copying but not once the restore was requested', async () => {
    let report: (progress: AgentTransfer.TransferProgress) => void = () => undefined
    mocks.transferAgent.mockImplementationOnce((...args: unknown[]) => {
      report = args[3] as typeof report

      return new Promise(() => undefined)
    })

    renderPage()
    await startTransfer()
    await waitFor(() => expect(mocks.transferAgent).toHaveBeenCalled())

    report({ fraction: 0.9, phase: 'upload' })
    expect(await screen.findByRole('button', { name: 'Cancel' })).toBeTruthy()

    report({ phase: 'restore' })
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Cancel' })).toBeNull())
  })

  it('sends a signed-out Webapp to Settings ▸ Providers instead of a portal window', async () => {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
    mocks.signedIn = false

    renderPage()

    expect(await screen.findByText(/Settings ▸ Providers/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open Providers' })).toBeTruthy()
  })
})

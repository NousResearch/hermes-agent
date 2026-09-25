import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'

import { CommandCenterView } from './index'

const api = vi.hoisted(() => ({
  getActionStatus: vi.fn(() =>
    Promise.resolve({ exit_code: 0, lines: [], name: 'gateway-start', pid: 1, running: false })
  ),
  getLogs: vi.fn(() => Promise.resolve({ lines: [] })),
  getStatus: vi.fn(() => Promise.resolve({ active_sessions: 0, gateway_running: false, version: 'test' })),
  getUsageAnalytics: vi.fn(() => Promise.resolve({})),
  restartGateway: vi.fn(),
  startGateway: vi.fn(() => Promise.resolve({ name: 'gateway-start', ok: true, pid: 1 })),
  stopGateway: vi.fn(),
  updateHermes: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  ...api
}))
vi.mock('./maintenance', () => ({ MaintenancePanel: () => null }))

afterEach(cleanup)

function renderCommandCenter() {
  return render(
    <MemoryRouter>
      <CommandCenterView
        initialSection="system"
        onClose={() => {}}
        onDeleteSession={() => Promise.resolve()}
        onOpenSession={() => {}}
      />
    </MemoryRouter>
  )
}

describe('Command Center gateway controls (#48189)', () => {
  it('offers Start when the messaging gateway is stopped', async () => {
    renderCommandCenter()

    expect(await screen.findByRole('button', { name: 'Start gateway' })).toBeTruthy()
  })

  it('starts the gateway and refreshes the authoritative status', async () => {
    renderCommandCenter()

    fireEvent.click(await screen.findByRole('button', { name: 'Start gateway' }))

    await waitFor(() => expect(api.startGateway).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(api.getStatus).toHaveBeenCalledTimes(2))
  })
})

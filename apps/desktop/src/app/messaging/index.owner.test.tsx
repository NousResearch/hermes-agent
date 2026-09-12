import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import type { HermesApiRequest } from '@/global'
import { $settingsScopeOverride as selection } from '@/store/settings-scope'
import type { MessagingPlatformInfo } from '@/types/hermes'

const requests: HermesApiRequest[] = []
let readPlatforms: (request: HermesApiRequest) => Promise<{ platforms: MessagingPlatformInfo[] }>

function platform(name: string): MessagingPlatformInfo {
  return {
    configured: false,
    description: 'Fixture platform',
    docs_url: '',
    enabled: false,
    env_vars: [],
    gateway_running: false,
    id: 'fixture-platform',
    name,
    state: 'disabled'
  }
}

beforeEach(() => {
  requests.length = 0
  selection.set({ connectionId: 'local', profile: 'shared-name' })

  setApiRequestConnection('fixture-ambient')
  readPlatforms = async () => ({ platforms: [platform('Fixture platform')] })
  vi.stubGlobal('hermesDesktop', {
    api: vi.fn(async (request: HermesApiRequest) => {
      requests.push(request)

      if (request.path === '/api/profiles') {
        return { profiles: [] }
      }

      if (request.path === '/api/messaging/platforms') {
        return readPlatforms(request)
      }

      if (request.path === '/api/pairing') {
        return { approved: [], pending: [] }
      }

      if (request.path === '/api/gateway/restart') {
        return { ok: true, name: 'gateway-restart' }
      }

      if (request.path.startsWith('/api/actions/')) {
        return { running: false, exit_code: 0 }
      }

      return { ok: true }
    })
  })
})

afterEach(() => {
  cleanup()
  setApiRequestConnection(null)
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

async function mount() {
  const { MessagingView } = await import('./index')

  let result!: ReturnType<typeof render>
  await act(async () => {
    result = render(
      <MemoryRouter>
        <MessagingView />
      </MemoryRouter>
    )
  })

  return result
}

it('cannot publish an old gateway response into the newly selected same-named profile', async () => {
  let finishLocal!: (value: { platforms: MessagingPlatformInfo[] }) => void
  readPlatforms = request =>
    request.connectionId === 'local'
      ? new Promise(resolve => {
          finishLocal = resolve
        })
      : Promise.resolve({ platforms: [platform('Remote fixture')] })
  await mount()
  await waitFor(() => expect(finishLocal).toBeTypeOf('function'))
  await act(async () => {
    selection.set({ connectionId: 'fixture-remote', profile: 'shared-name' })
  })
  await screen.findAllByText('Remote fixture')
  await act(async () => {
    finishLocal({ platforms: [platform('Local fixture')] })
  })
  expect(screen.queryAllByText('Local fixture')).toHaveLength(0)
  expect(screen.getAllByText('Remote fixture').length).toBeGreaterThan(0)
})

it('routes an explicit-owner restart and its status polling to the same gateway', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  await mount()
  await screen.findByRole('switch')
  await act(async () => {
    fireEvent.click(screen.getByRole('switch'))
  })
  await screen.findByRole('button', { name: 'Restart now' })
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Restart now' }))
  })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1500)
  })

  const operations = requests.filter(
    request => request.path === '/api/gateway/restart' || request.path.startsWith('/api/actions/')
  )

  expect(operations.length).toBeGreaterThan(1)

  for (const request of operations) {
    expect(request).toMatchObject({ connectionId: 'local', profile: 'shared-name' })
  }
})

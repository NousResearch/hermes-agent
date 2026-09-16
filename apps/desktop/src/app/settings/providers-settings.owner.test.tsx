import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { ConfirmHost } from '@/components/confirm-host'
import type { HermesApiRequest, HermesConnection } from '@/global'
import { $confirmRequest } from '@/store/confirm'
import { $connection } from '@/store/session'
import { $settingsScopeKey as ownerKey, $settingsScopeOverride as selection } from '@/store/settings-scope'

const calls: HermesApiRequest[] = []
let terminalManaged = false

beforeEach(() => {
  terminalManaged = false
  selection.set(null)
  $connection.set(null)
  calls.length = 0
  setApiRequestConnection('fixture-ambient')
  setApiRequestProfile('shared-name')
  vi.stubGlobal('hermesDesktop', {
    terminal: {},
    api: vi.fn(async (request: HermesApiRequest) => {
      calls.push(request)

      if (request.path === '/api/profiles') {
        return { profiles: [] }
      }

      if (request.path === '/api/env') {
        return {}
      }

      if (request.path === '/api/providers/oauth') {
        return {
          providers: [
            {
              id: 'fixture-provider',
              name: 'Fixture Provider',
              flow: terminalManaged ? 'external' : 'device_code',
              disconnectable: !terminalManaged,
              disconnect_command: terminalManaged ? 'fixture logout' : undefined,
              cli_command: '',
              docs_url: '',
              status: { logged_in: true }
            }
          ]
        }
      }

      return { ok: true }
    })
  })
})

afterEach(() => {
  cleanup()
  $confirmRequest.set(null)
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
})

async function mount(removable = true) {
  const { ProvidersSettings } = await import('./providers-settings')

  function Page() {
    const key = useStore(ownerKey)

    return <ProvidersSettings key={key} onClose={() => {}} onViewChange={() => {}} view="accounts" />
  }

  render(
    <>
      <Page />
      <ConfirmHost />
    </>
  )
  await screen.findByRole('button', { name: removable ? 'Remove Fixture Provider' : 'Disconnect Fixture Provider' })
}

it('visibly disables ambient-only terminal removal for an explicit owner', async () => {
  terminalManaged = true
  selection.set({ connectionId: 'local', profile: 'shared-name' })
  await mount(false)
  expect((screen.getByRole('button', { name: 'Disconnect Fixture Provider' }) as HTMLButtonElement).disabled).toBe(true)
})

it('does not let a deferred removal confirmation target a new ambient gateway', async () => {
  await mount()
  fireEvent.click(screen.getByRole('button', { name: 'Remove Fixture Provider' }))
  await screen.findByRole('button', { name: 'Disconnect' })
  await act(async () => {
    setApiRequestConnection('fixture-new-ambient')
    $connection.set({ baseUrl: 'https://fixture-new.example', profile: 'default', mode: 'remote' } as HermesConnection)
  })
  await screen.findByRole('button', { name: 'Remove Fixture Provider', hidden: true })
  fireEvent.click(screen.getByRole('button', { name: 'Disconnect' }))
  await waitFor(() => expect($confirmRequest.get()).toBeNull())
  await act(async () => {})
  expect(calls.filter(request => request.method === 'DELETE')).toEqual([])
})

it('keeps account reads, removal and its refresh on the mounted owner', async () => {
  const owner = { connectionId: 'local', profile: 'shared-name' }
  selection.set(owner)
  await mount()
  fireEvent.click(screen.getByRole('button', { name: 'Remove Fixture Provider' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Disconnect' }))
  await waitFor(() => expect(calls.filter(request => request.method === 'DELETE')).toHaveLength(1))
  await waitFor(() => expect(calls.filter(request => request.path === '/api/providers/oauth')).toHaveLength(2))

  for (const request of calls.filter(request => request.path !== '/api/profiles')) {
    expect(request).toMatchObject(owner)
  }
})

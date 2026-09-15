import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { createRef } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { type ProfileScope, setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import type { HermesApiRequest, HermesConnection } from '@/global'
import { queryClient } from '@/lib/query-client'
import { $connection } from '@/store/session'
import { $settingsScopeOverride as selection } from '@/store/settings-scope'

const calls: HermesApiRequest[] = []
let write: (request: HermesApiRequest) => Promise<{ ok: boolean }> = async () => ({ ok: true })

beforeEach(() => {
  calls.length = 0
  selection.set(null)
  $connection.set(null)
  write = async () => ({ ok: true })
  queryClient.clear()
  queryClient.setDefaultOptions({ queries: { retry: false } })
  setApiRequestConnection('fixture-ambient')
  setApiRequestProfile('ambient-profile')
  vi.stubGlobal('hermesDesktop', {
    api: vi.fn(async (request: HermesApiRequest) => {
      calls.push(request)

      if (request.path === '/api/profiles') {
        return { profiles: [] }
      }

      if (request.method === 'PUT') {
        return write(request)
      }

      if (request.path === '/api/config/schema') {
        return { fields: {} }
      }

      if (request.path === '/api/config') {
        return { checkpoints: { enabled: request.connectionId === 'fixture-remote' } }
      }

      return { available: false }
    })
  })
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

async function mount(profile?: ProfileScope) {
  const { ConfigSettings } = await import('./config-settings')

  return render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <ConfigSettings activeSectionId="safety" importInputRef={createRef<HTMLInputElement>()} profile={profile} />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

it('isolates same-named owners across config/schema reads, drafts and saves', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  const local = { connectionId: 'local', profile: 'shared-name' }
  const remote = { connectionId: 'fixture-remote', profile: 'shared-name' }
  selection.set(local)

  await mount()
  const first = await screen.findByRole('switch')
  expect(first.getAttribute('aria-checked')).toBe('false')
  await act(async () => {
    selection.set(remote)
  })
  await waitFor(() => expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('true'))
  fireEvent.click(screen.getByRole('switch'))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(700)
  })
  expect(calls.filter(call => call.method === 'PUT')).toEqual([
    { ...remote, path: '/api/config', method: 'PUT', body: { config: { checkpoints: { enabled: false } } } }
  ])

  for (const owner of [local, remote]) {
    expect(calls).toContainEqual(expect.objectContaining({ ...owner, path: '/api/config' }))
    expect(calls).toContainEqual(expect.objectContaining({ ...owner, path: '/api/config/schema' }))
  }
})

it('does not seed a legacy profile draft from a previous ambient gateway cache', async () => {
  selection.set('shared-name')
  await mount()
  expect((await screen.findByRole('switch')).getAttribute('aria-checked')).toBe('false')
  await act(async () => {
    setApiRequestConnection('fixture-remote')
    $connection.set({
      baseUrl: 'https://fixture-next.example',
      profile: 'shared-name',
      mode: 'remote'
    } as HermesConnection)
  })
  await waitFor(() => expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('true'))
})

it('cancels queued ambient saves after the mounted owner changes', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  let finishWrite!: (value: { ok: boolean }) => void
  write = () =>
    new Promise(resolve => {
      finishWrite = resolve
    })
  await mount()
  fireEvent.click(await screen.findByRole('switch'))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(700)
  })
  expect(calls.filter(call => call.method === 'PUT')).toHaveLength(1)
  fireEvent.click(screen.getByRole('switch'))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(700)
  })
  await act(async () => {
    setApiRequestConnection('fixture-new-ambient')
    $connection.set({ baseUrl: 'https://fixture-new.example', profile: 'default', mode: 'remote' } as HermesConnection)
  })
  await screen.findByRole('switch')
  await act(async () => {
    finishWrite({ ok: true })
    await vi.advanceTimersByTimeAsync(700)
  })
  const writes = calls.filter(call => call.method === 'PUT')
  expect(writes).toHaveLength(1)
  expect(writes[0].connectionId).toBe('fixture-ambient')
})

it('honors an explicit null profile instead of falling back to the settings selection', async () => {
  selection.set('fixture-selected')
  await mount(null)
  await screen.findByRole('switch')
  const reads = calls.filter(call => ['/api/config', '/api/config/schema'].includes(call.path))
  expect(reads.length).toBeGreaterThan(0)

  for (const request of reads) {
    expect(request.connectionId).toBe('fixture-ambient')
    expect(request).not.toHaveProperty('profile')
  }
})

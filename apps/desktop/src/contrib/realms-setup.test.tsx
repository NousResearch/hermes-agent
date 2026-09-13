import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { type ComponentType } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { createPluginI18n } from '@/i18n/plugin-i18n'

// @ts-expect-error Runtime plugins are plain JavaScript SDK consumers.
import realmsPlugin, { realmQueryOptions } from '../../../../plugins/hermes-realms/desktop/plugin.js'

import { SESSION_AREAS, type SessionContribution, type SessionContributionProps } from './session'

const owner = { connectionId: 'local', profile: 'default', storedSessionId: 'history', runtimeSessionId: 'runtime' }
const clients: QueryClient[] = []
const disposers: (() => void)[] = []

function mount(rest: ReturnType<typeof vi.fn>) {
  const renders = new Map<string, ComponentType<SessionContributionProps>>()

  const ctx = {
    rest,
    i18n: createPluginI18n('hermes-realms', dispose => {
      disposers.push(dispose)

      return dispose
    }),
    register: ({ area, data }: { area: string; data: SessionContribution }) => renders.set(area, data.render)
  }

  realmsPlugin.register(ctx)
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  clients.push(client)
  client.setQueryData(realmQueryOptions(ctx, owner).queryKey, {
    kind: 'realm',
    mode: 'realm',
    realms: [],
    setup: { ready: false, message: 'Run hermes realms install-driver in this profile.' }
  })
  const StatusRow = renders.get(SESSION_AREAS.statusStack)!

  const tree = (session = owner) => (
    <QueryClientProvider client={client}>
      <StatusRow session={session} />
    </QueryClientProvider>
  )

  return { ...render(tree()), tree }
}

afterEach(() => {
  cleanup()
  clients.splice(0).forEach(client => client.clear())
  disposers.splice(0).forEach(dispose => dispose())
})

it('repairs through explicit native confirmation, preserves failures for retry, and reads back readiness', async () => {
  let ready = false
  let fail = true
  const consent = 'pinned-owner-and-revision'

  const rest = vi.fn(async (path: string) => {
    if (path === '/realms/setup/prepare') {
      return {
        kind: 'realm',
        ready: false,
        action: 'repair',
        summary: 'Verify the existing driver.',
        details: [],
        consent
      }
    }

    if (path === '/realms/setup/start') {
      if (fail) {
        throw new Error('Permission denied. Retry setup.')
      }

      return { id: 'job', state: 'running', message: 'Verifying driver…' }
    }

    if (path.startsWith('/realms/setup/jobs/')) {
      ready = true

      return { id: 'job', state: 'succeeded', message: 'Ready' }
    }

    return { kind: 'realm', mode: 'realm', realms: [], setup: { ready } }
  })

  const view = mount(rest)

  expect(view.queryByRole('dialog')).toBeNull()
  fireEvent.click(view.getByRole('button', { name: 'Repair…' }))
  await view.findByRole('dialog')
  expect(rest.mock.calls.some(([path]) => path === '/realms/setup/start')).toBe(false)
  fireEvent.click(view.getByRole('button', { name: 'Cancel' }))
  expect(view.queryByRole('dialog')).toBeNull()
  expect(rest.mock.calls.some(([path]) => path === '/realms/setup/start')).toBe(false)
  fireEvent.click(view.getByRole('button', { name: 'Repair…' }))
  await view.findByRole('dialog')
  fireEvent.click(view.getByRole('button', { name: 'Repair' }))
  await view.findByText('Permission denied. Retry setup.')
  expect(view.getByRole('dialog')).toBeTruthy()
  fail = false
  fireEvent.click(view.getByRole('button', { name: 'Repair' }))
  await waitFor(() => expect(view.queryByRole('dialog')).toBeNull())
  await view.findByText('Ready')

  const call = rest.mock.calls.find(([path]) => path === '/realms/setup/start') as unknown as [
    string,
    { scope: typeof owner; body: object }
  ]

  expect(call[1].scope).toEqual(owner)
  expect(call[1].body).toEqual({ runtime_session_id: 'runtime', stored_session_id: 'history', kind: 'realm', consent })
  expect(view.container.textContent).not.toContain('install-driver')
})

it('makes the VM choice discoverable without switching before consent or opening a stale owner dialog', async () => {
  let resolveReview!: (value: object) => void

  const rest = vi.fn((path: string) =>
    path === '/realms/setup/prepare'
      ? new Promise(resolve => {
          resolveReview = resolve
        })
      : Promise.resolve({ kind: 'realm', mode: 'realm', realms: [], setup: { ready: false } })
  )

  const view = mount(rest)

  fireEvent.click(view.getByRole('button', { name: 'Omarchy VM' }))
  fireEvent.click(view.getByRole('button', { name: 'Set up Omarchy VM…' }))
  expect(rest.mock.calls.some(([path]) => path === '/realms/setup/start')).toBe(false)
  view.rerender(view.tree({ ...owner, profile: 'other', storedSessionId: 'other-history' }))
  await act(async () => {
    resolveReview({
      kind: 'omarchy-vm',
      ready: false,
      action: 'install',
      summary: 'Download and build the VM.',
      details: [],
      consent: 'old'
    })
  })
  expect(view.queryByRole('dialog')).toBeNull()
  expect(rest.mock.calls.some(([path]) => path === '/realms/setup/start')).toBe(false)
})

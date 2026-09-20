import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'
import type { MemoryProviderConfig, MemoryProviderField } from '@/types/hermes'

import { ProviderConfigPanel } from './provider-config'

// The real request layer runs so the PUT body and its owner pin are what the backend would receive.
vi.mock('@/hermes', async () => ({ ...(await import('@/api/client')), ...(await import('@/api/system')) }))

const api = vi.fn()
const owner = { connectionId: 'local', profile: 'alpha' }
const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
window.hermesDesktop = { api } as never

const field = (key: string, metadata: Partial<MemoryProviderField>) =>
  ({ key, label: key, kind: 'text', value: '', inline: true, is_set: false, ...metadata }) as MemoryProviderField

const schema = (fields: MemoryProviderField[], capabilities: MemoryProviderConfig['capabilities']) =>
  ({ name: 'b', label: 'Provider B', docs_url: '', capabilities, fields }) as MemoryProviderConfig

const hostSchema = schema(
  [field('token', { kind: 'secret', is_set: true }), field('workspace', { value: 'initial' })],
  { save_without_activation: true }
)

const nativeSchema = schema(
  [
    field('enabled', { kind: 'bool', value: 'true' }),
    field('conditional', { value: 'original', when: { enabled: true } }),
    field('workspace', { value: 'existing' }),
    field('token', { kind: 'secret', is_set: true })
  ],
  { save_without_activation: true, supports_partial_updates: false, requires_full_form: true }
)

function mountPanel(config: MemoryProviderConfig) {
  api.mockImplementation(async (request: HermesApiRequest) => (request.method === 'PUT' ? { ok: true } : config))
  render(
    <QueryClientProvider client={client}>
      <ProviderConfigPanel active={false} profile={owner} provider="b" />
    </QueryClientProvider>
  )
}

const writes = () => api.mock.calls.map(([request]) => request as HermesApiRequest).filter(r => r.method === 'PUT')
const saveButton = () => screen.getByRole('button', { name: 'Save changes' }) as HTMLButtonElement

afterEach(() => {
  cleanup()
  client.clear()
  api.mockReset()
})

it('host storage saves the changed keys with activate:false to the owner; a blank secret keeps the stored one', async () => {
  mountPanel(hostSchema)
  const token = (await screen.findByPlaceholderText('Leave blank to keep current value')) as HTMLInputElement
  expect(saveButton().disabled).toBe(true)
  fireEvent.change(screen.getByLabelText('workspace'), { target: { value: 'edited' } })
  fireEvent.click(saveButton())
  await screen.findByRole('status')
  expect(writes()).toMatchObject([
    { connectionId: 'local', profile: 'alpha', path: '/api/memory/providers/b/config?surface=declared' }
  ])
  expect(writes()[0].body).toEqual({ values: { workspace: 'edited' }, activate: false })
  expect(token.value).toBe('')
})

it('a native full-form writer submits every visible field; a field whose condition fails is hidden and omitted', async () => {
  mountPanel(nativeSchema)
  await screen.findByText(/Open Full configuration/)
  fireEvent.click(screen.getByRole('button', { name: 'Full configuration' }))
  const dialog = within(screen.getByRole('dialog'))
  expect(dialog.getByLabelText('conditional')).toBeTruthy()
  fireEvent.click(dialog.getByRole('switch', { name: 'enabled' }))
  expect(dialog.queryByLabelText('conditional')).toBeNull()
  fireEvent.click(dialog.getByRole('button', { name: 'Save changes' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  expect(writes().map(r => r.body)).toEqual([{ values: { enabled: 'false', workspace: 'existing' }, activate: false }])
})

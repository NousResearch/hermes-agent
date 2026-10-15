// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { FoundrySettings } from '@/api/azure-foundry'

const connectionState = atom<string | null>('local')
const profileState = atom('bot-a')

const calls = vi.hoisted(() => ({
  get: vi.fn(),
  discover: vi.fn(),
  remove: vi.fn(),
  confirm: vi.fn(),
  save: vi.fn(),
  test: vi.fn(),
  notify: vi.fn(),
  notifyError: vi.fn()
}))
vi.mock('@/hermes', () => ({
  deleteEnvVar: calls.remove,
  discoverFoundryDeployments: calls.discover,
  getFoundrySettings: calls.get,
  saveFoundrySettings: calls.save,
  validateFoundrySettings: calls.test
}))
vi.mock('@/store/confirm', () => ({ confirm: calls.confirm }))
vi.mock('@/store/notifications', () => ({ notify: calls.notify, notifyError: calls.notifyError }))
vi.mock('@/store/settings-scope', () => ({ $settingsRequestProfile: profileState }))
vi.mock('@/store/connections', () => ({ $activeConnectionId: connectionState }))

const { AzureFoundrySettings } = await import('./azure-foundry-settings')

const saved: FoundrySettings = {
  base_url: 'https://resource.services.ai.azure.com/openai/v1',
  model: 'deployment-alias',
  api_mode: 'codex_responses',
  auth: { kind: 'api_key' },
  api_key_set: true,
  is_current: true
}

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
  vi.resetAllMocks()
  profileState.set('bot-a')
  connectionState.set('local')
  calls.confirm.mockResolvedValue(true)
  calls.remove.mockResolvedValue({ ok: true })
  calls.get.mockResolvedValue(saved)
  calls.save.mockImplementation(async payload => ({ ...payload, is_current: true, api_key_set: false }))
  calls.discover.mockResolvedValue({ deployments: ['deployment-alias', 'another-model'] })
  calls.test.mockResolvedValue({ ok: true, reachable: true, message: '' })
})
afterEach(cleanup)

function surface() {
  return <AzureFoundrySettings expanded fallback={<p>Legacy API key settings</p>} onToggle={vi.fn()} />
}

it('offers CLI within Entra and sends account selectors to the same bot for test and save', async () => {
  render(surface())
  await screen.findByDisplayValue('deployment-alias')
  fireEvent.click(screen.getByRole('button', { name: 'Microsoft Entra ID' }))
  fireEvent.keyDown(screen.getByRole('combobox', { name: 'Get credentials from' }), { key: 'ArrowDown' })
  fireEvent.click(await screen.findByRole('option', { name: 'Azure CLI login' }))
  expect(screen.queryByLabelText('API Key')).toBeNull()
  fireEvent.click(screen.getByText('Advanced settings'))
  fireEvent.change(screen.getByLabelText('Tenant ID'), { target: { value: 'my-tenant' } })
  expect(screen.getByLabelText('Subscription name or ID')).toHaveProperty('disabled', true)
  expect(screen.getByLabelText('Token scope')).toBeTruthy()
  fireEvent.change(screen.getByLabelText('Tenant ID'), { target: { value: '' } })
  expect(screen.getByLabelText('Subscription name or ID')).toHaveProperty('disabled', false)
  fireEvent.change(screen.getByLabelText('Subscription name or ID'), { target: { value: 'my-subscription' } })
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Test' })))
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Save and use' })))

  const payload = {
    base_url: saved.base_url,
    model: saved.model,
    api_mode: 'codex_responses',
    auth: {
      kind: 'entra_id',
      entra: { credential_source: 'azure_cli', scope: '', subscription: 'my-subscription' }
    }
  }

  expect(calls.test).toHaveBeenCalledWith(payload, { connectionId: 'local', profile: 'bot-a' })
  expect(calls.save).toHaveBeenCalledWith(payload, { connectionId: 'local', profile: 'bot-a' })
})

it('keeps a saved API key private and preserves the draft after a failed save', async () => {
  calls.save.mockRejectedValue(new Error('Offline'))
  render(surface())
  await screen.findByPlaceholderText('Leave blank to keep current key')
  fireEvent.change(screen.getByLabelText('Deployment name'), { target: { value: 'edited-deployment' } })
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Save and use' })))
  expect(calls.save.mock.calls[0]?.[0].auth).toEqual({ kind: 'api_key', api_key: undefined })
  expect(screen.getByDisplayValue('edited-deployment')).toBeTruthy()
  expect(calls.notifyError).toHaveBeenCalled()
})

it('discards a pending response after moving to a different connection and profile', async () => {
  let finish: (data: FoundrySettings) => void = () => {
    throw new Error('Request not started')
  }

  calls.test.mockImplementation(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  render(surface())
  await screen.findByDisplayValue('deployment-alias')
  fireEvent.click(screen.getByRole('button', { name: 'Test' }))
  calls.get.mockResolvedValue({ ...saved, model: 'other-deployment' })
  await act(() => {
    profileState.set('bot-b')
    connectionState.set('remote')
  })
  await screen.findByDisplayValue('other-deployment')
  await act(() => finish(saved))
  expect(calls.notify).not.toHaveBeenCalled()
  expect(calls.get).toHaveBeenLastCalledWith({ connectionId: 'remote', profile: 'bot-b' })
})

it('uses the old API-key card only when the backend lacks the route', async () => {
  calls.get.mockRejectedValueOnce(new Error('404: No such API endpoint'))
  const view = render(surface())
  await screen.findByText('Legacy API key settings')
  view.unmount()
  calls.get.mockRejectedValueOnce(new Error('Connection timed out'))
  render(surface())
  await screen.findByText('Could not load Foundry settings.')
  expect(screen.queryByText('Legacy API key settings')).toBeNull()
  expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
})

it('removes the saved API key through the existing confirmation and scoped credential API', async () => {
  render(surface())
  await screen.findByPlaceholderText('Leave blank to keep current key')
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Refresh deployments' })))
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Remove' })))
  expect(calls.confirm).toHaveBeenCalledWith(expect.objectContaining({ destructive: true }))
  expect(calls.remove).toHaveBeenCalledWith('AZURE_FOUNDRY_API_KEY', { connectionId: 'local', profile: 'bot-a' })
  await waitFor(() => expect(screen.queryByPlaceholderText('Leave blank to keep current key')).toBeNull())
  fireEvent.click(screen.getByRole('button', { name: 'Show options' }))
  expect(screen.queryByText('another-model')).toBeNull()
  expect(calls.save).not.toHaveBeenCalled()
})

it('discovers with unsaved credentials without a deployment or a configuration write', async () => {
  calls.get.mockResolvedValue({
    ...saved,
    model: '',
    auth: { kind: 'entra_id', entra: { credential_source: 'azure_cli', scope: '' } }
  })
  render(surface())
  await screen.findByRole('combobox', { name: 'Get credentials from' })
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Refresh deployments' })))
  expect(calls.discover).toHaveBeenCalledWith(
    expect.objectContaining({
      model: '',
      auth: { kind: 'entra_id', entra: { credential_source: 'azure_cli', scope: '' } }
    }),
    { connectionId: 'local', profile: 'bot-a' }
  )
  fireEvent.click(screen.getByRole('button', { name: 'Show options' }))
  fireEvent.click(await screen.findByText('another-model'))
  expect(screen.getByDisplayValue('another-model')).toBeTruthy()
  expect(calls.save).not.toHaveBeenCalled()
  fireEvent.change(screen.getByLabelText('Deployment name'), { target: { value: 'manual-deployment' } })
  expect(screen.getByDisplayValue('manual-deployment')).toBeTruthy()
})

it('preserves automatic credentials and permits manual deployment entry when listing is unavailable', async () => {
  calls.get.mockResolvedValue({
    ...saved,
    auth: { kind: 'entra_id', entra: { credential_source: 'default', scope: '' } }
  })
  calls.discover.mockResolvedValue({ deployments: [] })
  render(surface())
  await screen.findByText('Automatic Azure credentials')
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Refresh deployments' })))
  expect(calls.notify).toHaveBeenCalledWith(
    expect.objectContaining({ kind: 'warning', message: expect.stringContaining('Enter a deployment name') })
  )
  fireEvent.change(screen.getByLabelText('Deployment name'), { target: { value: 'manual-deployment' } })
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Save and use' })))
  expect(calls.save).toHaveBeenCalledWith(
    expect.objectContaining({
      model: 'manual-deployment',
      auth: { kind: 'entra_id', entra: { credential_source: 'default', scope: '' } }
    }),
    expect.anything()
  )
})

it('discards discovery results after switching profiles', async () => {
  let finish: (data: { deployments: string[] }) => void = () => {
    throw new Error('Discovery not started')
  }
  calls.discover.mockImplementation(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  render(surface())
  await screen.findByDisplayValue('deployment-alias')
  fireEvent.click(screen.getByRole('button', { name: 'Refresh deployments' }))
  await act(() => profileState.set('bot-b'))
  await screen.findByDisplayValue('deployment-alias')
  await act(() => finish({ deployments: ['old-profile-model'] }))
  fireEvent.click(screen.getByRole('button', { name: 'Show options' }))
  expect(screen.queryByText('old-profile-model')).toBeNull()
  expect(calls.notify).not.toHaveBeenCalled()
})

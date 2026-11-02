import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import {
  activateOmlxModel,
  deleteOmlxModel,
  downloadOmlxModel,
  getOmlxStatus,
  installOmlx,
  type OmlxStatus,
  selectLocalEngine,
  setOmlxServer
} from '@/api/omlx'
import { enLocalModels } from '@/i18n/en_local_models'
import { notifyError } from '@/store/notifications'

import { deleteModel } from './local-models-actions'
import { LocalModelsEngine } from './local-models-engine'
import { SideloadedModelRow } from './local-models-model-rows'

const { owner } = vi.hoisted(() => ({ owner: { connectionId: 'local-A', profile: 'work' } }))
let client: QueryClient

vi.mock('@/api/omlx', () => ({
  getOmlxStatus: vi.fn(),
  selectLocalEngine: vi.fn(),
  installOmlx: vi.fn(),
  downloadOmlxModel: vi.fn(),
  activateOmlxModel: vi.fn(),
  deleteOmlxModel: vi.fn(),
  setOmlxServer: vi.fn()
}))
vi.mock('./local-models-actions', () => ({
  deleteModel: vi.fn(),
  useLocalModelsActionScope: () => ({ owner, client, copy: enLocalModels }),
  isActiveStatus: (status: string) => status === 'running' || status === 'paused'
}))
vi.mock('@/store/local-runtime-jobs', () => ({
  localModelsKey: (value: typeof owner, resource: string) => [
    'local-models',
    value.connectionId,
    value.profile,
    resource
  ],
  localModelsRequestScope: (value: typeof owner) => value,
  isCurrentLocalModelsOwner: () => true,
  refreshLocalModels: vi.fn(),
  useLocalRuntimeJobs: () => [],
  watchLocalRuntimeJobs: vi.fn()
}))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
vi.mock('./profile-scope', () => ({ ActiveProfileNote: () => null }))
vi.mock(import('./local-model-download-progress'), async importOriginal => ({
  ...(await importOriginal()),
  LocalModelDownloadActions: () => null,
  LocalModelDownloadProgress: () => <span />
}))

let status: OmlxStatus
beforeEach(() => {
  vi.clearAllMocks()
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  status = {
    supported: true,
    installed: true,
    running: false,
    selected: false,
    model_dir: '/models',
    base_url: 'http://127.0.0.1:8811/v1',
    version: 'test',
    models: [{ id: 'model-A', active: false }]
  }
  vi.mocked(getOmlxStatus).mockImplementation(async () => structuredClone(status))
  vi.mocked(selectLocalEngine).mockImplementation(async engine => {
    status.selected = engine === 'omlx'

    return { ok: true }
  })
})
afterEach(() => {
  vi.restoreAllMocks()
  cleanup()
  client.clear()
})

function mount(): void {
  render(
    <QueryClientProvider client={client}>
      <LocalModelsEngine>
        <p>Existing llama.cpp controls</p>
      </LocalModelsEngine>
    </QueryClientProvider>
  )
}

it('persists the engine choice in its owner scope and keeps llama.cpp available', async () => {
  mount()
  await waitFor(() => expect((screen.getByRole('button', { name: 'oMLX' }) as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(screen.getByRole('button', { name: 'oMLX' }))
  await screen.findByText('oMLX runtime installed')
  expect(selectLocalEngine).toHaveBeenCalledWith('omlx', owner)
  expect(screen.queryByText('Existing llama.cpp controls')).toBeNull()
  await waitFor(() =>
    expect((screen.getByRole('button', { name: 'llama.cpp' }) as HTMLButtonElement).disabled).toBe(false)
  )
  fireEvent.click(screen.getByRole('button', { name: 'llama.cpp' }))
  await screen.findByText('Existing llama.cpp controls')
  expect(selectLocalEngine).toHaveBeenCalledWith('llamacpp', owner)
})

it('wires managed install, start/stop, download and model activation to the same owner', async () => {
  status.selected = true
  status.installed = false
  vi.mocked(installOmlx).mockResolvedValue({ job_id: 'install' })
  vi.mocked(downloadOmlxModel).mockResolvedValue({ job_id: 'download' })
  vi.mocked(activateOmlxModel).mockResolvedValue({ ok: true })
  vi.mocked(setOmlxServer).mockResolvedValue({ ok: true })
  mount()
  fireEvent.click(await screen.findByRole('button', { name: 'Install runtime' }))
  await waitFor(() => expect(installOmlx).toHaveBeenCalledWith(owner))
  status.installed = true
  await client.invalidateQueries()
  fireEvent.click(await screen.findByRole('button', { name: 'Turn on' }))
  await waitFor(() => expect(setOmlxServer).toHaveBeenCalledWith('start', owner))
  await waitFor(() => expect((screen.getByRole('button', { name: 'Use' }) as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(screen.getByRole('button', { name: 'Use' }))
  await waitFor(() => expect(activateOmlxModel).toHaveBeenCalledWith('model-A', owner))
  await waitFor(() =>
    expect((screen.getByLabelText('Hugging Face model repository') as HTMLButtonElement).disabled).toBe(false)
  )
  fireEvent.change(screen.getByLabelText('Hugging Face model repository'), { target: { value: 'mlx-community/model' } })
  fireEvent.click(screen.getByRole('button', { name: 'Download model' }))
  await waitFor(() => expect(downloadOmlxModel).toHaveBeenCalledWith('mlx-community/model', owner))
})

it('does not switch engines when persistence fails', async () => {
  vi.mocked(selectLocalEngine).mockRejectedValue(new Error('write failed'))
  mount()
  await waitFor(() => expect((screen.getByRole('button', { name: 'oMLX' }) as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(screen.getByRole('button', { name: 'oMLX' }))
  await waitFor(() => expect(notifyError).toHaveBeenCalled())
  expect(screen.getByText('Existing llama.cpp controls')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'llama.cpp' }).getAttribute('aria-pressed')).toBe('true')
})

it('reuses model deletion confirmation and sends deletion only to the scoped oMLX API', async () => {
  status.selected = true
  vi.mocked(deleteOmlxModel).mockResolvedValue({ ok: true })
  const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
  mount()
  const remove = await screen.findByRole('button', { name: enLocalModels.deleteAction })
  fireEvent.click(remove)
  expect(confirm).toHaveBeenCalledWith(enLocalModels.deleteConfirm('model-A'))
  expect(deleteOmlxModel).not.toHaveBeenCalled()
  confirm.mockReturnValue(true)
  fireEvent.click(remove)
  await waitFor(() => expect(deleteOmlxModel).toHaveBeenCalledWith('model-A', owner))
  expect(deleteModel).not.toHaveBeenCalled()
})

it('keeps the shared row on llama.cpp actions when no engine callbacks are supplied', async () => {
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  render(
    <SideloadedModelRow jobs={[]} model={{ id: 'model.gguf', size_label: '1 GB' }} status={{ active_model_id: null }} />
  )
  fireEvent.click(screen.getByRole('button', { name: enLocalModels.deleteAction }))
  expect(deleteModel).toHaveBeenCalledWith({ owner, client, copy: enLocalModels }, 'model.gguf')
  expect(deleteOmlxModel).not.toHaveBeenCalled()
  await waitFor(() =>
    expect((screen.getByRole('button', { name: enLocalModels.deleteAction }) as HTMLButtonElement).disabled).toBe(false)
  )
})

it('hides the engine choice and ignores a stale oMLX preference on unsupported hosts', async () => {
  status.supported = false
  status.selected = true
  mount()
  await waitFor(() =>
    expect(client.getQueryData(['local-models', owner.connectionId, owner.profile, 'omlx'])).toMatchObject({
      supported: false,
      selected: true
    })
  )
  await screen.findByText('Existing llama.cpp controls')
  expect(screen.queryByRole('group', { name: 'Inference engine' })).toBeNull()
  expect(screen.queryByRole('button', { name: 'oMLX' })).toBeNull()
  expect(screen.queryByText('oMLX runtime installed')).toBeNull()
  expect(selectLocalEngine).not.toHaveBeenCalled()
})

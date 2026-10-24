vi.mock('@/store/profile', async (): Promise<object> => {
  const { atom } = await import('nanostores')

  return {
    $activeGatewayProfile: atom<string>('default'),
    $profiles: atom<Array<{ name: string; is_default?: boolean }>>([]),
    normalizeProfileKey: (profile: string | null): string => profile || 'default'
  }
})
vi.mock('@/store/session', async (): Promise<object> => {
  const { atom } = await import('nanostores')

  return { $connection: atom(null), $defaultReasoningEffort: atom<string>('') }
})

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import * as hermes from '@/hermes'
import { I18nProvider } from '@/i18n'
import { localModelsOwner } from '@/store/local-runtime-jobs'
import type { LocalEngine } from '@/types/hermes'

import { LocalModelsEngineSection } from './local-models-engine-section'
import { LocalModelsOwnerProvider } from './local-models-owner'

// The engine registry pane: picker + CRUD over llama.cpp builds. Transport is
// mocked; what matters is which payload reaches which route and what the pane
// paints from it (including the quiet opt-out on a pre-registry gateway).
vi.mock('@/hermes', () => ({
  addLocalEngine: vi.fn(),
  deleteLocalEngine: vi.fn(),
  getLocalEngines: vi.fn(),
  probeLocalEngine: vi.fn(),
  selectLocalEngine: vi.fn(),
  setApiRequestProfile: vi.fn(),
  updateLocalEngine: vi.fn()
}))

const mocked = vi.mocked(hermes)

const MANAGED: LocalEngine = {
  active: true,
  args: [],
  backend: 'auto',
  builtin: true,
  env: {},
  id: 'managed',
  installed: true,
  kind: 'managed',
  name: 'Hermes managed llama.cpp',
  path: ''
}

const CUSTOM: LocalEngine = {
  active: false,
  args: ['--n-gpu-layers', '99'],
  backend: 'vulkan',
  builtin: false,
  env: { GGML_VK_VISIBLE_DEVICES: '0' },
  id: 'my-build',
  installed: true,
  kind: 'external',
  name: 'My Vulkan build',
  path: 'D:/llama/build/bin/llama-server.exe'
}

function enginesPayload(...engines: LocalEngine[]) {
  return { engines }
}

function renderSection() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <I18nProvider>
        <LocalModelsOwnerProvider value={localModelsOwner('default', null)}>
          <LocalModelsEngineSection />
        </LocalModelsOwnerProvider>
      </I18nProvider>
    </QueryClientProvider>
  )
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('LocalModelsEngineSection', () => {
  it('lists the registry with the active engine in the picker', async () => {
    mocked.getLocalEngines.mockResolvedValue(enginesPayload(MANAGED, CUSTOM))
    renderSection()

    await screen.findByText('My Vulkan build')
    // Name appears twice: the picker shows the active engine, the list row below.
    expect(screen.getAllByText('Hermes managed llama.cpp').length).toBeGreaterThanOrEqual(2)
    // Pills: exactly one active (managed), one managed badge.
    expect(screen.getAllByText('active')).toHaveLength(1)
    expect(screen.getByText('managed')).toBeTruthy()
    // The managed engine is never editable or deletable; the custom one is.
    expect(screen.queryAllByLabelText('Edit engine')).toHaveLength(1)
    expect(screen.getByLabelText('Remove')).toBeTruthy()
    // The picker shows the active engine's name.
    expect(screen.getByRole('combobox').textContent).toContain('Hermes managed llama.cpp')
  })

  it('hides itself when the gateway has no engine registry (404)', async () => {
    mocked.getLocalEngines.mockRejectedValue(new Error('HTTP 404'))
    const { container } = renderSection()

    await waitFor(() => expect(mocked.getLocalEngines).toHaveBeenCalled())
    expect(container.textContent).toBe('')
    expect(screen.queryByRole('combobox')).toBeNull()
  })

  it('flags a custom engine whose binary is gone', async () => {
    mocked.getLocalEngines.mockResolvedValue(enginesPayload(MANAGED, { ...CUSTOM, installed: false }))
    renderSection()

    await screen.findByText('No runnable file at this path.')
  })

  it('registers a new engine from the form, parsing args and env', async () => {
    mocked.getLocalEngines.mockResolvedValue(enginesPayload(MANAGED))
    mocked.addLocalEngine.mockResolvedValue({ engine: CUSTOM, ok: true })
    renderSection()

    fireEvent.click(await screen.findByRole('button', { name: 'Register a binary' }))
    fireEvent.change(screen.getByPlaceholderText('llama.cpp (Vulkan test)'), { target: { value: 'My Vulkan build' } })
    fireEvent.change(screen.getByPlaceholderText('C:\\path\\to\\llama-server.exe'), {
      target: { value: ' D:/llama/build/bin/llama-server.exe ' }
    })
    fireEvent.change(screen.getByPlaceholderText('--n-gpu-layers 99'), {
      target: { value: '--n-gpu-layers\n99\n\n--flash-attn' }
    })
    fireEvent.change(screen.getByPlaceholderText('GGML_VK_VISIBLE_DEVICES=0'), {
      target: { value: 'GGML_VK_VISIBLE_DEVICES=0\nnot-an-identifier=skip' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(mocked.addLocalEngine).toHaveBeenCalled())
    const [fields] = mocked.addLocalEngine.mock.calls[0]

    expect(fields).toMatchObject({
      args: ['--n-gpu-layers', '99', '--flash-attn'],
      backend: 'auto',
      env: { GGML_VK_VISIBLE_DEVICES: '0' },
      name: 'My Vulkan build',
      path: 'D:/llama/build/bin/llama-server.exe'
    })
  })

  it('probes the path on screen before it is registered', async () => {
    mocked.getLocalEngines.mockResolvedValue(enginesPayload(MANAGED))
    mocked.probeLocalEngine.mockResolvedValue({ ok: true, version: 'version: b4096 (TEST)' })
    renderSection()

    fireEvent.click(await screen.findByRole('button', { name: 'Register a binary' }))
    fireEvent.change(screen.getByPlaceholderText('C:\\path\\to\\llama-server.exe'), {
      target: { value: 'D:/somewhere/llama-server.exe' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Test' }))

    await waitFor(() => expect(mocked.probeLocalEngine).toHaveBeenCalled())
    expect(mocked.probeLocalEngine.mock.calls[0][0]).toEqual({ path: 'D:/somewhere/llama-server.exe' })
    await screen.findByText('Engine answered: version: b4096 (TEST)')
  })

  it('removes a custom engine and edits it in place', async () => {
    mocked.getLocalEngines.mockResolvedValue(enginesPayload(MANAGED, CUSTOM))
    mocked.deleteLocalEngine.mockResolvedValue({ ok: true, was_active: false })
    mocked.updateLocalEngine.mockResolvedValue({ engine: CUSTOM, ok: true })
    renderSection()

    fireEvent.click(await screen.findByRole('button', { name: 'Remove' }))
    await waitFor(() => expect(mocked.deleteLocalEngine).toHaveBeenCalledWith('my-build', expect.anything()))

    fireEvent.click(screen.getByRole('button', { name: 'Edit engine' }))
    // The form opens prefilled from the registry record.
    expect((screen.getByPlaceholderText('C:\\path\\to\\llama-server.exe') as HTMLInputElement).value).toBe(CUSTOM.path)
    expect((screen.getByPlaceholderText('llama.cpp (Vulkan test)') as HTMLInputElement).value).toBe(CUSTOM.name)

    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(mocked.updateLocalEngine).toHaveBeenCalledWith('my-build', expect.objectContaining({
      args: ['--n-gpu-layers', '99'],
      env: { GGML_VK_VISIBLE_DEVICES: '0' }
    }), expect.anything()))
  })
})

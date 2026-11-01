import type { ModelOptionProvider } from '@hermes/shared'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import type * as Hermes from '@/hermes'
import type { LocalCatalogModel, LocalModelsStatus } from '@/types/hermes'

const backend = vi.hoisted(() => ({ catalog: vi.fn(), providers: vi.fn(), status: vi.fn() }))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof Hermes>()),
  getGlobalModelOptions: () => backend.providers(),
  getLocalCatalog: () => backend.catalog(),
  getLocalModelsStatus: () => backend.status()
}))

const row = (id: string, recommended: boolean): LocalCatalogModel =>
  ({ fits: true, id, recommended }) as unknown as LocalCatalogModel

const NO_RUNTIME = { models: [], runtime_installed: false } as unknown as LocalModelsStatus

const provider = (api_url: null | string): ModelOptionProvider => ({ api_url, name: 'Custom', slug: 'custom' })

const connection = (mode: 'local' | 'remote', baseUrl = 'http://127.0.0.1:9') =>
  ({ baseUrl, isFullscreen: false, mode }) as HermesConnection

async function connectLocal() {
  const { $localModelsEnabled } = await import('./local-models-flag')
  const { $connection } = await import('./session')
  const { refreshLocalSetupEligibility } = await import('./local-setup-offer')

  $localModelsEnabled.set(true)
  $connection.set(connection('local'))

  return { $connection, refreshLocalSetupEligibility }
}

async function eligibilityFor(models: LocalCatalogModel[]) {
  backend.status.mockResolvedValue(NO_RUNTIME)
  backend.catalog.mockResolvedValue({ models })
  const { refreshLocalSetupEligibility } = await connectLocal()

  return refreshLocalSetupEligibility()
}

describe('local-setup offer eligibility', () => {
  beforeEach(() => {
    vi.resetModules()
    localStorage.clear()
    backend.providers.mockReset()
    backend.providers.mockResolvedValue({ providers: [] })
  })

  it('offers the model the catalog recommends', async () => {
    const result = await eligibilityFor([row('qwen3.8-27b', false), row('qwen3.6-35b-a3b', true)])

    expect(result.fit?.model.id).toBe('qwen3.6-35b-a3b')
  })

  it('stays quiet on a machine with no recommendation', async () => {
    // A GTX 1080 with 16 GB of RAM: the 27B runs only with weights spilled to system memory, so the
    // catalog recommends nothing and the offer must not name it.
    const result = await eligibilityFor([row('qwen3.8-27b', false)])

    expect(result.fit).toBeNull()
  })

  it('stays quiet when a provider already serves this machine', async () => {
    // The user replaced the bundled runtime with their own server: the row must not ask them to
    // set up a model they already run.
    backend.providers.mockResolvedValue({ providers: [provider('http://127.0.0.1:8080/v1')] })

    const result = await eligibilityFor([row('qwen3.6-35b-a3b', true)])

    expect(result.fit).toBeNull()
    expect(result.reason).toBe('a provider already points at a local endpoint')
  })

  it('keeps the offer when every provider is on another machine', async () => {
    backend.providers.mockResolvedValue({
      providers: [provider('https://api.deepseek.com'), provider('https://api.openai.com/v1')]
    })

    const result = await eligibilityFor([row('qwen3.6-35b-a3b', true)])

    expect(result.fit?.model.id).toBe('qwen3.6-35b-a3b')
  })

  it('keeps the offer when the provider read fails', async () => {
    // Fail open: an unreadable provider list is not evidence of a local endpoint.
    backend.providers.mockRejectedValue(new Error('provider read failed'))

    const result = await eligibilityFor([row('qwen3.6-35b-a3b', true)])

    expect(result.fit?.model.id).toBe('qwen3.6-35b-a3b')
  })

  it('keeps the offer when the provider read answers nothing usable', async () => {
    backend.providers.mockResolvedValue({})

    const result = await eligibilityFor([row('qwen3.6-35b-a3b', true)])

    expect(result.fit?.model.id).toBe('qwen3.6-35b-a3b')
  })

  it('never offers setup on a remote connection, whatever its providers say', async () => {
    const { $connection } = await connectLocal()
    const { refreshLocalSetupEligibility } = await import('./local-setup-offer')

    $connection.set(connection('remote', 'https://homelab.example'))

    const result = await refreshLocalSetupEligibility()

    expect(result.fit).toBeNull()
    expect(result.reason).toBe('connection is remote, not local')
    expect(backend.providers).not.toHaveBeenCalled()
  })

  it("answers nothing from a read a connection change overtook, never the old machine's fit", async () => {
    let finish = (_status: LocalModelsStatus) => {}
    backend.status.mockReturnValue(new Promise<LocalModelsStatus>(resolve => (finish = resolve)))
    backend.catalog.mockResolvedValue({ models: [row('qwen3.6-35b-a3b', true)] })
    const { $connection, refreshLocalSetupEligibility } = await connectLocal()

    const read = refreshLocalSetupEligibility()
    $connection.set(connection('remote', 'https://homelab.example'))
    finish(NO_RUNTIME)

    expect((await read).fit).toBeNull()
  })
})

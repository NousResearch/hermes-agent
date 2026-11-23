import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import type * as Hermes from '@/hermes'
import type { LocalCatalogModel, LocalModelsStatus } from '@/types/hermes'

const backend = vi.hoisted(() => ({ catalog: vi.fn(), status: vi.fn() }))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof Hermes>()),
  getLocalCatalog: () => backend.catalog(),
  getLocalModelsStatus: () => backend.status()
}))

const row = (id: string, recommended: boolean): LocalCatalogModel =>
  ({ fits: true, id, recommended }) as unknown as LocalCatalogModel

const NO_RUNTIME = { models: [], runtime_installed: false } as unknown as LocalModelsStatus

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

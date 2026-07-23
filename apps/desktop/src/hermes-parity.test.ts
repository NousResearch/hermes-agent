import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getCuratorStatus,
  getMcpCatalog,
  getMemoryStatus,
  getSkillHubSources,
  getToolsetModels,
  installSkillFromHub,
  resetMemory,
  runDebugShare,
  searchSkillsHub,
  selectToolsetModel,
  setCuratorPaused,
  setMcpServerEnabled,
  testMcpServer
} from './hermes'

describe('Hermes REST parity helpers (hub / mcp / maintenance)', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn().mockResolvedValue({})
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('loads hub sources with a network-tolerant timeout', async () => {
    await getSkillHubSources()

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/skills/hub/sources', timeoutMs: 45_000 }))
  })

  it('encodes hub search params', async () => {
    await searchSkillsHub('gif search', 'official', 5)

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/api/skills/hub/search?q=gif+search&source=official&limit=5' })
    )
  })

  it('installs a hub skill by identifier', async () => {
    await installSkillFromHub('official/gifs/gif-search')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/skills/hub/install',
        method: 'POST',
        body: { identifier: 'official/gifs/gif-search' }
      })
    )
  })

  it('tests an MCP server with a boot-tolerant timeout and encoded name', async () => {
    await testMcpServer('file system')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/mcp/servers/file%20system/test',
        method: 'POST',
        timeoutMs: 60_000
      })
    )
  })

  it('toggles MCP server enablement', async () => {
    await setMcpServerEnabled('filesystem', false)

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/mcp/servers/filesystem/enabled',
        method: 'PUT',
        body: { enabled: false }
      })
    )
  })

  it('reads the MCP catalog', async () => {
    await getMcpCatalog()

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/mcp/catalog' }))
  })

  it('reads memory status and resets a specific target', async () => {
    await getMemoryStatus()
    await resetMemory('user')

    expect(api).toHaveBeenNthCalledWith(1, expect.objectContaining({ path: '/api/memory' }))
    expect(api).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ path: '/api/memory/reset', method: 'POST', body: { target: 'user' } })
    )
  })

  it('manages the curator', async () => {
    await getCuratorStatus()
    await setCuratorPaused(true)

    expect(api).toHaveBeenNthCalledWith(1, expect.objectContaining({ path: '/api/curator' }))
    expect(api).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ path: '/api/curator/paused', method: 'PUT', body: { paused: true } })
    )
  })

  it('runs debug share synchronously with an upload-tolerant timeout', async () => {
    await runDebugShare()

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/api/ops/debug-share', method: 'POST', timeoutMs: 120_000 })
    )
  })

  it('reads a backend model catalog scoped to a provider row', async () => {
    await getToolsetModels('image_gen', 'FAL.ai')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/api/tools/toolsets/image_gen/models?provider=FAL.ai' })
    )
  })

  it('persists a backend model selection', async () => {
    await selectToolsetModel('image_gen', 'z-image-turbo', 'FAL.ai')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/tools/toolsets/image_gen/model',
        method: 'PUT',
        body: { model: 'z-image-turbo', provider: 'FAL.ai' }
      })
    )
  })
})

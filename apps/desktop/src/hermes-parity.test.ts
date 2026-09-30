import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getToolsetModels, searchSkillsHub, testMcpServer } from './hermes'

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

  it('encodes hub search params', async () => {
    await searchSkillsHub('gif search', 'official', 5)

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/api/skills/hub/search?q=gif+search&source=official&limit=5' })
    )
  })

  it('tests an MCP server with an encoded name', async () => {
    await testMcpServer('file system')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/mcp/servers/file%20system/test',
        method: 'POST'
      })
    )
  })

  it('reads a backend model catalog scoped to a provider row', async () => {
    await getToolsetModels('image_gen', 'FAL.ai')

    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/api/tools/toolsets/image_gen/models?provider=FAL.ai' })
    )
  })
})

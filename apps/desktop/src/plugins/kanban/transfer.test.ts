import { afterEach, describe, expect, it, vi } from 'vitest'

import { runImportBoardFlow } from './transfer'

const doors = vi.hoisted(() => ({ notify: vi.fn() }))

const boardApi = vi.hoisted(() => ({
  exportBoard: vi.fn(),
  importBoard: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')
  const { QueryClient } = await import('@tanstack/react-query')

  return {
    atom,
    queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }),
    host: doors,
    useValue: vi.fn()
  }
})

vi.mock('./api', () => boardApi)

afterEach(() => {
  vi.clearAllMocks()
})

const k = { boardImported: (name: string) => `imported ${name}`, importBoardTitle: 'Import board' } as never

const imported = { board: 'ops', current: 'ops', name: 'Ops', renamed: false, warnings: [] }

function osWith(overrides: Record<string, unknown>) {
  return { pickOpenPath: vi.fn().mockResolvedValue(null), pickSavePath: vi.fn(), ...overrides } as never
}

describe('runImportBoardFlow', () => {
  it('imports the path the native dialog named, without asking the browser door', async () => {
    const pickBrowserFile = vi.fn()
    const os = osWith({ pickBrowserFile, pickOpenPath: vi.fn().mockResolvedValue('/Users/u/Downloads/ops.tar.gz') })
    boardApi.importBoard.mockResolvedValue(imported)

    await expect(runImportBoardFlow(os, k)).resolves.toBe('ops')

    expect(boardApi.importBoard).toHaveBeenCalledWith('/Users/u/Downloads/ops.tar.gz')
    expect(pickBrowserFile).not.toHaveBeenCalled()
  })

  it('imports the staged copy a browser pick produced', async () => {
    // A browser-hosted Desktop has no OS dialog, so pickOpenPath answers null.
    // Before the ladder, that left board import unreachable in the browser even
    // though the staged archive is a path this backend reads fine (#156470).
    const os = osWith({
      pickBrowserFile: vi.fn().mockResolvedValue({
        name: 'ops.tar.gz',
        path: '/home/u/.hermes/uploads/web-5c1f0a9b2d3e4f60-ops.tar.gz'
      })
    })

    boardApi.importBoard.mockResolvedValue(imported)

    await expect(runImportBoardFlow(os, k)).resolves.toBe('ops')

    expect(boardApi.importBoard).toHaveBeenCalledWith('/home/u/.hermes/uploads/web-5c1f0a9b2d3e4f60-ops.tar.gz')
  })

  it('reports a failed browser upload instead of doing nothing', async () => {
    // The picker rejects when the archive is over the browser's size cap or the
    // gateway refuses it. Board archives carry hundreds of megabytes of
    // attachments, so this is the common path in a browser, not an edge: it has
    // to say something rather than look like the user cancelled.
    const os = osWith({
      pickBrowserFile: vi.fn().mockRejectedValue(new Error('The file is larger than 16 MiB, the largest file the browser can attach.'))
    })

    await expect(runImportBoardFlow(os, k)).resolves.toBeNull()

    expect(boardApi.importBoard).not.toHaveBeenCalled()
    expect(doors.notify).toHaveBeenCalledWith({
      kind: 'error',
      message: 'The file is larger than 16 MiB, the largest file the browser can attach.'
    })
  })

  it('stays a no-op when both pickers are cancelled, and on a shell without the browser door', async () => {
    boardApi.importBoard.mockResolvedValue(imported)

    await expect(runImportBoardFlow(osWith({ pickBrowserFile: vi.fn().mockResolvedValue(null) }), k)).resolves.toBeNull()
    // An older shell has no pickBrowserFile at all: it must not throw.
    await expect(runImportBoardFlow(osWith({}), k)).resolves.toBeNull()

    expect(boardApi.importBoard).not.toHaveBeenCalled()
  })
})

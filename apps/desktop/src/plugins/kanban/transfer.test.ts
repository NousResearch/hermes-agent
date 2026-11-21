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
  it('imports the path the native dialog named', async () => {
    // The browser door answers null in the native app — there is no upload to
    // describe — so the OS dialog is what names the file. Asking the browser
    // door first costs one call and keeps the native case on the user's own
    // path rather than a staged copy.
    const pickBrowserFile = vi.fn().mockResolvedValue(null)
    const os = osWith({ pickBrowserFile, pickOpenPath: vi.fn().mockResolvedValue('/Users/u/Downloads/ops.tar.gz') })
    boardApi.importBoard.mockResolvedValue(imported)

    await expect(runImportBoardFlow(os, k)).resolves.toBe('ops')

    expect(boardApi.importBoard).toHaveBeenCalledWith('/Users/u/Downloads/ops.tar.gz')
    expect(pickBrowserFile).toHaveBeenCalled()
  })

  it('imports the staged copy a browser pick produced', async () => {
    // A browser-hosted Desktop has no OS dialog. The browser door answers the
    // copy the app staged, which is a path this backend reads fine, and it is
    // the door that says what the path is (#156470).
    const pickOpenPath = vi.fn()

    const os = osWith({
      pickBrowserFile: vi.fn().mockResolvedValue({
        name: 'ops.tar.gz',
        path: '/home/u/.hermes/uploads/web-5c1f0a9b2d3e4f60-ops.tar.gz'
      }),
      pickOpenPath
    })

    boardApi.importBoard.mockResolvedValue(imported)

    await expect(runImportBoardFlow(os, k)).resolves.toBe('ops')

    expect(boardApi.importBoard).toHaveBeenCalledWith('/home/u/.hermes/uploads/web-5c1f0a9b2d3e4f60-ops.tar.gz')
    expect(pickOpenPath).not.toHaveBeenCalled()
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

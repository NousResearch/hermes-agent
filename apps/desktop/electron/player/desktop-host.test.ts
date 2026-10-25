import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  disposeIpc: vi.fn(),
  disposeManager: vi.fn(),
  disposeMediaJobs: vi.fn(),
  registerOwner: vi.fn(),
  registerIpc: vi.fn(),
  constructManager: vi.fn(),
  constructMediaJobs: vi.fn()
}))

vi.mock('electron', () => ({
  app: {
    getPath: vi.fn(() => '/task/user-data'),
    getAppPath: vi.fn(() => '/task/app'),
    isPackaged: false
  },
  dialog: { showOpenDialog: vi.fn(), showSaveDialog: vi.fn() },
  ipcMain: {},
  shell: { showItemInFolder: vi.fn() }
}))

vi.mock('../player-ipc', () => ({ registerPlayerIpc: mocks.registerIpc }))
vi.mock('./manager', () => ({ PlayerManager: mocks.constructManager }))
vi.mock('../player-media-jobs', () => ({ PlayerMediaJobs: mocks.constructMediaJobs }))

import { createDesktopPlayerHost } from './desktop-host'

describe('desktop player host disposal', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.registerIpc.mockReturnValue({ registerOwner: mocks.registerOwner, dispose: mocks.disposeIpc })
    mocks.constructManager.mockImplementation(function constructManager() {
      return { dispose: mocks.disposeManager }
    })
    mocks.constructMediaJobs.mockImplementation(function constructMediaJobs() {
      return { dispose: mocks.disposeMediaJobs }
    })
  })

  it('disposes IPC, manager, and media jobs once across re-entered quit teardown', () => {
    const host = createDesktopPlayerHost()
    const owner = {} as Parameters<typeof host.registerOwner>[0]

    host.registerOwner(owner, 'file:///task/app/index.html')
    host.dispose()
    host.dispose()

    expect(mocks.registerOwner).toHaveBeenCalledOnce()
    expect(mocks.disposeIpc).toHaveBeenCalledOnce()
    expect(mocks.disposeManager).toHaveBeenCalledOnce()
    expect(mocks.disposeMediaJobs).toHaveBeenCalledOnce()
  })
})

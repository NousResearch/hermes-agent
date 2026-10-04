import { expect, it, vi } from 'vitest'

import { registerGitIpc } from './git-ipc'

interface PrListBridge {
  git: { review: { prList: (...args: unknown[]) => Promise<{ prs: { url: string }[] }> } }
}

const bridge = vi.hoisted(() => ({
  exposed: null as PrListBridge | null,
  handlers: new Map<string, (...args: unknown[]) => unknown>(),
  execFile: vi.fn()
}))

vi.mock('node:child_process', () => ({ execFile: bridge.execFile }))
vi.mock('electron', () => ({
  contextBridge: {
    exposeInMainWorld: (_name: string, api: PrListBridge) => {
      bridge.exposed = api
    }
  },
  ipcMain: { handle: (name: string, handler: (...args: unknown[]) => unknown) => bridge.handlers.set(name, handler) },
  ipcRenderer: {
    sendSync: () => ({}),
    invoke: (name: string, ...args: unknown[]) => bridge.handlers.get(name)!({}, ...args)
  },
  webFrame: {},
  webUtils: {}
}))

it('carries URL-only PR hydration through the actual preload and IPC handler', async () => {
  const url = 'https://github.com/other/repository/pull/42'

  bridge.execFile.mockImplementation((_bin, args, _options, callback) => {
    callback(null, JSON.stringify({ headRefName: 'feature', number: 42, url: args[2], state: 'OPEN' }))
  })
  registerGitIpc({ resolveGitBinary: () => 'git', resolveGhBinary: () => 'gh' })
  await import('./preload')

  const result = await bridge.exposed!.git!.review.prList('', [], [], [url])

  expect(result.prs.map(pr => pr.url)).toEqual([url])
  expect(bridge.execFile.mock.calls[0][1].slice(0, 3)).toEqual(['pr', 'view', url])
})

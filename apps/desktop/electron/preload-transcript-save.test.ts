import { expect, it, vi } from 'vitest'

const host = vi.hoisted(() => ({
  contextBridge: { exposeInMainWorld: vi.fn() },
  ipcRenderer: {
    invoke: vi.fn(async () => ({ saved: true, path: 'C:/chosen.docx' })),
    on: vi.fn(),
    removeListener: vi.fn(),
    send: vi.fn(),
    sendSync: vi.fn(() => ({}))
  },
  webFrame: {},
  webUtils: {}
}))

vi.mock('electron', () => host)

it('advertises binary-save support on the real preload bridge and forwards the explicit encoding', async () => {
  await import('./preload')
  const [name, bridge] = host.contextBridge.exposeInMainWorld.mock.calls[0]
  expect(name).toBe('hermesDesktop')
  expect(bridge.saveTextFileEncodings).toEqual(['base64'])
  const payload = { content: 'AAE=', encoding: 'base64', defaultPath: 'sample.docx' }
  await bridge.saveTextFileAs(payload)
  expect(host.ipcRenderer.invoke).toHaveBeenLastCalledWith('hermes:saveTextFileAs', payload)
})

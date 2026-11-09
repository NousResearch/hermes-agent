import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterAll, expect, test, vi } from 'vitest'

const handlers = new Map<string, (...args: unknown[]) => unknown>()
const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-metrics-'))
const sent: string[] = []

vi.mock('electron', () => ({
  app: { getPath: () => userData, getVersion: () => '1.0.0' },
  BrowserWindow: { getAllWindows: () => [{ webContents: { send: (channel: string) => sent.push(channel) } }] },
  ipcMain: {
    handle: (channel: string, fn: (...args: unknown[]) => unknown) => {
      handlers.set(channel, fn)
    }
  }
}))

vi.mock('./install-stamp', () => ({ INSTALL_STAMP: { commitDate: 1_700_000_000 } }))

const { registerDesktopSharedMetrics } = await import('./desktop-shared-metrics')

afterAll(() => fs.rmSync(userData, { recursive: true, force: true }))

const strategy = (mechanism: string) => ({ mechanism, apply: async () => ({ ok: true, handedOff: false }) }) as never

test('a packaged apply is reported once through the renderer IPC; a checkout hand-off never is', async () => {
  const metrics = registerDesktopSharedMetrics()

  expect(handlers.has('hermes:startup-latency:claim')).toBe(true)
  expect(handlers.has('hermes:desktop-metrics:crash:take')).toBe(true)

  await metrics.trackUpdateApply(null, strategy('checkout'))
  expect(handlers.get('hermes:updates:metric:take')!({})).toBeNull()

  await metrics.trackUpdateApply(strategy('electron-updater'), strategy('electron-updater'))
  expect(sent).toContain('hermes:updates:metric:pending')
  expect(handlers.get('hermes:updates:metric:take')!({})).toMatchObject({
    mechanism: 'electron-updater',
    outcome: 'noop',
    from_commit_date: 1_700_000_000
  })
})

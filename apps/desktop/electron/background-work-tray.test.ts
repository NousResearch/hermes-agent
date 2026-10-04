import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, test, vi } from 'vitest'

const { trayInstances, menuBuild } = vi.hoisted(() => ({
  trayInstances: [] as Array<Record<string, ReturnType<typeof vi.fn>>>,
  menuBuild: vi.fn(template => template)
}))

vi.mock('electron', () => ({
  Menu: { buildFromTemplate: menuBuild },
  nativeImage: {
    createFromDataURL: vi.fn(() => ({ setTemplateImage: vi.fn() }))
  },
  Tray: class {
    setImage = vi.fn()
    setTitle = vi.fn()
    setToolTip = vi.fn()
    setContextMenu = vi.fn()
    destroy = vi.fn()

    constructor() {
      trayInstances.push(this as unknown as Record<string, ReturnType<typeof vi.fn>>)
    }
  }
}))

import { createBackgroundWorkTray } from './background-work-tray'

beforeEach(() => {
  vi.useFakeTimers()
  trayInstances.length = 0
  menuBuild.mockClear()
})

afterEach(() => vi.useRealTimers())

test('macOS status item changes from idle to busy and is destroyed cleanly', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-work-tray-'))
  const controller = createBackgroundWorkTray(root, { platform: 'darwin', pollMilliseconds: 500 })
  controller.start()

  assert.equal(trayInstances.length, 1)
  assert.equal(trayInstances[0].setToolTip.mock.calls.at(-1)?.[0], 'Hermes · Idle')

  const directory = path.join(root, 'cache', 'background-work')
  fs.mkdirSync(directory, { recursive: true })
  fs.writeFileSync(path.join(directory, 'live.json'), JSON.stringify({
    key: 'live', title: 'Build indicator', worker: 'kanban', profile: 'coder', model: 'gpt',
    pid: process.pid, started_at: Date.now() / 1000, state: 'running'
  }))
  vi.advanceTimersByTime(500)

  assert.equal(trayInstances[0].setTitle.mock.calls.at(-1)?.[0], '1')
  assert.match(String(trayInstances[0].setToolTip.mock.calls.at(-1)?.[0]), /1 background worker/)

  controller.stop()
  assert.equal(trayInstances[0].destroy.mock.calls.length, 1)
  fs.rmSync(root, { recursive: true, force: true })
})

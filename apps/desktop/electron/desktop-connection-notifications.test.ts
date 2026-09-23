import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createDesktopConnectionNotifications } from './desktop-connection-notifications'

test('connection notifications target the current main window and every live registry window', () => {
  const sent: unknown[] = []
  const live = { isDestroyed: () => false, webContents: { isDestroyed: () => false, send: (...args: unknown[]) => sent.push(args) } }
  const dead = { isDestroyed: () => false, webContents: { isDestroyed: () => true, send: () => { throw new Error('dead') } } }
  let mainWindow: any = null

  const notifications = createDesktopConnectionNotifications({
    BrowserWindow: { getAllWindows: () => [live, dead] },
    getMainWindow: () => mainWindow
  })

  notifications.sendConnectionApplied()
  mainWindow = live
  notifications.sendConnectionApplied()
  notifications.broadcastConnectionsChanged({ connectionId: 'remote', reason: 'updated' })

  assert.deepEqual(sent, [
    ['hermes:connection:applied'],
    ['hermes:connections:changed', { connectionId: 'remote', reason: 'updated' }]
  ])
})

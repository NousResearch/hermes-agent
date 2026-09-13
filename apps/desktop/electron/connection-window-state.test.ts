import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const here = path.dirname(fileURLToPath(import.meta.url))
const mainSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

function ipcHandlerSource(channel: string, nextRegistration: string): string {
  const start = mainSource.indexOf(`ipcMain.handle('${channel}', `)
  const end = mainSource.indexOf(nextRegistration, start)

  expect(start).toBeGreaterThan(-1)
  expect(end).toBeGreaterThan(start)

  return mainSource.slice(start, end)
}

describe('connection IPC window state', () => {
  it('overlays cached primary connection chrome with the sender window live state', () => {
    const handler = ipcHandlerSource('hermes:connection', "ipcMain.handle('hermes:connection:for', ")

    expect(handler).toContain(
      'const windowState = getWindowState(BrowserWindow.fromWebContents(event.sender) || mainWindow)'
    )
    expect(handler).toContain('{ ...connection, ...windowState, connectionId }')
    expect(handler).toContain('{ ...connection, ...windowState }')
  })

  it('overlays cached registry connection chrome with the sender window live state', () => {
    const handler = ipcHandlerSource('hermes:connection:for', 'const windowConnectionRoutes')

    expect(handler).toContain(
      'const windowState = getWindowState(BrowserWindow.fromWebContents(event.sender) || mainWindow)'
    )
    expect(handler).toContain('{ ...connection, ...windowState, connectionId: id, registryScoped: true }')
  })
})

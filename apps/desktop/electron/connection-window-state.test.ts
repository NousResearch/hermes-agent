import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const here = path.dirname(fileURLToPath(import.meta.url))
const mainSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

function sliceSource(start: string, end: string): string {
  const from = mainSource.indexOf(start)
  const to = mainSource.indexOf(end, from)

  expect(from).toBeGreaterThan(-1)
  expect(to).toBeGreaterThan(from)

  return mainSource.slice(from, to)
}

describe('connection IPC window state', () => {
  it('overlays the caller window live state onto every connection shape', () => {
    const route = sliceSource(
      'async function connectDesktopProfileRoute(',
      "ipcMain.handle('hermes:connection:for', "
    )

    // Read at reply time, not taken from whatever the renderer cached at its first dial.
    expect(route).toContain(
      'const windowState = getWindowState((sender && BrowserWindow.fromWebContents(sender)) || mainWindow)'
    )
    // Registry-scoped, primary-resolved and bare replies all carry it.
    expect(route).toContain(
      '{ ...connection, ...windowState, connectionId: route.connectionId, registryScoped: true }'
    )
    expect(route).toContain('{ ...connection, ...windowState, connectionId }')
    expect(route).toContain('{ ...connection, ...windowState }')
  })

  it('threads the calling window through from both IPC handlers', () => {
    const primary = sliceSource(
      "ipcMain.handle('hermes:connection', ",
      'async function connectDesktopProfileRoute('
    )
    const scoped = sliceSource("ipcMain.handle('hermes:connection:for', ", 'const windowConnectionRoutes')

    expect(primary).toContain(
      'connectDesktopProfileRoute(route, spawnPriorityFrom(extra?.priority), event.sender)'
    )
    // Dropping the sender is silent: the reply falls back to mainWindow, so a secondary
    // window is told the main window's chrome. That is the bug this guards.
    expect(scoped).toContain('event.sender')
    expect(scoped).not.toContain('_event')
  })
})

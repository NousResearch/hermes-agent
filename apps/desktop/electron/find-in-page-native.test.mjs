import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { app, BrowserWindow } from 'electron'

const runtimeDir = mkdtempSync(join(tmpdir(), 'hermes-find-in-page-'))
app.setPath('userData', runtimeDir)
app.setPath('sessionData', runtimeDir)

async function findCount(window, query, afterFirstResult) {
  return await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`findInPage timed out for ${query}`)), 5000)
    let requestId
    let firstResult = true

    const onResult = (_event, result) => {
      if (result.requestId !== requestId) return
      if (firstResult) {
        firstResult = false
        afterFirstResult?.()
      }
      if (!result.finalUpdate) return
      clearTimeout(timeout)
      window.webContents.off('found-in-page', onResult)
      resolve(result.matches)
    }

    window.webContents.on('found-in-page', onResult)
    requestId = window.webContents.findInPage(query)
  })
}

async function run() {
  const window = new BrowserWindow({
    show: true,
    x: 0,
    y: 0,
    width: 320,
    height: 240,
    skipTaskbar: true,
    opacity: 0,
    webPreferences: { backgroundThrottling: false }
  })

  try {
    const html = ['<input id="query" type="search" aria-label="Find in page" value="needle">', '<p>needle</p>'].join('')
    const fixturePath = join(runtimeDir, 'fixture.html')
    writeFileSync(fixturePath, html)
    await window.loadFile(fixturePath)
    await window.webContents.executeJavaScript('document.body.innerText')

    assert.equal(await findCount(window, 'needle'), 2, 'control: search input is indexed')
    window.webContents.stopFindInPage('clearSelection')

    await window.webContents.executeJavaScript('query.focus(); query.setSelectionRange(6, 6); query.inert = true')
    const count = await findCount(window, 'needle', () => {
      void window.webContents.executeJavaScript('query.inert = false; query.focus(); query.setSelectionRange(6, 6)')
    })

    assert.equal(count, 1, 'transient inert excludes the visible query from Chromium indexing')

    const state = await window.webContents.executeJavaScript(`JSON.stringify({
      type: query.type,
      explicitRole: query.getAttribute('role'),
      inert: query.inert,
      focused: document.activeElement === query,
      selectionStart: query.selectionStart,
      value: query.value
    })`)
    assert.deepEqual(JSON.parse(state), {
      type: 'search',
      explicitRole: null,
      inert: false,
      focused: true,
      selectionStart: 6,
      value: 'needle'
    })

    window.webContents.debugger.attach('1.3')
    try {
      await window.webContents.debugger.sendCommand('Accessibility.enable')
      const { nodes } = await window.webContents.debugger.sendCommand('Accessibility.getFullAXTree')
      assert.ok(
        nodes.some(node => node.role?.value === 'searchbox' && node.name?.value === 'Find in page'),
        'Chromium accessibility tree exposes a truthful searchbox'
      )
    } finally {
      window.webContents.debugger.detach()
    }
  } finally {
    window.destroy()
  }
}

app
  .whenReady()
  .then(run)
  .then(() => app.exit(0))
  .catch(error => {
    console.error(error)
    app.exit(1)
  })

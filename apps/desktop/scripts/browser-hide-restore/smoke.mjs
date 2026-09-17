import assert from 'node:assert/strict'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { createServer as httpServer } from 'node:http'
import { createServer } from 'vite'
import { _electron as electron } from '@playwright/test'
import electronPath from 'electron'

const artifacts = resolve('../../.hide-restore-artifacts')
await mkdir(artifacts, { recursive: true })
// Drop any prior success receipt so a crashed run cannot look green.
await writeFile(resolve(artifacts, 'smoke-result.json'), JSON.stringify({ success: false, pending: true }))

const temp = await mkdtemp(resolve(artifacts, 'electron-'))
let loads = 0
let pageServer
let vite
let app
const logs = []

const cleanup = async () => {
  const errors = []
  for (const step of [
    async () => {
      await writeFile(resolve(artifacts, 'renderer.log'), logs.join('\n'))
    },
    async () => {
      await app?.close()
    },
    async () => {
      await vite?.close()
    },
    async () => {
      if (pageServer?.listening) {
        await new Promise(r => pageServer.close(r))
      }
    },
    async () => {
      await rm(temp, { recursive: true, force: true })
    }
  ]) {
    try {
      await step()
    } catch (error) {
      errors.push(error)
    }
  }
  if (errors.length) {
    throw errors[0]
  }
}

try {
  pageServer = httpServer((req, res) => {
    if (req.url !== '/page') {
      res.writeHead(204).end()
      return
    }
    loads++
    res.setHeader('content-type', 'text/html')
    res.end(
      `<html><body><h1>Local deterministic page</h1><input id="draft"><input id="entry"><button id="count" onclick="window.clicks++;document.querySelector('#result').textContent=window.clicks">Increment</button><p id="result">0</p><script>window.clicks=0;window.ticks=0;window.identity=crypto.randomUUID();setInterval(()=>window.ticks++,50)</script></body></html>`
    )
  })
  await new Promise(r => pageServer.listen(0, '127.0.0.1', r))
  vite = await createServer({ server: { host: '127.0.0.1', port: 0 } })
  await vite.listen()
  const fixtureUrl = `${vite.resolvedUrls.local[0]}scripts/browser-hide-restore/fixture.html?page=${encodeURIComponent(`http://127.0.0.1:${pageServer.address().port}/page`)}`
  const env = { ...process.env, HIDE_RESTORE_USER_DATA: temp, HIDE_RESTORE_FIXTURE_URL: fixtureUrl }
  delete env.ELECTRON_RUN_AS_NODE
  app = await electron.launch({ executablePath: electronPath, args: ['scripts/browser-hide-restore/main.cjs'], env })
  const page = await app.firstWindow()
  page.on('console', msg => logs.push(`${msg.type()}: ${msg.text()}`))
  page.on('pageerror', err => logs.push(`PAGEERROR: ${err.stack}`))
  await page.waitForFunction(
    () => {
      try {
        return document.querySelector('webview')?.getWebContentsId()
      } catch {
        return false
      }
    },
    null,
    { timeout: 60000 }
  )
  await page.waitForFunction(async () => {
    try {
      return await window.fixture.run('Boolean(window.identity)')
    } catch {
      return false
    }
  })
  const guestId = await page.evaluate(() => document.querySelector('webview').getWebContentsId())
  const before = await page.evaluate(() =>
    window.fixture.run(
      `document.querySelector('#draft').value='unsaved'; ({identity:window.identity,ticks:window.ticks,width:innerWidth,height:innerHeight})`
    )
  )
  await page.screenshot({ path: resolve(artifacts, 'visible.png') })
  await page.locator('[data-tree-group="browser"] [data-tree-tab]').first().click({ button: 'right' })
  await page.getByRole('menuitem', { name: 'Hide', exact: true }).click({ timeout: 3000 })
  await page.waitForFunction(() => document.querySelector('webview')?.closest('[data-pane-hidden]'))
  assert.equal(await page.evaluate(() => window.fixture.visibleGuest()), false)
  assert.equal(await page.evaluate(() => document.querySelector('webview').getWebContentsId()), guestId)
  const hidden = await page.evaluate(() =>
    window.fixture.run('({identity:window.identity,ticks:window.ticks,width:innerWidth,height:innerHeight})')
  )
  assert.equal(hidden.width, before.width, 'hidden guest keeps its viewport width')
  assert.equal(hidden.height, before.height, 'hidden guest keeps its viewport height')
  await page.waitForFunction(async ticks => (await window.fixture.run('window.ticks')) > ticks, hidden.ticks)
  await page.getByRole('textbox', { name: 'Composer' }).focus()
  console.log('focus before', await page.evaluate(() => document.activeElement?.outerHTML))
  const action = await page.evaluate(() => window.fixture.drive({ kind: 'click', selector: '#count' }))
  assert.equal(action.success, true, JSON.stringify(action))
  assert.equal(await page.evaluate(() => window.fixture.run('window.clicks')), 1)
  assert.equal(await page.evaluate(() => window.fixture.visibleGuest()), false, 'automation must not reveal')
  const typed = await page.evaluate(() =>
    window.fixture.drive({ kind: 'type', selector: '#entry', text: 'hidden automation' })
  )
  assert.equal(typed.success, true, JSON.stringify(typed))
  assert.equal(
    await page.evaluate(() => window.fixture.run('document.querySelector("#entry").value')),
    'hidden automation'
  )
  assert.equal(await page.evaluate(() => window.fixture.run('document.querySelector("#draft").value')), 'unsaved')
  assert.ok(await page.evaluate(() => window.fixture.read()))
  console.log('focus after', await page.evaluate(() => document.activeElement?.outerHTML))
  assert.equal(
    await page.evaluate(() => document.activeElement?.getAttribute('aria-label')),
    'Composer',
    'hidden automation must not take host focus'
  )
  assert.equal(await page.evaluate(() => window.fixture.visibleGuest()), false)
  await page.screenshot({ path: resolve(artifacts, 'hidden.png') })
  await page.locator('[data-tree-group="browser"] [data-tree-tab]').first().click()
  await page.waitForFunction(() => window.fixture.visibleGuest())
  assert.equal(await page.evaluate(() => document.querySelector('webview').getWebContentsId()), guestId)
  assert.equal(await page.evaluate(() => window.fixture.run('window.identity')), before.identity)
  assert.equal(loads, 1, 'hide/restore does not navigate')
  for (const id of ['file-a', 'file-b', 'file-c']) {
    await page.evaluate(id => window.fixture.activate(id), id)
    await page.waitForTimeout(50)
  }
  assert.equal(
    await page.evaluate(() => document.querySelector('webview').getWebContentsId()),
    guestId,
    'LRU cannot park browser'
  )
  await page.evaluate(() => window.fixture.activate(window.fixture.paneId))
  await page.waitForFunction(() => window.fixture.visibleGuest())
  await page.evaluate(() => window.fixture.hide())
  await page.locator('[data-tree-group="browser"] [data-tree-tab]').first().click({ button: 'right' })
  await page.getByRole('menuitem', { name: 'Close', exact: true }).click()
  await page.waitForFunction(() => !document.querySelector('webview') && !window.fixture.runnerPresent())
  assert.equal(
    await app.evaluate(({ webContents }, id) => Boolean(webContents.fromId(id)), guestId),
    false,
    'Close destroys webContents'
  )
  await writeFile(
    resolve(artifacts, 'smoke-result.json'),
    JSON.stringify({ success: true, guestId, before, hidden, loads, action, typed }, null, 2)
  )
  console.log('PASS: live hidden guest, viewport, timers, drive click/type, read, restore, LRU, close')
} finally {
  await cleanup()
}

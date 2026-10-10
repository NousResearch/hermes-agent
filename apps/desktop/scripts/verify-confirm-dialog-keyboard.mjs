import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'
import { chromium } from '@playwright/test'

// Standalone browser smoke for native button activation (jsdom cannot emulate it).
// Requires an installed Playwright Chromium, or PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH.
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')
const desktop = path.join(repo, 'apps/desktop')
const fixture = path.join(desktop, 'confirm-dialog-browser-fixture.tsx')
const html = path.join(desktop, 'confirm-dialog-browser-fixture.html')
fs.writeFileSync(html, '<div id="root"></div><script type="module" src="/confirm-dialog-browser-fixture.tsx"></script>')
fs.writeFileSync(
  fixture,
  `import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { ConfirmDialog } from './src/components/ui/confirm-dialog'
function Fixture() {
  const [open, setOpen] = useState(true)
  return <><button id="open" onClick={() => setOpen(true)}>Open</button><ConfirmDialog
    open={open} onClose={() => { window.calls.close++; setOpen(false) }}
    onConfirm={() => { window.calls.confirm++; return new Promise(resolve => { window.resolveConfirm = resolve }) }}
    secondaryAction={{label: 'Secondary', onClick: () => {window.calls.secondary++}}}
    title="Destructive dialog" destructive><input aria-label="Reason" /></ConfirmDialog></>
}
window.calls = { close: 0, confirm: 0, secondary: 0 }
createRoot(document.getElementById('root')).render(<Fixture />)
`
)
let server, browser
try {
  server = await createServer({
    root: desktop,
    configFile: path.join(desktop, 'vite.config.ts'),
    server: { host: '127.0.0.1', port: 0 }
  })
  await server.listen()
  const address = server.httpServer.address()
  browser = await chromium.launch({
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || undefined,
    headless: true
  })
  const page = await browser.newPage()
  const pageErrors = []
  page.on('pageerror', error => pageErrors.push(error.message))
  await page.goto(`http://127.0.0.1:${address.port}/confirm-dialog-browser-fixture.html`)
  const button = name => page.getByRole('button', { name, exact: true })
  const calls = () => page.evaluate(() => window.calls)
  await button('Cancel').waitFor()
  if (!(await button('Cancel').evaluate(node => node === document.activeElement)))
    throw Error('Destructive dialog did not focus Cancel')
  await page.keyboard.press('Enter')
  if ((await calls()).close !== 1 || (await calls()).confirm !== 0) throw Error('Enter on Cancel failed')
  await page.locator('#open').click()
  await button('Cancel').focus()
  await page.keyboard.press('Space')
  if ((await calls()).close !== 2 || (await calls()).confirm !== 0) throw Error('Space on Cancel failed')
  await page.locator('#open').click()
  await button('Secondary').focus()
  await page.keyboard.press('Enter')
  if ((await calls()).secondary !== 1 || (await calls()).confirm !== 0) throw Error('Enter on secondary failed')
  await page.locator('#open').click()
  await button('Secondary').focus()
  await page.keyboard.press('Space')
  if ((await calls()).secondary !== 2 || (await calls()).confirm !== 0) throw Error('Space on secondary failed')
  await page.locator('#open').click()
  await page.getByRole('textbox', { name: 'Reason' }).focus()
  await page.keyboard.press('Space')
  await page.keyboard.press('Enter')
  if ((await calls()).confirm !== 0) throw Error('Typing in input confirmed')
  await button('Confirm').focus()
  await page.keyboard.press('Enter')
  if ((await calls()).confirm !== 1) throw Error('Enter on Confirm failed')
  await page.keyboard.press('Enter')
  await page.keyboard.press('Space')
  if ((await calls()).confirm !== 1) throw Error('Busy repeat was not suppressed')
  await page.evaluate(() => window.resolveConfirm())
  await page.getByRole('dialog').waitFor({ state: 'hidden' })
  await page.locator('#open').click()
  await button('Confirm').focus()
  await page.keyboard.press('Space')
  if ((await calls()).confirm !== 2) throw Error('Space on Confirm failed')
  if (pageErrors.length) throw Error(pageErrors.join('\n'))
  console.log('PASS: native Enter/Space on Cancel, secondary, Confirm; input keys; pending repeat; destructive focus')
} finally {
  await browser?.close()
  await server?.close()
  fs.rmSync(fixture, { force: true })
  fs.rmSync(html, { force: true })
}

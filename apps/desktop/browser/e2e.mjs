#!/usr/bin/env node
/* global window, document, innerWidth */
// Real browser + Hermes backend + repository mock inference; no paid provider calls.
import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

import { workspaceTool } from '../../../scripts/build/frontend-common.mjs'
import { MOCK_REPLY, startMockServer } from '../../../tests-js/scripts/mock-server.ts'
import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config.ts'

const root = fileURLToPath(new URL('../../../', import.meta.url))
const python = process.env.HERMES_PYTHON
if (!python) throw new Error('Run through scripts/run-in-hermes-env so HERMES_PYTHON names the prepared runtime.')
const playwrightModule = await import(pathToFileURL(workspaceTool(root, 'apps/desktop', '@playwright/test')).href)
const { chromium } = playwrightModule.default || playwrightModule
const sandbox = await mkdtemp(path.join(tmpdir(), 'hermes-browser-e2e-'))
const home = path.join(sandbox, 'home')
await mkdir(home)
const mock = await startMockServer()
writeMockProviderConfig(home, mock.url, undefined, JSON.stringify({
  dashboard: { public_url: 'http://browser.localhost', basic_auth: {
    username: 'browser-test', password: 'browser-test-password', secret: 'inert-browser-e2e-signing-secret-1234567890'
  } }, terminal: { backend: 'local', cwd: sandbox }
}))
writeEnvFile(home, 'e2e-mock-key', mock.url)
const sample = path.join(sandbox, 'sample.txt')
await writeFile(sample, 'Browser backend file round trip\n')
const child = spawn(python, ['-m', 'hermes_cli.main', 'serve', '--ui', 'desktop', '--skip-build', '--isolated', '--port', '0'], {
  cwd: root,
  env: { ...process.env, HERMES_HOME: home, HERMES_RUNTIME_DIR: path.join(sandbox, 'runtime'),
    HERMES_WEB_DIST: path.join(root, 'apps/desktop/dist/browser'), HERMES_DESKTOP: '', HERMES_SERVE_HEADLESS: '', HERMES_DASHBOARD_PUBLIC_URL: 'http://browser.localhost',
    HERMES_DASHBOARD_BASIC_AUTH_USERNAME: 'browser-test', HERMES_DASHBOARD_BASIC_AUTH_PASSWORD: 'browser-test-password',
    HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH: '', HERMES_DASHBOARD_BASIC_AUTH_SECRET: 'inert-browser-e2e-signing-secret-1234567890' },
  stdio: ['ignore', 'pipe', 'pipe']
})
const exited = once(child, 'exit')
let browser
let page
let backendErrors = ''
child.stderr.on('data', data => { backendErrors = (backendErrors + String(data)).slice(-6000) })
try {
  const port = await new Promise((resolve, reject) => {
    let output = ''
    const timer = setTimeout(() => reject(new Error('Backend readiness timed out.')), 60_000)
    child.stdout.on('data', data => {
      output += String(data)
      const match = /HERMES_DASHBOARD_READY port=(\d+)/.exec(output)
      if (match) { clearTimeout(timer); resolve(Number(match[1])) }
    })
    child.once('exit', code => { clearTimeout(timer); reject(new Error(`Backend exited before readiness (${code}).`)) })
    child.stderr.on('data', () => {})
  })
  browser = await chromium.launch({ headless: true })
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } })
  page = await context.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(`http://browser.localhost:${port}/`)
  await page.locator('input[name="username"]').fill('browser-test')
  await page.locator('input[name="password"]').fill('browser-test-password')
  await page.locator('button[type="submit"]').click()
  await page.locator('[contenteditable="true"]').waitFor()
  const optOut = page.getByRole('button', { name: 'No thanks', exact: true })
  if (await optOut.isVisible()) await optOut.click()
  const editor = page.locator('[contenteditable="true"]')
  await editor.fill('Hello from the browser client')
  await editor.press('Enter')
  await page.getByText(MOCK_REPLY, { exact: true }).first().waitFor({ timeout: 45_000 })
  await editor.fill('E2E_CORRECTION_SWITCH_TRIGGER')
  await editor.press('Enter')
  const finalReply = 'The corrected task finished.'
  await page.getByText(finalReply, { exact: true }).first().waitFor({ timeout: 45_000 })
  await page.reload()
  await page.getByText(finalReply, { exact: true }).first().waitFor({ timeout: 30_000 })
  const content = await page.evaluate(async sample => window.hermesDesktop.api({
    path: `/api/fs/read-text?path=${encodeURIComponent(sample)}`
  }), sample)
  if (content.text !== 'Browser backend file round trip\n') throw new Error('Backend file read failed.')
  const sessionId = decodeURIComponent(new URL(page.url()).hash.slice(2))
  const popoutPromise = context.waitForEvent('page')
  await page.evaluate(id => window.hermesDesktop.openSessionWindow(id, { watch: true }), sessionId)
  const popout = await popoutPromise
  await popout.getByText(finalReply, { exact: true }).first().waitFor({ timeout: 30_000 })
  if (new URL(popout.url()).searchParams.get('watch') !== '1') throw new Error('Spectator route missing.')
  await popout.close()
  const phoneContext = await browser.newContext({ viewport: { width: 393, height: 852 }, isMobile: true, hasTouch: true, storageState: await context.storageState() })
  const phone = await phoneContext.newPage()
  await phone.goto(page.url())
  await phone.getByText(finalReply, { exact: true }).first().waitFor({ timeout: 30_000 })
  await phone.locator('[contenteditable="true"]').click()
  const phoneFont = await phone.locator('[contenteditable="true"]').evaluate(element => Number.parseFloat(window.getComputedStyle(element).fontSize))
  if (phoneFont < 16) throw new Error('Touch composer risks Safari focus zoom below 16px.')
  if (await phone.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Phone layout overflows horizontally.')
  const tabClose = phone.locator('[data-slot="pane-tab-close-affordance"] button').first()
  await tabClose.click({ trial: true })
  if (await tabClose.locator('..').evaluate(element => window.getComputedStyle(element).opacity) !== '1') throw new Error('Touch tab close depends on hover.')
  const footerTargets = await phone.locator('[data-slot="statusbar"] button, [data-slot="statusbar"] a').evaluateAll(elements =>
    elements.filter(element => element.checkVisibility()).map(element => element.getBoundingClientRect().height))
  if (!footerTargets.length || footerTargets.some(height => height < 44)) throw new Error('Footer controls need 44px touch targets.')
  await phone.getByRole('button', { name: 'Show sidebar', exact: true }).tap()
  await phone.locator('[data-narrow-overlay]').waitFor()
  await phone.touchscreen.tap(365, 400)
  await phone.locator('[data-narrow-overlay]').waitFor({ state: 'hidden' })
  await phone.goto(`http://browser.localhost:${port}/#/profiles`)
  const profileEditor = phone.locator('.cm-content[contenteditable="true"]').first()
  await profileEditor.waitFor()
  await profileEditor.click()
  const profileFont = await profileEditor.evaluate(element => Number.parseFloat(window.getComputedStyle(element).fontSize))
  if (profileFont < 16) throw new Error('Touch profile editor risks Safari focus zoom below 16px.')
  if (errors.length) throw new Error(errors.join('\n'))
  console.log('PASS: password login, cookie-authenticated gateway, streaming chat, tool calls, reload/resume, backend files, spectator popout, phone layout')
} catch (error) {
  console.error('Browser state:', await page?.locator('body').innerText().catch(() => 'unavailable'))
  console.error('Backend diagnostics:', backendErrors)
  throw error
} finally {
  await browser?.close()
  if (child.exitCode === null) child.kill('SIGTERM')
  await exited
  await mock.close()
  await rm(sandbox, { recursive: true, force: true })
}

import { readdirSync, readFileSync } from 'node:fs'
import { createServer, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

const PAGE = `<!doctype html><html><head><title>Design fixture</title></head><body>
  <button id="native-target" style="padding: 24px">Inspect this button</button>
  <p id="native-control">Ordinary page control</p>
</body></html>`

let fixture: MockBackendFixture | null = null
let server: Server | null = null

test('existing Electron Browser selects a guest element and inserts explicitly marked DOM context', async () => {
  test.setTimeout(300_000)

  try {
    server = createServer((_request, response) => {
      response.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      response.end(PAGE)
    })
    await new Promise<void>(resolve => server!.listen(0, '127.0.0.1', resolve))
    const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}/`

    fixture = await setupMockBackend()
    try {
      await waitForAppReady(fixture, 35_000)
    } catch (error) {
      // Capture the isolated app's boot state; a connecting overlay is not a
      // Design Mode failure and must not be reported as one.
      const boot = await fixture.page.evaluate(() => (
        window as Window & { hermesDesktop?: { getBootProgress: () => Promise<unknown> } }
      ).hermesDesktop?.getBootProgress()).catch(() => null)

      console.error('Isolated Desktop boot progress:', JSON.stringify(boot))
      console.error('Electron exit code:', fixture.app.process().exitCode)
      for (const root of [
        fixture.sandbox.root,
        fixture.sandbox.hermesHome,
        fixture.sandbox.userDataDir,
        `${fixture.sandbox.hermesHome}/logs`,
        `${fixture.sandbox.userDataDir}/logs`
      ]) {
        try {
          console.error('Sandbox entries:', root, readdirSync(root).join(', '))
        } catch { /* Already removed during an app crash. */ }
      }
      try {
        const lines = readFileSync(`${fixture.sandbox.hermesHome}/logs/desktop.log`, 'utf8').split(/\r?\n/)
        const relevant = lines.filter(line => /boot|backend|spawn|fail|error|python|port/i.test(line)).slice(-18)

        console.error('Isolated Desktop log markers:', relevant.map(line => line
          .replace(/Bearer\s+\S+/gi, 'Bearer [REDACTED]')
          .replace(/(token|key|secret|password)\s*[=:]\s*\S+/gi, '$1=[REDACTED]')
          .slice(0, 300)).join('\n'))
      } catch { /* The test still reports the boot-state failure. */ }
      throw error
    }
    const page = fixture.page

    await page.keyboard.press('Control+k')
    await page.getByText('Toggle browser', { exact: true }).click()
    const address = page.getByRole('textbox', { name: 'Address' })
    await address.fill(url)
    await address.press('Enter')

    const guest = page.locator('webview').first()
    await expect
      .poll(() => guest.evaluate(webview => (webview as Electron.WebviewTag).executeJavaScript('document.title')))
      .toBe('Design fixture')

    await page.getByRole('button', { name: 'Design Mode', exact: true }).click()
    await expect
      .poll(() =>
        guest.evaluate(webview =>
          (webview as Electron.WebviewTag).executeJavaScript(
            'Boolean(document.querySelector("hermes-design")) && typeof window.__hermesDesign === "undefined"'
          )
        )
      )
      .toBe(true)
    await expect
      .poll(() =>
        guest.evaluate(webview =>
          (webview as Electron.WebviewTag).executeJavaScript(
            'document.querySelector("hermes-design")?.shadowRoot?.querySelector("[data-design-selected]")?.style.display'
          )
        )
      )
      .toBe('none')

    const point = await guest.evaluate(async webview => {
      const rect = webview.getBoundingClientRect()

      const inner = (await (webview as Electron.WebviewTag).executeJavaScript(
        'document.getElementById("native-target").getBoundingClientRect().toJSON()'
      )) as DOMRect

      return { x: rect.x + inner.x + inner.width / 2, y: rect.y + inner.y + inner.height / 2 }
    })

    await page.mouse.click(point.x, point.y)
    await expect(page.getByRole('button', { name: 'Add selection to prompt' })).toBeVisible()
    await page.getByRole('button', { name: 'Add selection to prompt' }).click()
    await expect(page.locator('[contenteditable="true"]').first()).toContainText(
      'Selected DOM context (not a verified source location):'
    )
    await expect(page.locator('[contenteditable="true"]').first()).toContainText('native-target')

    await page.getByRole('button', { name: 'Stop Design Mode' }).click()
    await expect(page.getByRole('button', { name: 'Design Mode', exact: true })).toBeVisible()
    await expect
      .poll(() =>
        guest.evaluate(webview =>
          (webview as Electron.WebviewTag).executeJavaScript(
            '!document.querySelector("hermes-design") && typeof window.__hermesDesign === "undefined"'
          )
        )
      )
      .toBe(true)
  } finally {
    await fixture?.cleanup()
    fixture = null

    if (server) {
      await new Promise<void>((resolve, reject) => server!.close(error => (error ? reject(error) : resolve())))
      server = null
    }
  }
})

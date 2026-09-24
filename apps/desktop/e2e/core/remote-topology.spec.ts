/**
 * Remote-backend topology: the Desktop attached by URL + token to a real
 * `hermes serve` it did NOT spawn, running under a different HOME and
 * filesystem root (./remote-helpers.ts). Only the LLM is faked.
 *
 *  - first chat over the remote route: the turn persists in the BACKEND's
 *    state.db, and the client never starts a local backend;
 *  - attaching an image that exists only on the client sends its bytes, never
 *    the client path (#120730): the model receives the image, the client path
 *    never reaches the backend, no "image not found";
 *  - a sidebar rename survives the backend being killed and restarted
 *    (#121192): the title is in the backend's state.db and the sidebar still
 *    shows it after the app reconnects.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, type Page, test } from '@playwright/test'

import {
  backendProcesses,
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  send,
  storedSessionForMarker,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { assertTranscriptOracle, installDuplicateSampler, type OracleTarget } from './oracle'
import { startScriptedProvider } from './provider'
import { remoteEnv, sessionRows, startRemoteBackend, uniquePng } from './remote-helpers'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

function imageTurnUser(body: any): any {
  const messages: any[] = Array.isArray(body?.messages) ? body.messages : []

  return [...messages].reverse().find(m => m?.role === 'user')
}

const dataUrlBytes = (url: string) => Buffer.from(/;base64,(.*)$/s.exec(url)?.[1] ?? '', 'base64')

/** Every image part in the last user message of a provider request (main turn or vision call). */
function imageParts(body: any): string[] {
  const user = imageTurnUser(body)
  const content = Array.isArray(user?.content) ? user.content : []

  return content
    .filter((part: any) => part?.type === 'image_url')
    .map((part: any) => String(part.image_url?.url ?? part.image_url ?? ''))
}

/** The sidebar session row showing `text` (rows own a [data-row-actions] cluster; chat bubbles do not). */
function sidebarRow(page: Page, text: string) {
  const row = '*:has(> [data-row-actions])'

  return page.locator(`${row}:not(${row} *)`).filter({ hasText: text, visible: true }).first()
}

async function stubImagePicker(app: Awaited<ReturnType<typeof launchCoreApp>>['app'], filePath: string) {
  await app.evaluate(({ dialog }, filePath) => {
    ;(dialog as any).showOpenDialog = async () => ({ canceled: false, filePaths: [filePath] })
  }, filePath)
}

test('remote backend: first chat, image bytes not client paths, rename across a backend restart', async () => {
  const provider = await startScriptedProvider()
  const backendBox = createCoreSandbox('remote-backend')
  const clientBox = createCoreSandbox('remote-client')
  writeProviderHome(backendBox.hermesHome, provider.url, 'agent:\n  image_input_mode: native\n')
  const backend = await startRemoteBackend(backendBox)
  const { app, page } = await launchCoreApp(coreAppEnv(clientBox, remoteEnv(backend)))
  const ws = recordWebSockets(page)

  const finished = (marker: string) =>
    expect
      .poll(() => provider.completions.some(c => c.marker === marker && c.finished), {
        timeout: 120_000,
        message: `provider finished ${marker}`
      })
      .toBe(true)

  const session: OracleTarget = { sessionId: '', expectUserMarkers: [] }

  try {
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)

    await test.step('first chat goes to the remote backend, never a local one', async () => {
      provider.script(U(1), [{ text: [`${A(1)} `, 'remote ', 'hello'] }])
      await send(page, `${U(1)} hi remote`, 'Enter', ws)
      await finished(U(1))
      await expect(viewport(page)).toContainText(A(1))
      await expect.poll(() => storedSessionForMarker(backendBox, 'default', U(1))).not.toBeNull()
      session.sessionId = await currentSessionId(page)
      session.expectUserMarkers.push(U(1))
      await assertTranscriptOracle(page, ws, provider, session, 'remote first chat')
      expect(new URL(ws.sockets[0]!.url).port, 'the socket dials the remote backend').toBe(String(backend.port))
      expect(backendProcesses(clientBox), 'the client spawned no backend of its own').toEqual([])
      expect(fs.existsSync(path.join(clientBox.hermesHome, 'state.db')), 'no client-side state.db').toBe(false)
    })

    await test.step('an image that exists only on the client reaches the model as bytes (#120730)', async () => {
      const png = uniquePng(`client-only-${nonce}`)
      const clientPath = path.join(clientBox.root, 'client-only-pictures', `shot-${nonce}.png`)
      fs.mkdirSync(path.dirname(clientPath), { recursive: true })
      fs.writeFileSync(clientPath, png)
      await stubImagePicker(app, clientPath)
      await page.locator('[data-slot="composer-root"] button:has(.codicon-add)').filter({ visible: true }).first().click()
      await page.getByRole('menuitem', { name: /image/i }).first().click()
      await expect(page.locator('[data-slot="composer-root"]').getByText(`shot-${nonce}.png`).first()).toBeVisible()

      provider.script(U(2), [{ text: [`${A(2)} `, 'saw ', 'it'] }])
      await send(page, `${U(2)} what is in this picture`, 'Enter', ws)
      await finished(U(2))
      await expect(viewport(page)).toContainText(A(2))
      await expect(page.getByText(/image not found/i)).toHaveCount(0)

      // The client read the file and shipped its bytes: exactly one byte-upload
      // attach, never a path-based attach naming the client file.
      const attaches = ws.sent.filter(f => f.method === 'image.attach' || f.method === 'image.attach_bytes')
      expect(
        attaches.map(f => f.method),
        'one byte upload, no path attach'
      ).toEqual(['image.attach_bytes'])
      expect(Buffer.from(String(attaches[0]!.params.content_base64), 'base64').equals(png)).toBe(true)

      // The backend staged those bytes under ITS home, and the model saw that image.
      const turn = provider.completions.find(c => c.marker === U(2))!
      const user = imageTurnUser(turn.body)
      const userText = typeof user?.content === 'string' ? user.content : JSON.stringify(user?.content ?? '')
      const at = userText.indexOf(backendBox.hermesHome)
      const staged = at >= 0 ? /^\S+?\.png/.exec(userText.slice(at))?.[0] : undefined
      expect(staged, `the turn references a backend-side image path: ${userText.slice(0, 400)}`).toBeTruthy()
      expect(fs.readFileSync(staged!).equals(png), 'backend-staged image is byte-identical').toBe(true)
      const seen = provider.completions.flatMap(c => imageParts(c.body)).map(dataUrlBytes)
      expect(
        seen.some(bytes => bytes.equals(png)),
        'the model received the client image bytes'
      ).toBe(true)

      const everySent = JSON.stringify(provider.completions.map(c => c.body))
      expect(everySent.includes(clientBox.root), 'no client-side path ever reached the backend').toBe(false)
      const sentOnWire = JSON.stringify(ws.sent)
      expect(sentOnWire.includes(`"path":"${clientPath}"`), 'the client path was never sent as an attach path').toBe(false)
      session.expectUserMarkers.push(U(2))
    })

    await test.step('a sidebar rename survives a backend kill + restart (#121192)', async () => {
      const title = `Renamed ${nonce}`
      const row = sidebarRow(page, U(1))
      await expect(row).toBeVisible({ timeout: 60_000 })
      await row.click({ button: 'right' })
      await page.getByRole('menuitem', { name: /rename/i }).first().click()
      const input = page.getByRole('dialog').getByRole('textbox').first()
      await expect(input).toBeVisible()
      await input.fill(title)
      await input.press('Enter')
      await expect
        .poll(() => sessionRows(backendBox).find(r => r.id === session.sessionId)?.title ?? null, {
          message: 'rename persisted in the backend state.db'
        })
        .toBe(title)

      await backend.restart()
      await expect
        .poll(() => sessionRows(backendBox).find(r => r.id === session.sessionId)?.title ?? null)
        .toBe(title)
      await page.reload()
      await waitForInteractive(app, page)
      await expect(sidebarRow(page, title)).toBeVisible({ timeout: 60_000 })

      // The renamed session keeps working after the restart.
      provider.script(U(3), [{ text: [`${A(3)} `, 'after ', 'restart'] }])
      await page.evaluate(id => {
        window.location.hash = `#/${encodeURIComponent(id)}`
      }, session.sessionId)
      await expect(viewport(page)).toContainText(A(2), { timeout: 60_000 })
      await send(page, `${U(3)} still there`, 'Enter', ws)
      await finished(U(3))
      await expect
        .poll(() => sessionRows(backendBox).find(r => r.id === session.sessionId)?.title ?? null)
        .toBe(title)
    })
  } finally {
    await app.close().catch(() => undefined)
    await backend.kill()
    await provider.close()
    clientBox.cleanup()
    backendBox.cleanup()
  }
})

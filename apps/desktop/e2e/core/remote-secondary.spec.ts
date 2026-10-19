/**
 * Bot Mode across two machines (#120730, the reported shape): the Desktop
 * runs its OWN local primary backend (Electron-spawned `hermes serve`, the
 * default profile only) and has a second, REMOTE connection registered in
 * connections.json — a real `hermes serve` the test spawns under a different
 * HOME/HERMES_HOME (./remote-helpers.ts) that hosts the bot profile. The user
 * opens that bot from the Bots roster, so the chat is owned by the remote
 * secondary while the window's primary stays local. Only the LLM is faked.
 *
 * Attaching an image that exists only on the client must reach the remote
 * backend as bytes (image.attach_bytes), never as a client path the remote
 * cannot resolve ("image not found: /Users/... (4016)"). The client decides
 * per session: use-prompt-actions `uploadBytes = remote || ...` with
 * `remote = isSessionRemote(session)` — the OWNER's mode, not the ambient
 * (local) connection's.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, type Page, test } from '@playwright/test'

import {
  backendProcesses,
  composer,
  coreAppEnv,
  createCoreSandbox,
  launchCoreApp,
  recordWebSockets,
  storedSessionForMarker,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { startScriptedProvider } from './provider'
import { startRemoteBackend, uniquePng } from './remote-helpers'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U1 = `U1-${nonce}`
const A1 = `A1-${nonce}`

const REMOTE_ID = 'linuxhost'
const REMOTE_LABEL = 'Linux host'
const BOT = 's120730bot'
const IMAGE_CONFIG = 'agent:\n  image_input_mode: native\n'

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

function lastUser(body: any): any {
  const messages: any[] = Array.isArray(body?.messages) ? body.messages : []

  return [...messages].reverse().find(m => m?.role === 'user')
}

const dataUrlBytes = (url: string) => Buffer.from(/;base64,(.*)$/s.exec(url)?.[1] ?? '', 'base64')

/** Every image part in the last user message of a provider request (main turn or vision call). */
function imageParts(body: any): string[] {
  const content = Array.isArray(lastUser(body)?.content) ? lastUser(body).content : []

  return content
    .filter((part: any) => part?.type === 'image_url')
    .map((part: any) => String(part.image_url?.url ?? part.image_url ?? ''))
}

interface AttachCall {
  socketUrl: string
  method: string
  params: any
  reply?: { result?: any; error?: { code?: number; message?: string } }
}

/** image.attach / image.attach_bytes requests with the socket they went out on and their JSON-RPC reply. */
function recordImageAttaches(page: Page): AttachCall[] {
  const calls: AttachCall[] = []

  page.on('websocket', ws => {
    if (!ws.url().includes('/api/ws')) {
      return
    }

    const pending = new Map<string, AttachCall>()
    ws.on('framesent', frame => {
      try {
        const msg = JSON.parse(String(frame.payload))

        if (msg?.method === 'image.attach' || msg?.method === 'image.attach_bytes') {
          const call: AttachCall = { socketUrl: ws.url(), method: msg.method, params: msg.params }
          calls.push(call)
          pending.set(String(msg.id), call)
        }
      } catch {
        /* non-JSON */
      }
    })
    ws.on('framereceived', frame => {
      try {
        const msg = JSON.parse(String(frame.payload))
        const call = msg?.id !== undefined ? pending.get(String(msg.id)) : undefined

        if (call) {
          call.reply = { result: msg.result, error: msg.error }
          pending.delete(String(msg.id))
        }
      } catch {
        /* non-JSON */
      }
    })
  })

  return calls
}

function writeConnectionsRegistry(userDataDir: string, url: string, token: string): void {
  fs.writeFileSync(
    path.join(userDataDir, 'connections.json'),
    JSON.stringify({
      version: 2,
      primary: 'local',
      launchMode: 'primary',
      lastUsed: 'local',
      connections: [
        { id: 'local', kind: 'local', label: 'This device' },
        {
          id: REMOTE_ID,
          kind: 'remote',
          label: REMOTE_LABEL,
          url,
          authMode: 'token',
          token: { encoding: 'plain', value: token }
        }
      ]
    }),
    { encoding: 'utf8', mode: 0o600 }
  )
}

const port = (url: string) => new URL(url).port
const attachReplyText = (calls: AttachCall[]) => JSON.stringify(calls.map(c => [c.method, c.reply ?? 'no reply']))

test('bot on a remote secondary connection: a client-only image reaches it as bytes, not a client path (#120730)', async () => {
  const provider = await startScriptedProvider()
  const remoteBox = createCoreSandbox('secondary-remote')
  const clientBox = createCoreSandbox('secondary-client')
  // The client's own local backend (default profile) can answer too, so a
  // mis-routed turn would still complete — only the owner assertions tell.
  writeProviderHome(clientBox.hermesHome, provider.url, IMAGE_CONFIG)
  writeProviderHome(remoteBox.hermesHome, provider.url, IMAGE_CONFIG)
  const botHome = path.join(remoteBox.hermesHome, 'profiles', BOT)
  writeProviderHome(botHome, provider.url, IMAGE_CONFIG)
  // The picture exists on the client. The remote backend gets a private mount
  // namespace in which this directory is empty: the other machine's view.
  const png = uniquePng(`client-only-${nonce}`)
  const picturesDir = path.join(clientBox.root, 'client-only-pictures')
  const clientPath = path.join(picturesDir, `shot-${nonce}.png`)
  fs.mkdirSync(picturesDir, { recursive: true })
  fs.writeFileSync(clientPath, png)
  const remote = await startRemoteBackend(remoteBox, { hide: [picturesDir] })
  writeConnectionsRegistry(clientBox.userDataDir, remote.url, remote.token)

  if (remote.hidden) {
    // Seen through the backend's own root: the directory is there, the file is not.
    const backendView = (p: string) => `/proc/${remote.pid()}/root${p}`
    expect(fs.statSync(backendView(picturesDir)).isDirectory(), 'backend root view readable').toBe(true)
    expect(fs.existsSync(backendView(clientPath)), 'the remote backend cannot see the client file').toBe(false)
    expect(fs.existsSync(clientPath), 'the client can').toBe(true)
  } else {
    // No unprivileged user namespaces on this runner: the backend could read
    // the client path, so only the wire assertions below can catch a path attach.
    test.info().annotations.push({ type: 'fidelity', description: 'client file visible to the backend (shared fs)' })
  }

  const { app, page } = await launchCoreApp(coreAppEnv(clientBox))
  const ws = recordWebSockets(page)
  const attaches = recordImageAttaches(page)

  try {
    await waitForInteractive(app, page)

    await test.step('the window runs on the LOCAL primary backend', async () => {
      await expect
        .poll(() => backendProcesses(clientBox).length, { message: 'Electron spawned a local backend' })
        .toBe(1)
      await expect.poll(() => ws.sockets.length, { message: 'primary socket dialed' }).toBeGreaterThan(0)
      expect(port(ws.sockets[0]!.url), 'the primary socket is the local backend, not the remote').not.toBe(
        String(remote.port)
      )
    })

    await test.step('open the remote bot from the Bots roster', async () => {
      await page
        .getByRole('button', { name: 'Bots', exact: true })
        .or(page.getByRole('tab', { name: 'Bots', exact: true }))
        .first()
        .click()
      const row = page.locator(`[data-slot="bots-roster"] [data-roster-key="${REMOTE_ID}::${BOT}"]`)
      await expect(row, 'the remote connection enumerated its bot').toBeVisible({ timeout: 60_000 })
      await row.click()
      // The bot's canonical chat is resumed on a socket to the REMOTE backend, scoped to the bot profile.
      await expect
        .poll(
          () =>
            ws.sent.some(f => {
              const url = new URL(ws.sockets[f.socket]!.url)

              return (
                f.method === 'session.resume' &&
                url.port === String(remote.port) &&
                url.searchParams.get('profile') === BOT
              )
            }),
          { message: 'bot chat resumed on the remote backend', timeout: 90_000 }
        )
        .toBe(true)
      await expect(composer(page)).toBeVisible()
    })

    await test.step('attach a client-only image and send', async () => {
      await app.evaluate(({ dialog }, filePath) => {
        ;(dialog as any).showOpenDialog = async () => ({ canceled: false, filePaths: [filePath] })
      }, clientPath)
      await page
        .locator('[data-slot="composer-root"] button:has(.codicon-add)')
        .filter({ visible: true })
        .first()
        .click()
      await page.getByRole('menuitem', { name: /image/i }).first().click()
      const chip = page.locator('[data-slot="composer-root"]').getByText(`shot-${nonce}.png`).first()
      await expect(chip).toBeVisible()
      provider.script(U1, [{ text: [`${A1} `, 'saw ', 'it'] }])
      const box = composer(page)
      await box.click()
      await box.fill(`${U1} what is in this picture`)
      await expect(box).toContainText(U1)
      // Enter until the submit is taken (the composer keeps the draft while a
      // gateway is still waking); stop at the first attach so it never repeats.
      await expect
        .poll(
          async () => {
            if (attaches.length > 0) {
              return 'submitted'
            }

            await box.press('Enter')

            return 'pending'
          },
          { timeout: 120_000, intervals: [1_000, 2_000, 4_000], message: 'the send staged its attachment' }
        )
        .toBe('submitted')

      // Settle: every attach answered, then the turn finished or the attach failed.
      await expect
        .poll(
          () => {
            if (attaches.some(c => !c.reply)) {
              return 'attach pending'
            }

            if (attaches.some(c => c.reply?.error || c.reply?.result?.attached === false)) {
              return 'settled'
            }

            return provider.completions.some(c => c.marker === U1 && c.finished) ? 'settled' : 'turn pending'
          },
          { timeout: 120_000, message: `send settled; attach replies ${attachReplyText(attaches)}` }
        )
        .toBe('settled')
    })

    await test.step('the image crossed as bytes to the remote owner', async () => {
      // Exactly one byte upload, never a path-based attach naming the client file.
      expect(
        attaches.map(c => c.method),
        `one byte upload, no path attach; attach replies ${attachReplyText(attaches)}`
      ).toEqual(['image.attach_bytes'])
      const [call] = attaches
      expect(port(call!.socketUrl), 'the bytes went to the remote backend').toBe(String(remote.port))
      expect(new URL(call!.socketUrl).searchParams.get('profile'), 'on the bot profile socket').toBe(BOT)
      expect(Buffer.from(String(call!.params.content_base64), 'base64').equals(png), 'uploaded bytes == file').toBe(
        true
      )
      expect(call!.reply?.error, 'attach_bytes succeeded').toBeUndefined()
      expect(call!.reply?.result?.attached, 'attach_bytes attached').toBe(true)
    })

    await test.step('the remote backend ran the turn and the model saw the image', async () => {
      await expect(viewport(page)).toContainText(A1)
      await expect
        .poll(() => storedSessionForMarker(remoteBox, BOT, U1), {
          message: 'turn persisted in the REMOTE bot state.db'
        })
        .not.toBeNull()
      expect(storedSessionForMarker(clientBox, 'default', U1), 'nothing persisted in the LOCAL state.db').toBeNull()
      expect(storedSessionForMarker(clientBox, BOT, U1), 'no local bot profile ran it either').toBeNull()
      const submit = ws.sent.find(f => f.method === 'prompt.submit' && String(f.params?.text ?? '').includes(U1))
      expect(submit, 'prompt.submit was sent').toBeTruthy()
      expect(port(ws.sockets[submit!.socket]!.url), 'the turn was submitted to the remote backend').toBe(
        String(remote.port)
      )

      // The remote staged those bytes under ITS bot home, byte-identical.
      const turn = provider.completions.find(c => c.marker === U1)!
      const user = lastUser(turn.body)
      const userText = typeof user?.content === 'string' ? user.content : JSON.stringify(user?.content ?? '')
      const at = userText.indexOf(remoteBox.hermesHome)
      const staged = at >= 0 ? /^\S+?\.png/.exec(userText.slice(at))?.[0] : undefined
      expect(staged, `the turn references a remote-side image path: ${userText.slice(0, 400)}`).toBeTruthy()
      expect(fs.readFileSync(staged!).equals(png), 'remote-staged image is byte-identical').toBe(true)
      const seen = provider.completions.flatMap(c => imageParts(c.body)).map(dataUrlBytes)
      expect(
        seen.some(bytes => bytes.equals(png)),
        'the model received the client image bytes'
      ).toBe(true)

      expect(
        JSON.stringify(provider.completions.map(c => c.body)).includes(clientBox.root),
        'no client-side path ever reached the backend'
      ).toBe(false)
      expect(JSON.stringify(ws.sent).includes(clientPath), 'the client path never went on the wire').toBe(false)
    })

    // The reported symptom, from the backend's own replies and the UI.
    const notFound =
      attaches.some(c => /image not found/i.test(String(c.reply?.error?.message ?? c.reply?.result?.message ?? ''))) ||
      (await page.getByText(/image not found/i).count()) > 0

    expect(notFound, `no "image not found"; attach replies ${attachReplyText(attaches)}`).toBe(false)
  } finally {
    await app.close().catch(() => undefined)
    await remote.kill()
    await provider.close()
    clientBox.cleanup()
    remoteBox.cleanup()
  }
})

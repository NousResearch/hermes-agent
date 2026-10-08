import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import * as fs from 'node:fs'
import * as net from 'node:net'
import * as path from 'node:path'
import { crc32, deflateSync } from 'node:zlib'

import { buildAppEnv, createSandbox, launchDesktop, writeEnvFile, writeMockProviderConfig } from './fixtures'
import { startMockServer } from './mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { expect as baseExpect, type ElectronApplication, test } from './test'

// Large-file decoding and native I/O must also settle on a busy CI worker.
const expect = baseExpect.configure({ timeout: 30_000 })

const REPO = path.resolve(import.meta.dirname, '../../..')
const TITLE = 'Remote media delivery'
const TOKEN = 'remote-media-e2e-only-token'

function pngChunk(type: string, data: Buffer): Buffer {
  const body = Buffer.concat([Buffer.from(type), data])
  const result = Buffer.alloc(body.length + 8)
  result.writeUInt32BE(data.length, 0)
  body.copy(result, 4)
  result.writeUInt32BE(crc32(body), body.length + 4)

  return result
}

/** A valid, decodable RGB PNG over the 16 MiB data-URL cap. No fixture-sized
 * binary in git and no image-library/ffmpeg dependency on the test runner. */
function largePng(): Buffer {
  const width = 2600
  const height = 2200
  const pixels = Buffer.alloc((width * 3 + 1) * height)

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const offset = y * (width * 3 + 1) + 1 + x * 3
      pixels[offset] = 40 + Math.floor((x / width) * 170)
      pixels[offset + 1] = 190 - Math.floor((y / height) * 90)
      pixels[offset + 2] = 165
    }
  }

  const header = Buffer.alloc(13)
  header.writeUInt32BE(width, 0)
  header.writeUInt32BE(height, 4)
  header[8] = 8
  header[9] = 2

  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk('IHDR', header),
    pngChunk('IDAT', deflateSync(pixels, { level: 0 })),
    pngChunk('IEND', Buffer.alloc(0))
  ])
}

async function unusedPort(): Promise<number> {
  const server = net.createServer()
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const port = (server.address() as net.AddressInfo).port
  await new Promise<void>(resolve => server.close(() => resolve()))

  return port
}

// Playwright requires destructured fixture parameters, even when none are used.
// eslint-disable-next-line no-empty-pattern
test('remote image displays, zooms, saves byte-for-byte and survives a reload alongside video', async ({}, testInfo) => {
  test.setTimeout(180_000)
  const sandbox = createSandbox('remote-media')
  const exportsDir = path.join(sandbox.root, 'exports')
  fs.mkdirSync(exportsDir)
  const imagePath = path.join(exportsDir, '写真 full size.png')
  const videoPath = path.join(exportsDir, 'preview.webm')
  const bytes = largePng()
  fs.writeFileSync(imagePath, bytes)
  fs.copyFileSync(path.join(import.meta.dirname, 'media-fixtures/preview.webm'), videoPath)

  const mock = await startMockServer({
    reply: `Here are your image and video.\n\nMEDIA:"${imagePath}"\n\nMEDIA:"${videoPath}"`
  })

  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  const port = await unusedPort()
  const baseUrl = `http://127.0.0.1:${port}`
  let app: ElectronApplication | undefined
  let backend: ReturnType<typeof spawn> | undefined

  try {
    const builder = await RealSessionBuilder.start(sandbox.hermesHome)

    try {
      await builder.createSession({ title: TITLE, turns: ['Show the image and video from the remote gateway.'] })
    } finally {
      await builder.close()
    }

    backend = spawn(
      'uv',
      [
        'run',
        '--active',
        '--no-sync',
        'python',
        '-c',
        `from hermes_cli.web_server import start_server; start_server(port=${port}, open_browser=False, headless=True)`
      ],
      {
        cwd: REPO,
        env: {
          ...buildAppEnv(sandbox),
          PYTHONPATH: REPO,
          HERMES_SERVE_HEADLESS: '1',
          HERMES_DASHBOARD_SESSION_TOKEN: TOKEN,
          HERMES_DASHBOARD_FILES_ROOT: exportsDir
        },
        stdio: 'ignore'
      }
    )
    await expect
      .poll(
        async () => {
          try {
            return (await fetch(`${baseUrl}/api/status`)).status
          } catch {
            return 0
          }
        },
        { timeout: 60_000 }
      )
      .toBe(200)

    const params = new URLSearchParams({ path: imagePath })

    const rejected = await fetch(`${baseUrl}/api/fs/read-data-url?${params}`, {
      headers: { 'X-Hermes-Session-Token': TOKEN }
    })

    expect(rejected.status).toBe(413)

    const launched = await launchDesktop(
      buildAppEnv(sandbox, {
        HERMES_DESKTOP_REMOTE_URL: baseUrl,
        HERMES_DESKTOP_REMOTE_TOKEN: TOKEN
      })
    )

    app = launched.app
    const { page } = launched
    const row = page.locator('[data-slot="sidebar"] button').filter({ hasText: TITLE }).first()
    await row.waitFor({ state: 'visible', timeout: 60_000 })
    await row.click()
    const image = page.locator('[data-slot="aui_markdown-image"] img').first()
    await expect(image).toBeVisible({ timeout: 30_000 })
    await expect.poll(() => image.evaluate(node => (node as HTMLImageElement).naturalWidth)).toBe(2600)
    await expect(image).toHaveAttribute('src', /^hermes-media:\/\/remote\//)
    expect(await image.getAttribute('src')).not.toContain(TOKEN)

    const video = page.locator('video').first()
    await expect
      .poll(() =>
        video.evaluate(element => {
          const node = element as HTMLVideoElement

          return Number.isFinite(node.duration) && node.duration > 0
        })
      )
      .toBe(true)
    await video.evaluate(async element => {
      const node = element as HTMLVideoElement
      node.muted = true
      await node.play()
    })
    await expect.poll(() => video.evaluate(node => (node as HTMLVideoElement).currentTime)).toBeGreaterThan(0)

    await image.click()
    const zoomed = page.getByRole('dialog').locator('img')
    await expect.poll(() => zoomed.evaluate(node => (node as HTMLImageElement).naturalWidth)).toBe(2600)
    await page.keyboard.press('Escape')

    for (const saveMethod of ['button', 'context-menu']) {
      const savePath = path.join(sandbox.root, `saved-${saveMethod}.png`)
      await app.evaluate(({ dialog }, filePath) => {
        dialog.showSaveDialog = async () => ({ canceled: false, filePath })
      }, savePath)

      if (saveMethod === 'button') {
        await page
          .locator('[data-slot="aui_markdown-image"]')
          .first()
          .getByRole('button', { name: 'Download image' })
          .click()
      } else {
        await image.click({ button: 'right' })
        await page.getByRole('menuitem', { name: 'Save image as…' }).click()
      }

      await expect.poll(() => fs.existsSync(savePath)).toBe(true)
      await expect.poll(() => fs.statSync(savePath).size).toBe(bytes.length)
      expect(createHash('sha256').update(fs.readFileSync(savePath)).digest('hex')).toBe(
        createHash('sha256').update(bytes).digest('hex')
      )
    }

    await page.screenshot({ path: testInfo.outputPath('remote-media-delivery.png') })
    await page.reload()
    await row.waitFor({ state: 'visible', timeout: 60_000 })
    await row.click()
    await expect
      .poll(() => image.evaluate(node => (node as HTMLImageElement).naturalWidth), { timeout: 30_000 })
      .toBe(2600)
    await page.screenshot({ path: testInfo.outputPath('remote-media-reloaded.png') })
  } finally {
    await app?.close().catch(() => undefined)

    if (backend && backend.exitCode === null && backend.signalCode === null) {
      const exited = new Promise<void>(resolve => backend!.once('exit', () => resolve()))
      backend.kill('SIGTERM')
      await exited
    }

    await mock.close()
    sandbox.cleanup()
  }
})

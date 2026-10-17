/** Built Electron UI, real backend and downloader, loopback model bytes only. */
import { spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as http from 'node:http'
import * as path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'

import { buildAppEnv, createSandbox, launchDesktop } from './fixtures'
import { type ElectronApplication, expect, type Page, test } from './test'

const REPO_ROOT = path.resolve(import.meta.dirname, '../../..')
const BLOCK = 1 << 20
const payload = Buffer.alloc(4 * BLOCK, 'w')
const companion = Buffer.from('companion asset')

async function jobs(page: Page) {
  return page.evaluate(async () => {
    const desktop = (
      window as unknown as {
        hermesDesktop: {
          api: (request: { path: string }) => Promise<{
            jobs: Array<{ job_id: string; status: string; phase: string; done_bytes: number; total_bytes: number }>
          }>
        }
      }
    ).hermesDesktop

    const response = await desktop.api({ path: '/api/local-models/jobs' })

    return response.jobs
  })
}

async function openLocalModels(page: Page, expectResume = true): Promise<void> {
  await page.getByRole('button', { name: 'Open settings', exact: true }).click({ timeout: 90_000 })
  await page.getByRole('button', { name: 'Providers', exact: true }).click()
  await page.getByRole('button', { name: 'Local Models', exact: true }).click()

  if (expectResume) {
    await expect(page.getByRole('button', { name: 'Resume', exact: true })).toBeVisible({ timeout: 30_000 })
  }
}

for (const action of ['resume', 'discard'] as const) {
  test(`built UI pauses backoff and can ${action} a saved multi-file plan after app restart`, async () => {
    const testInfo = test.info()
    test.setTimeout(180_000)
    const sandbox = createSandbox('download-recovery')

    const python =
      process.env.HERMES_DESKTOP_PYTHON ??
      path.join(REPO_ROOT, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')

    const env = buildAppEnv(sandbox, { HERMES_DESKTOP_PYTHON: python })

    const stopBackend = () =>
      spawnSync(python, ['-m', 'hermes_cli.main', 'gateway', 'stop'], {
        cwd: REPO_ROOT,
        env,
        encoding: 'utf8',
        timeout: 30_000
      })

    let app: ElectronApplication | undefined
    let outage = true
    const requests: Array<{ path: string; start: number }> = []

    const server = http.createServer((request, response) => {
      const body = request.url === '/companion' ? companion : payload
      const match = /^bytes=(\d+)-(\d+)$/.exec(request.headers.range ?? '')

      if (!match) {
        response.writeHead(400).end()

        return
      }

      const start = Number(match[1])
      const end = Number(match[2])
      const probe = start === 0 && end === 0

      if (!probe) {
        requests.push({ path: request.url!, start })
      }

      if (!probe && outage) {
        response.writeHead(503).end()

        return
      }

      response.writeHead(206, {
        'Content-Range': `bytes ${start}-${end}/${body.length}`,
        'Content-Length': end - start + 1,
        ETag: '"recovery-fixture-v1"'
      })
      response.end(body.subarray(start, end + 1))
    })

    try {
      await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
      const address = server.address() as { port: number }
      const endpoint = `http://127.0.0.1:${address.port}`
      writeMockProviderConfig(sandbox.hermesHome, endpoint)
      writeEnvFile(sandbox.hermesHome)

      // Seed only durable disk state. No renderer state, job callback, API
      // response, or transfer operation is mocked.
      const seed = spawnSync(
        python,
        [
          '-c',
          [
            'import json, hashlib, sys',
            'from hermes_cli.local_runtime import bootstrap, catalog, download_plans',
            'from pm.paths import partials_root',
            'entry = catalog.CATALOG[0]',
            'models = bootstrap.models_dir(); models.mkdir(parents=True)',
            'first = models / "completed.gguf"; first.write_bytes(b"completed first shard")',
            'second = models / (entry.variants[0].model_id + ".gguf")',
            'asset = bootstrap.assets_dir() / "companion.gguf"',
            'url = sys.argv[1] + "/weights"; block = 1 << 20',
            'root = partials_root(); root.mkdir(parents=True)',
            'key = hashlib.sha256(url.encode()).hexdigest()',
            'with (root / (key + ".part")).open("wb") as stream:',
            ' stream.truncate(4 * block); stream.seek(0); stream.write(b"w" * block)',
            '(root / (key + ".ranges")).write_text(json.dumps({"total": 4 * block, "etag": \'"recovery-fixture-v1"\', "sha256": "", "ranges": [[0, block]]}))',
            'plan = [(sys.argv[1] + "/complete", first, first.stat().st_size), (url, second, 4 * block), (sys.argv[1] + "/companion", asset, 15)]',
            'download_plans.save_plan({"target": entry.display_name, "model_id": entry.id}, plan)',
            'print(json.dumps({"first": str(first), "second": str(second), "asset": str(asset), "part": str(root / (key + ".part")), "side": str(root / (key + ".ranges"))}))'
          ].join('\n'),
          endpoint
        ],
        { cwd: REPO_ROOT, env, encoding: 'utf8', timeout: 30_000 }
      )

      expect(seed.status, seed.stderr).toBe(0)

      const files = JSON.parse(seed.stdout) as {
        first: string
        second: string
        asset: string
        part: string
        side: string
      }

      const firstTime = fs.statSync(files.first).mtimeMs
      let launched = await launchDesktop(env)
      app = launched.app
      let page = launched.page
      await openLocalModels(page)
      const original = (await jobs(page))[0]
      expect(original.done_bytes).toBe(BLOCK + fs.statSync(files.first).size)
      await page.getByRole('button', { name: 'Resume', exact: true }).click()
      await expect(page.getByText('Connection interrupted. Retrying automatically.')).toBeVisible({ timeout: 15_000 })
      await page.screenshot({ path: testInfo.outputPath('retrying.png') })
      await page.getByRole('button', { name: 'Pause', exact: true }).click()
      await expect(page.getByRole('button', { name: 'Resume', exact: true })).toBeVisible({ timeout: 5_000 })
      await expect(page.getByText('Connection interrupted. Retrying automatically.')).toBeHidden()
      await page.getByRole('button', { name: 'Resume', exact: true }).click()
      await expect(page.getByText(/Download interrupted\. Resume to continue from saved progress/)).toBeVisible({
        timeout: 30_000
      })
      await expect(page.getByRole('button', { name: 'Resume', exact: true })).toBeEnabled()
      await page.screenshot({ path: testInfo.outputPath('interrupted.png') })
      expect((await jobs(page))[0].done_bytes).toBe(original.done_bytes)
      expect(fs.existsSync(files.second)).toBe(false)
      expect(fs.existsSync(files.asset)).toBe(false)

      await app.close()
      app = undefined
      // The gateway can outlive Electron. Stop only this sandbox's gateway
      // to prove recovery from disk, not from a surviving callback.
      const stopped = stopBackend()
      expect(stopped.status, stopped.stderr).toBe(0)
      launched = await launchDesktop(env)
      app = launched.app
      page = launched.page
      await openLocalModels(page)
      const restored = (await jobs(page))[0]
      expect(restored.job_id).not.toBe(original.job_id)
      expect(restored.done_bytes).toBe(original.done_bytes)

      if (action === 'discard') {
        const discard = page.getByRole('button', { name: 'Discard', exact: true })
        await discard.hover()
        await expect(page.getByRole('tooltip')).toContainText('Completed files will stay.')
        await page.screenshot({ path: testInfo.outputPath('discard.png') })
        await discard.click()
        await expect.poll(() => jobs(page)).toEqual([])
        expect(fs.existsSync(files.part)).toBe(false)
        expect(fs.existsSync(files.side)).toBe(false)
        await expect(discard).toBeHidden()
        await app.close()
        app = undefined
        const stoppedAfterDiscard = stopBackend()
        expect(stoppedAfterDiscard.status, stoppedAfterDiscard.stderr).toBe(0)
        launched = await launchDesktop(env)
        app = launched.app
        page = launched.page
        await openLocalModels(page, false)
        expect(await jobs(page)).toEqual([])
        expect(fs.readFileSync(files.first)).toEqual(Buffer.from('completed first shard'))
        expect(fs.statSync(files.first).mtimeMs).toBe(firstTime)
        expect(fs.existsSync(files.second)).toBe(false)
        expect(fs.existsSync(files.asset)).toBe(false)

        return
      }

      outage = false
      await page.getByRole('button', { name: 'Resume', exact: true }).click()
      await expect.poll(async () => (await jobs(page))[0].status, { timeout: 30_000 }).toBe('done')
      await expect(page.getByRole('button', { name: 'Pause', exact: true })).toBeHidden({ timeout: 10_000 })
      await expect(page.getByRole('button', { name: 'Resume', exact: true })).toBeHidden()
      await page.screenshot({ path: testInfo.outputPath('completed.png') })
      expect(fs.readFileSync(files.second)).toEqual(payload)
      expect(fs.readFileSync(files.asset)).toEqual(companion)
      expect(fs.statSync(files.first).mtimeMs).toBe(firstTime)
      expect(requests.filter(row => row.path === '/weights').every(row => row.start >= BLOCK)).toBe(true)
      expect(requests.some(row => row.path === '/complete')).toBe(false)
      const complete = (await jobs(page))[0]
      expect(complete.done_bytes).toBe(complete.total_bytes)
      await testInfo.attach('transfer-requests', { body: JSON.stringify(requests), contentType: 'application/json' })
    } finally {
      await app?.close()
      stopBackend()
      server.closeAllConnections()
      await new Promise<void>(resolve => server.close(() => resolve()))
      sandbox.cleanup()
    }
  })
}

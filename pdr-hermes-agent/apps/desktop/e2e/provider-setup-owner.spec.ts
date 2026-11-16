/**
 * Provider setup must keep the Settings owner's gateway AND profile.
 * Two real serve backends; only OpenAI-compatible model discovery is a fixture.
 * Build dist/ separately, then run this spec with HERMES_DESKTOP_PYTHON pointing
 * at a dependency-complete interpreter. No OAuth account or inference is used.
 */
import { type ChildProcess, spawn, spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as http from 'node:http'
import * as net from 'node:net'
import * as path from 'node:path'

import { createSandbox, findElectron, type Sandbox } from './fixtures'
import {
  _electron,
  collectErrorBanners,
  type ElectronApplication,
  expect,
  installErrorBannerGuard,
  type Page,
  test
} from './test'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')
const REMOTE_ID = 'athena-fixture'
const REMOTE_LABEL = 'Athena fixture'
const PROFILE = 'leverage-ai'
const REMOTE_TOKEN = 'provider-owner-fixture-token'

interface RpcReceipt {
  endpoint: string
  socketProfile: string | null
  method: string
  params: Record<string, unknown>
  result?: unknown
  error?: unknown
}

function isolatedEnv(home: string, hermesHome: string): Record<string, string> {
  // Allowlist rather than inheriting real provider keys, profile selection,
  // browser account state, live desktop overrides, or the host session bus.
  const env: Record<string, string> = {
    PATH: process.env.PATH ?? '',
    HOME: home,
    USERPROFILE: home,
    HERMES_HOME: hermesHome,
    PYTHONPATH: REPO_ROOT,
    PYTHONNOUSERSITE: '1',
    PYTHONDONTWRITEBYTECODE: '1',
    TMPDIR: process.env.TMPDIR ?? path.dirname(home),
    LANG: 'C.UTF-8',
    TZ: 'UTC',
    XDG_SESSION_TYPE: 'x11',
    ELECTRON_OZONE_PLATFORM_HINT: 'x11'
  }

  for (const key of ['DISPLAY', 'XAUTHORITY', 'TEST_WORKER_INDEX', 'SYSTEMROOT', 'WINDIR']) {
    if (process.env[key]) {
      env[key] = process.env[key]!
    }
  }

  for (const kind of ['CONFIG', 'CACHE', 'DATA', 'STATE', 'RUNTIME']) {
    const dir = path.join(home, `xdg-${kind.toLowerCase()}`)
    fs.mkdirSync(dir, { recursive: true, mode: 0o700 })
    env[`XDG_${kind}_${kind === 'RUNTIME' ? 'DIR' : 'HOME'}`] = dir
  }

  return env
}

function pythonBinary(): string {
  const python =
    process.env.HERMES_DESKTOP_PYTHON ??
    path.join(REPO_ROOT, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')

  if (!fs.existsSync(python)) {
    throw new Error('Set HERMES_DESKTOP_PYTHON to an isolated dependency-complete interpreter')
  }

  return python
}

function verifyImports(python: string, env: Record<string, string>): string {
  const probe = spawnSync(
    python,
    [
      '-c',
      [
        'import json, pathlib, sys',
        'import hermes_cli.main, hermes_cli.web_server, tui_gateway.server',
        'root = pathlib.Path.cwd().resolve()',
        'paths = {name: str(pathlib.Path(sys.modules[name].__file__).resolve()) for name in ("hermes_cli.main", "hermes_cli.web_server", "tui_gateway.server")}',
        'assert all(pathlib.Path(p).is_relative_to(root) for p in paths.values()), paths',
        'sys.__stdout__.write(json.dumps(paths))'
      ].join('\n')
    ],
    { cwd: REPO_ROOT, env, encoding: 'utf8', timeout: 60_000 }
  )

  expect(probe.status, probe.stderr).toBe(0)

  return probe.stdout
}

function seedConfig(home: string, endpoint: string, model: string): void {
  fs.mkdirSync(home, { recursive: true })
  // JSON is YAML, and these fixtures need no secret file at all.
  fs.writeFileSync(
    path.join(home, 'config.yaml'),
    JSON.stringify(
      {
        model: { provider: 'custom', default: model, base_url: `${endpoint}/v1` },
        auxiliary: { title_generation: { enabled: false } }
      },
      null,
      2
    )
  )
}

function configModel(python: string, env: Record<string, string>, home: string): unknown {
  const read = spawnSync(
    python,
    [
      '-c',
      'import json, pathlib, sys, yaml; print(json.dumps(yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())["model"]))',
      path.join(home, 'config.yaml')
    ],
    { cwd: REPO_ROOT, env, encoding: 'utf8', timeout: 10_000 }
  )

  expect(read.status, read.stderr).toBe(0)

  return JSON.parse(read.stdout)
}

async function listen(server: net.Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => resolve((server.address() as net.AddressInfo).port))
  })
}

async function stopChild(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) {
    return
  }

  await new Promise<void>(resolve => {
    const timer = setTimeout(() => child.kill('SIGKILL'), 10_000)
    child.once('exit', () => {
      clearTimeout(timer)
      resolve()
    })
    child.kill('SIGTERM')
  })
}

async function startRemote(python: string, env: Record<string, string>, logPath: string) {
  const reservation = net.createServer()
  const port = await listen(reservation)
  await new Promise<void>(resolve => reservation.close(() => resolve()))
  const url = `http://127.0.0.1:${port}`
  const log = fs.openSync(logPath, 'w')

  const child = spawn(
    python,
    ['-m', 'hermes_cli.main', 'serve', '--isolated', '--host', '127.0.0.1', '--port', String(port), '--skip-build'],
    {
      cwd: REPO_ROOT,
      env: { ...env, HERMES_DASHBOARD_SESSION_TOKEN: REMOTE_TOKEN },
      stdio: ['ignore', log, log]
    }
  )

  fs.closeSync(log)

  try {
    await expect
      .poll(
        async () => {
          if (child.exitCode !== null) {
            throw new Error(`Remote serve exited: ${child.exitCode}; see ${logPath}`)
          }

          return fetch(`${url}/api/status`, {
            headers: { 'X-Hermes-Session-Token': REMOTE_TOKEN },
            signal: AbortSignal.timeout(2_000)
          })
            .then(response => response.status)
            .catch(() => 0)
        },
        { timeout: 90_000, intervals: [200, 500, 1000] }
      )
      .toBe(200)
  } catch (error) {
    await stopChild(child)
    throw error
  }

  return { url, child }
}

function watchRpc(page: Page, receipts: RpcReceipt[]): void {
  page.on('websocket', socket => {
    const url = new URL(socket.url())
    const pending = new Map<unknown, RpcReceipt>()
    socket.on('framesent', ({ payload }) => {
      const message = JSON.parse(payload.toString())

      if (!message.method) {
        return
      }

      const receipt: RpcReceipt = {
        endpoint: url.origin,
        socketProfile: url.searchParams.get('profile'),
        method: message.method,
        params: message.params ?? {}
      }

      receipts.push(receipt)
      pending.set(message.id, receipt)
    })
    socket.on('framereceived', ({ payload }) => {
      const message = JSON.parse(payload.toString())
      const receipt = pending.get(message.id)

      if (receipt) {
        receipt.result = message.result
        receipt.error = message.error
      }
    })
  })
}

const gatewayGroup = (page: Page, id: string) =>
  page.locator(`[data-slot="profile-rail-gateway"][data-connection-id="${id}"]`)

async function selectGateway(page: Page, id: string, label: string, profile: string): Promise<void> {
  const group = gatewayGroup(page, id)
  await expect(group).toBeVisible({ timeout: 60_000 })
  await group.getByRole('button', { name: `${profile} · ${label}`, exact: true }).click()
  await expect(page.getByRole('button', { name: /^Registered gateways: / })).toHaveAttribute(
    'aria-label',
    `Registered gateways: ${label}`,
    { timeout: 90_000 }
  )
  await expect(group).toHaveAttribute('data-active', 'true')
  await expect(group.getByRole('button', { name: profile, exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect(page.locator('[data-slot="statusbar"]').getByText('ready', { exact: true })).toBeVisible({
    timeout: 60_000
  })
}

async function openProviderKeys(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Open settings', exact: true }).click()
  await page.getByRole('button', { name: 'Providers', exact: true }).click()
  await page.getByRole('button', { name: 'API keys', exact: true }).click()
  await expect(page.getByRole('button', { name: /^Local \/ custom endpoint/ })).toBeVisible()
}

test('Settings provider setup stays on its gateway/profile across A → B → A', async () => {
  test.setTimeout(300_000)
  const original = createSandbox('provider-owner')
  const localHome = path.join(original.root, 'local')
  const remoteHome = path.join(original.root, 'remote')
  const sandbox: Sandbox = { ...original, hermesHome: path.join(localHome, '.hermes') }
  const remoteHermesHome = path.join(remoteHome, '.hermes')
  const remoteProfileHome = path.join(remoteHermesHome, 'profiles', PROFILE)
  const python = pythonBinary()
  const localEnv = isolatedEnv(localHome, sandbox.hermesHome)
  const remoteEnv = isolatedEnv(remoteHome, remoteHermesHome)
  const receipts: RpcReceipt[] = []
  const discoveryRequests: string[] = []
  let advertisedModel = 'remote-first'

  const endpoint = http.createServer((request, response) => {
    discoveryRequests.push(request.url ?? '')
    // Discovery only. An unexpected inference request must not look successful.
    response.writeHead(request.url === '/v1/models' ? 200 : 404, { 'Content-Type': 'application/json' })
    response.end(
      JSON.stringify(
        request.url === '/v1/models'
          ? { object: 'list', data: [{ id: advertisedModel, object: 'model', owned_by: 'e2e-fixture' }] }
          : { error: 'This regression fixture only serves model discovery' }
      )
    )
  })

  let app: ElectronApplication | undefined
  let page: Page | undefined
  let remote: Awaited<ReturnType<typeof startRemote>> | undefined

  try {
    const endpointUrl = `http://127.0.0.1:${await listen(endpoint)}`
    seedConfig(sandbox.hermesHome, endpointUrl, 'local-sentinel')
    seedConfig(remoteHermesHome, endpointUrl, 'remote-default-sentinel')
    seedConfig(remoteProfileHome, endpointUrl, 'remote-before-setup')
    await test
      .info()
      .attach('worktree-imports', { body: verifyImports(python, remoteEnv), contentType: 'application/json' })
    remote = await startRemote(python, remoteEnv, test.info().outputPath('remote-serve.log'))
    fs.writeFileSync(
      path.join(sandbox.userDataDir, 'connections.json'),
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
            url: remote.url,
            authMode: 'token',
            token: { encoding: 'plain', value: REMOTE_TOKEN }
          }
        ]
      })
    )
    expect(fs.existsSync(path.join(DESKTOP_ROOT, 'dist/electron-main.mjs')), 'Build desktop dist before E2E').toBe(true)
    app = await _electron.launch({
      executablePath: findElectron(),
      args: [DESKTOP_ROOT, '--disable-gpu', '--no-sandbox'],
      cwd: DESKTOP_ROOT,
      ...(process.env.PROVIDER_SETUP_VIDEO === '1'
        ? {
            recordVideo: { dir: test.info().outputPath('video'), size: { width: 1220, height: 800 } }
          }
        : {}),
      env: {
        ...localEnv,
        HERMES_DESKTOP_PYTHON: python,
        HERMES_DESKTOP_USER_DATA_DIR: sandbox.userDataDir,
        HERMES_DESKTOP_IGNORE_EXISTING: '1',
        HERMES_DESKTOP_HERMES_ROOT: REPO_ROOT,
        HERMES_DESKTOP_APP_NAME: `ProviderOwnerE2E-${Date.now()}`,
        HERMES_DESKTOP_SKIP_QUIT_CONFIRM: '1'
      }
    })
    page = await app.firstWindow()
    installErrorBannerGuard(page)
    watchRpc(page, receipts)
    await expect(page.getByRole('button', { name: 'Open settings', exact: true })).toBeVisible({ timeout: 90_000 })
    await expect(page.locator('[data-slot="statusbar"]').getByText('ready', { exact: true })).toBeVisible({
      timeout: 60_000
    })
    const localConfig = fs.readFileSync(path.join(sandbox.hermesHome, 'config.yaml'), 'utf8')
    const remoteDefault = fs.readFileSync(path.join(remoteHermesHome, 'config.yaml'), 'utf8')

    const localProfiles = () =>
      fs.existsSync(path.join(sandbox.hermesHome, 'profiles'))
        ? fs.readdirSync(path.join(sandbox.hermesHome, 'profiles')).sort()
        : []

    expect(localProfiles()).toEqual([])

    // A: inspect the local default through Settings, without changing it.
    await openProviderKeys(page)
    await page.getByRole('button', { name: 'Close settings', exact: true }).click()
    await selectGateway(page, REMOTE_ID, REMOTE_LABEL, PROFILE)

    for (const model of ['remote-first', 'remote-second']) {
      advertisedModel = model
      await openProviderKeys(page)
      await expect(page.getByText(new RegExp(`Changes on this page apply to.*${PROFILE}`))).toBeVisible()
      await page.getByRole('button', { name: /^Local \/ custom endpoint/ }).click()
      const input = page.getByPlaceholder('http://127.0.0.1:8000/v1')
      await expect(input).toBeVisible()
      await input.fill(`${endpointUrl}/v1`)
      await input.scrollIntoViewIfNeeded()
      await page.screenshot({ path: test.info().outputPath(`${model}-endpoint-form.png`) })
      const start = receipts.length
      await page.getByRole('button', { name: 'Connect', exact: true }).click()
      await expect(input).toBeHidden({ timeout: 45_000 })
      await expect
        .poll(() => configModel(python, remoteEnv, remoteProfileHome))
        .toMatchObject({
          provider: expect.stringMatching(/^custom(?::.+)?$/),
          default: model,
          base_url: `${endpointUrl}/v1`
        })

      const setupCalls = receipts
        .slice(start)
        .filter(row => ['reload.env', 'setup.status', 'setup.runtime_check'].includes(row.method))

      // A scoped setup must not mutate the launch profile's process environment.
      expect(setupCalls.some(row => row.method === 'reload.env')).toBe(false)

      for (const method of ['setup.status', 'setup.runtime_check']) {
        const calls = setupCalls.filter(row => row.method === method)
        expect(calls, `wire request for ${method}`).not.toHaveLength(0)

        for (const call of calls) {
          expect(new URL(call.endpoint).host).toBe(new URL(remote.url).host)
          expect(call.params.profile).toBe(PROFILE)
          expect(call.error).toBeUndefined()
        }
      }

      expect
        .soft(
          setupCalls.find(row => row.method === 'setup.runtime_check')?.result,
          'readiness must resolve the model just saved in the owning profile'
        )
        .toMatchObject({ ok: true, model, profile: PROFILE })
      expect(fs.readFileSync(path.join(sandbox.hermesHome, 'config.yaml'), 'utf8')).toBe(localConfig)
      expect(localProfiles()).toEqual([])
      expect(fs.readFileSync(path.join(remoteHermesHome, 'config.yaml'), 'utf8')).toBe(remoteDefault)
      await page.screenshot({ path: test.info().outputPath(`${model}-saved.png`) })
      await page.getByRole('button', { name: 'Close settings', exact: true }).click()
      await selectGateway(page, 'local', 'This device', 'default')
      await openProviderKeys(page)
      expect(configModel(python, localEnv, sandbox.hermesHome)).toMatchObject({ default: 'local-sentinel' })
      await page.screenshot({ path: test.info().outputPath(`${model}-back-on-local.png`) })
      await page.getByRole('button', { name: 'Close settings', exact: true }).click()

      if (model === 'remote-first') {
        await selectGateway(page, REMOTE_ID, REMOTE_LABEL, PROFILE)
      }
    }

    expect(discoveryRequests).toContain('/v1/models')
    expect(await collectErrorBanners(page)).toEqual([])
  } finally {
    if (page && !page.isClosed()) {
      await test.info().attach('last-ui', { body: await page.locator('body').innerText(), contentType: 'text/plain' })
      await page.screenshot({ path: test.info().outputPath('last-ui.png') })
      await collectErrorBanners(page)
    }

    fs.writeFileSync(test.info().outputPath('setup-wire.json'), JSON.stringify(receipts, null, 2))
    await test
      .info()
      .attach('setup-wire', { path: test.info().outputPath('setup-wire.json'), contentType: 'application/json' })
    await app?.close()

    if (remote) {
      await stopChild(remote.child)
    }

    await new Promise<void>(resolve => endpoint.close(() => resolve()))
    sandbox.cleanup()
  }
})

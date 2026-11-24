/**
 * E2E: a Bot Mode group chat seating bots from two connected gateways.
 *
 * "This device" is the Electron-managed local backend (a multiplexing gateway
 * whose groups.capabilities reports the gateway-owned room driver). "Homelab"
 * is a REAL second `hermes serve` with its own HOME + HERMES_HOME, registered
 * as a remote connection, both answering from one mock inference server.
 *
 * New Group Chat must offer Homelab's bot beside this device's, create the
 * mixed room, and the room must run a turn for EACH member on its own gateway:
 * the user's `@all` send carries an `E2E_SAY(<handle>)[…]` line per member
 * (tests-js/scripts/mock-server.ts::groupScriptedLine), and both scripted
 * replies must land in the persisted room log and render.
 *
 * CROSS_GATEWAY_LEG=main records the picker (the repro: Homelab's bot absent)
 * and stops. CROSS_GATEWAY_SHOTS=<dir> saves screenshots; PROBE_RESULT lines
 * carry the evidence. Needs `npm run build` (dist/) first.
 */

import { type ChildProcess, spawn, spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as net from 'node:net'
import * as path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, type MockBackendFixture, type Sandbox, waitForAppReady } from './fixtures'
import { type ElectronApplication, expect, type Page, test } from './test'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')
const REMOTE_LABEL = 'Homelab'
const REMOTE_ID = 'homelab'
const REMOTE_TOKEN = 'e2e-cross-gateway-homelab-token'
const LEG = process.env.CROSS_GATEWAY_LEG || 'head'
const ROOM = 'Two machines'
const MULTIPLEX = ['gateway:', '  multiplex_profiles: true', ''].join(String.fromCharCode(10))

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as net.AddressInfo
      server.close(() => resolve(port))
    })
  })
}

/** A real second gateway: its own HOME (the host singleton is per HOME) and
 *  HERMES_HOME, this tree's code, the runtime venv's interpreter. */
async function startRemoteGateway(root: string, mockUrl: string) {
  const fakeHome = path.join(root, 'homelab-root')
  const home = path.join(fakeHome, '.hermes')
  fs.mkdirSync(home, { recursive: true })
  writeMockProviderConfig(home, mockUrl, undefined, MULTIPLEX)
  writeEnvFile(home, undefined, mockUrl)
  const port = await freePort()
  const url = `http://127.0.0.1:${port}`
  const python = process.env.CROSS_GATEWAY_PYTHON || path.join(REPO_ROOT, '.venv', 'bin', 'python')

  const child: ChildProcess = spawn(python, ['-m', 'hermes_cli.main', 'serve', '--host', '127.0.0.1', '--port', String(port), '--skip-build'], {
    cwd: REPO_ROOT,
    detached: true,
    env: { ...process.env, HOME: fakeHome, HERMES_HOME: home, PYTHONPATH: REPO_ROOT, HERMES_DASHBOARD_SESSION_TOKEN: REMOTE_TOKEN },
    stdio: ['ignore', 'pipe', 'pipe']
  })

  let log = ''
  child.stdout?.on('data', (chunk: Buffer) => { log += chunk.toString() })
  child.stderr?.on('data', (chunk: Buffer) => { log += chunk.toString() })
  const deadline = Date.now() + 90_000

  while (Date.now() < deadline) {
    if (child.exitCode !== null) {throw new Error(`remote hermes serve exited early (${child.exitCode}):\n${log}`)}

    try {
      if ((await fetch(`${url}/api/status`, { headers: { 'X-Hermes-Session-Token': REMOTE_TOKEN } })).ok) {break}
    } catch {
      // not up yet
    }

    await new Promise(resolve => setTimeout(resolve, 500))
  }

  console.log(`PROBE_RESULT remote serve pid=${child.pid} url=${url} home=${home}`)

  return {
    home,
    url,
    log: () => log,
    close: async () => {
      if (child.pid && child.exitCode === null) {
        try { process.kill(-child.pid, 'SIGTERM') } catch { child.kill('SIGTERM') }
      }

      await new Promise(resolve => setTimeout(resolve, 500))
    }
  }
}

function writeConnectionsRegistry(sandbox: Sandbox, remoteUrl: string): void {
  fs.writeFileSync(path.join(sandbox.userDataDir, 'connections.json'), JSON.stringify({
    version: 2, primary: 'local', launchMode: 'primary', lastUsed: 'local',
    connections: [
      { id: 'local', kind: 'local', label: 'This device' },
      { id: REMOTE_ID, kind: 'remote', label: REMOTE_LABEL, url: remoteUrl, authMode: 'token', token: { encoding: 'plain', value: REMOTE_TOKEN } }
    ]
  }, null, 2), { encoding: 'utf8', mode: 0o600 })
}

async function shot(page: Page, name: string): Promise<void> {
  const dir = process.env.CROSS_GATEWAY_SHOTS

  if (dir) {
    fs.mkdirSync(dir, { recursive: true })
    await page.screenshot({ path: path.join(dir, `${LEG}-${name}.png`) })
  }
}

async function openBots(page: Page): Promise<void> {
  const tab = page.getByRole('button', { name: 'Bots', exact: true }).or(page.getByRole('tab', { name: 'Bots', exact: true })).first()
  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
}

/** Every picker row as { text, checkbox index }. */
async function pickerRows(page: Page) {
  const dialog = page.getByRole('dialog', { name: 'New Group Chat' })

  return dialog.locator('label:has([role="checkbox"])').evaluateAll(nodes => nodes.map(node => (node as HTMLElement).innerText.replace(/\s+/g, ' ').trim()))
}

test.describe('cross-gateway group chat', () => {
  test.describe.configure({ mode: 'serial' })

  let mock: Awaited<ReturnType<typeof startMockServer>>
  let sandbox: Sandbox
  let remote: Awaited<ReturnType<typeof startRemoteGateway>>
  let app: ElectronApplication
  let page: Page

  test.beforeAll(async () => {
    test.setTimeout(300_000)
    mock = await startMockServer()
    sandbox = createSandbox('xgw')
    writeMockProviderConfig(sandbox.hermesHome, mock.url, undefined, MULTIPLEX)
    writeEnvFile(sandbox.hermesHome, undefined, mock.url)
    // A second bot on this device, so main's picker opens (it needs two seatable
    // bots) and shows which machines it offers.
    const scribe = path.join(sandbox.hermesHome, 'profiles', 'scribe')
    writeMockProviderConfig(scribe, mock.url)
    writeEnvFile(scribe, undefined, mock.url)
    remote = await startRemoteGateway(sandbox.root, mock.url)
    writeConnectionsRegistry(sandbox, remote.url)
    ;({ app, page } = await launchDesktop(buildAppEnv(sandbox)))
    await waitForAppReady({ app, page } as MockBackendFixture, 120_000)
    await expect(page.locator('[data-slot="statusbar"]').getByText('ready', { exact: true })).toBeVisible({ timeout: 120_000 })
    await page.waitForTimeout(2_000)
  })

  test.afterAll(async () => {
    const dir = process.env.CROSS_GATEWAY_SHOTS

    if (dir && remote) {fs.writeFileSync(path.join(dir, `${LEG}-remote-serve.log`), remote.log())}

    await app?.close().catch(() => undefined)
    await remote?.close()
    await mock?.close()
    sandbox?.cleanup()
  })

  test('New Group Chat seats a bot from each machine and both answer in the room', async () => {
    test.setTimeout(420_000)
    const capabilities = await page.evaluate(() => (window as any).hermesDesktop?.api?.({ method: 'GET', path: '/api/status' }).catch(() => null))
    console.log('PROBE_RESULT local status', JSON.stringify(capabilities)?.slice(0, 300))

    await openBots(page)
    // Both gateways' primary bots are in the roster before the dialog opens.
    await expect(page.getByText(REMOTE_LABEL).first()).toBeVisible({ timeout: 90_000 })
    await expect(page.getByText(/scribe/i).first()).toBeVisible({ timeout: 90_000 })
    await shot(page, '0-roster')
    await page.getByRole('button', { name: 'New bot or group chat' }).click()
    const menuItem = page.getByRole('menuitem', { name: 'New Group Chat' })
    await expect(menuItem).toBeEnabled({ timeout: 30_000 })
    await menuItem.click()
    const dialog = page.getByRole('dialog', { name: 'New Group Chat' })
    await expect(dialog).toBeVisible()
    // Let groups.capabilities settle so the picker shows its final roster.
    await page.waitForTimeout(3_000)
    const rows = await pickerRows(page)
    console.log('PROBE_RESULT picker rows', JSON.stringify(rows))
    await shot(page, '1-picker')

    const remoteRow = rows.findIndex(text => text.includes(REMOTE_LABEL))
    const localRow = rows.findIndex(text => !text.includes(REMOTE_LABEL))

    if (LEG === 'main') {
      console.log(`PROBE_RESULT main picker offers Homelab bot: ${remoteRow >= 0}`)
      expect(remoteRow).toBe(-1)
      await page.keyboard.press('Escape')

      return
    }

    expect(remoteRow).toBeGreaterThanOrEqual(0)
    expect(localRow).toBeGreaterThanOrEqual(0)
    const checkboxes = dialog.locator('label:has([role="checkbox"]) [role="checkbox"]')
    await checkboxes.nth(localRow).click()
    await checkboxes.nth(remoteRow).click()
    await expect(dialog.locator('[data-group-engine="desktop"]')).toBeVisible()
    console.log('PROBE_RESULT dialog note', await dialog.locator('[data-group-engine="desktop"]').innerText())
    const handles = [rows[localRow], rows[remoteRow]].map(text => /@([a-z0-9][a-z0-9._-]*)/i.exec(text)?.[1] ?? '')
    console.log('PROBE_RESULT handles', JSON.stringify(handles))
    expect(handles.every(Boolean)).toBe(true)
    await dialog.getByRole('textbox', { name: 'Group name' }).fill(ROOM)
    await shot(page, '2-mixed-selection')
    await dialog.getByRole('button', { name: 'Create Group (2)' }).click()
    await expect(dialog).toBeHidden({ timeout: 20_000 })

    const composer = page.getByRole('textbox', { name: `Message ${ROOM}` }).filter({ visible: true })
    await expect(composer).toBeVisible({ timeout: 30_000 })
    await expect(page.getByText('Start gateway group')).toHaveCount(0)
    const send = `@all E2E_SAY(${handles[0]})[LOCAL_OK answered from This device] E2E_SAY(${handles[1]})[REMOTE_OK answered from Homelab]`
    await composer.fill(send)
    await composer.press('Enter')

    const roomLog = () => page.evaluate(room => {
      const rooms = JSON.parse(localStorage.getItem('hermes.plugin.hermes-bots.group-chats') || '{}')

      return (rooms[room]?.log || []).map((entry: any) => ({ from: entry.from?.name, connection: entry.from?.connectionId, text: entry.text }))
    }, ROOM)

    const replies = async () => (await roomLog()).filter((entry: any) => entry.from !== 'You').map((entry: any) => [entry.from, entry.text])
    // Each member's own scripted line, authored by that member, once.
    const dump = async () => {
      console.log('PROBE_RESULT room log', JSON.stringify(await roomLog()))
      console.log('PROBE_RESULT inference speakers', JSON.stringify(mock.receivedPrompts.map(prompt => /You are @[^\s]+/.exec(prompt)?.[0] ?? prompt.slice(0, 80))))
      await shot(page, '3-room-state')
    }

    await expect.poll(async () => (await replies()).map(([, text]: string[]) => text).filter((text: string) => /^(LOCAL|REMOTE)_OK/.test(text)).sort(), { timeout: 120_000, intervals: [2_000] }).toEqual([
      'LOCAL_OK answered from This device',
      'REMOTE_OK answered from Homelab'
    ]).catch(async error => { await dump(); throw error })
    await expect(page.getByText('LOCAL_OK answered from This device', { exact: true }).filter({ visible: true })).toHaveCount(1, { timeout: 30_000 })
    await expect(page.getByText('REMOTE_OK answered from Homelab', { exact: true }).filter({ visible: true })).toHaveCount(1)
    const log = await roomLog()
    console.log('PROBE_RESULT room log', JSON.stringify(log))
    // Each reply was produced on its own machine: read both gateways' state.db.
    const owner = (home: string, marker: string) => spawnSync('python3', ['-c', [
      'import glob, sqlite3, sys',
      'n = 0',
      'for db in glob.glob(sys.argv[1] + "/state.db") + glob.glob(sys.argv[1] + "/profiles/*/state.db"):',
      '    c = sqlite3.connect("file:" + db + "?mode=ro", uri=True)',
      '    n += c.execute("select count(*) from messages where role = \'assistant\' and content like ?", ("%" + sys.argv[2] + "%",)).fetchone()[0]',
      'print(n)'
    ].join(String.fromCharCode(10)), home, marker], { encoding: 'utf8' }).stdout.trim()
    const ownership = {
      thisDevice: { LOCAL_OK: owner(sandbox.hermesHome, 'LOCAL_OK answered'), REMOTE_OK: owner(sandbox.hermesHome, 'REMOTE_OK answered') },
      homelab: { LOCAL_OK: owner(remote.home, 'LOCAL_OK answered'), REMOTE_OK: owner(remote.home, 'REMOTE_OK answered') }
    }
    console.log('PROBE_RESULT assistant rows per gateway state.db', JSON.stringify(ownership))
    expect([Number(ownership.thisDevice.LOCAL_OK) > 0, ownership.thisDevice.REMOTE_OK, Number(ownership.homelab.REMOTE_OK) > 0, ownership.homelab.LOCAL_OK]).toEqual([true, '0', true, '0'])
    const prompts = mock.receivedPrompts.filter(prompt => prompt.includes('E2E_SAY('))
    console.log('PROBE_RESULT member prompts seen by inference', prompts.length)
    await page.waitForTimeout(1_500)
    await shot(page, '3-both-replies')
    // The round settles (later rounds pass) without a member error.
    await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0, { timeout: 120_000 })
    console.log('PROBE_RESULT member errors', JSON.stringify(await page.getByText(/hit an error/).allInnerTexts()))
    await shot(page, '4-settled')
    await expect(page.getByText(/hit an error/)).toHaveCount(0)
    console.log('PROBE_RESULT settled room log', JSON.stringify(await roomLog()))
    await shot(page, '4-settled')
  })

  test('control: two bots on this device still make a gateway-owned room', async () => {
    test.setTimeout(180_000)
    await openBots(page)
    await page.getByRole('button', { name: 'New bot or group chat' }).click()
    await page.getByRole('menuitem', { name: 'New Group Chat' }).click()
    const dialog = page.getByRole('dialog', { name: 'New Group Chat' })
    await expect(dialog).toBeVisible()
    await page.waitForTimeout(3_000)
    const rows = await pickerRows(page)
    const checkboxes = dialog.locator('label:has([role="checkbox"]) [role="checkbox"]')

    for (const index of rows.flatMap((text, i) => (text.includes(REMOTE_LABEL) ? [] : [i]))) {
      await checkboxes.nth(index).click()
    }

    await expect(dialog.locator('[data-group-engine]')).toHaveCount(0)
    await dialog.getByRole('textbox', { name: 'Group name' }).fill('This device only')
    await dialog.getByRole('button', { name: 'Create Group (2)' }).click()
    await expect(dialog).toBeHidden({ timeout: 20_000 })
    // The gateway-owned workspace's composer, not the Desktop-driven one.
    await expect(page.getByRole('textbox', { name: 'Group message' }).filter({ visible: true })).toBeVisible({ timeout: 30_000 })
    console.log('PROBE_RESULT control room opened in the gateway-owned workspace')
    await shot(page, '5-control-gateway-owned')
  })
})

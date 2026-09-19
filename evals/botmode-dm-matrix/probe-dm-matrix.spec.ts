import { execFileSync, spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { buildAppEnv, createSandbox, launchDesktop, waitForAppReady, writeEnvFile, writeMockProviderConfig, type MockBackendFixture } from './fixtures'
import { MOCK_REPLY, startMockServer } from '../../../tests-js/scripts/mock-server'
import { expect, test } from './test'

const repo = path.resolve(import.meta.dirname, '../../..')
let python = path.join(process.env.VIRTUAL_ENV || path.join(repo, '.venv'), 'bin', 'python')
let fixture: MockBackendFixture
let env: Record<string, string>
const evidence = process.env.BOT_DM_EVIDENCE || '/tmp/botmode-dm-matrix/native'

test.beforeAll(async () => {
  fs.mkdirSync(evidence, { recursive: true })
  const sandbox = createSandbox('dm-delivery')
  const mock = await startMockServer({ holdFirstCompletionContaining: 'Message from 🤖 beta (@beta): matrix-unowned-sentinel' })
  for (const name of ['default', 'alpha', 'beta', 'gamma']) {
    const home = name === 'default' ? sandbox.hermesHome : path.join(sandbox.hermesHome, 'profiles', name)
    fs.mkdirSync(home, { recursive: true })
    writeMockProviderConfig(home, mock.url)
    writeEnvFile(home)
    fs.writeFileSync(path.join(home, 'SOUL.md'), `# ${name}\nA Bot Mode teammate.\n`)
    fs.writeFileSync(path.join(home, 'profile.yaml'), 'name: ' + name + '\nui_meta:\n  hermes-bots: {}\n')
  }
  if (process.env.BOT_DM_SERVICE_PATH === '1') {
    const sourceVenv = path.dirname(path.dirname(python))
    const runtime = path.join(sandbox.root, 'runtime')
    fs.mkdirSync(path.join(runtime, 'bin'), { recursive: true })
    fs.copyFileSync(path.join(sourceVenv, 'pyvenv.cfg'), path.join(runtime, 'pyvenv.cfg'))
    fs.symlinkSync(path.join(sourceVenv, 'lib'), path.join(runtime, 'lib'))
    fs.symlinkSync(python, path.join(runtime, 'bin', 'python'))
    python = path.join(runtime, 'bin', 'python')
    fs.writeFileSync(path.join(runtime, 'bin', 'hermes'), `#!/bin/sh\ncd ${repo}\nexec ${python} -m hermes_cli.main "$@"\n`, { mode: 0o755 })
  }
  const bin = path.join(sandbox.root, 'bin')
  fs.mkdirSync(bin)
  fs.writeFileSync(path.join(bin, 'hermes'), `#!/bin/sh\ncd ${repo}\nexec ${python} -m hermes_cli.main "$@"\n`, { mode: 0o755 })
  if (process.env.BOT_DM_SERVICE_PATH === '1') {
    fs.writeFileSync(path.join(bin, 'hermes'), `#!/bin/sh\nprintf 'WRONG_PATH_HERMES invoked: %s\\n' "$*" >> ${path.join(evidence, 'wrong-path.log')}\nprintf 'old launcher rejects --query-file\\n' >&2\nexit 2\n`, { mode: 0o755 })
  }
  env = buildAppEnv(sandbox, { HOME: sandbox.root, HERMES_DESKTOP_PYTHON: python,
    HERMES_DESKTOP_HERMES: path.join(bin, 'hermes'), PATH: `${bin}:${process.env.PATH}`,
    PYTHONPATH: repo, HERMES_SINGLE_QUERY_LINGER_SECONDS: '30' })
  for (const name of ['alpha', 'beta', 'gamma']) {
    const h = path.join(sandbox.hermesHome, 'profiles', name)
    execFileSync(python, ['-c', 'import sys; from pathlib import Path; from hermes_state import SessionDB; d=SessionDB(db_path=Path(sys.argv[1])/"state.db"); d.create_session("matrix-"+sys.argv[2],"cli",cwd=sys.argv[3]); d.set_session_title("matrix-"+sys.argv[2],"Bot Chat"); d.close()', h, name, sandbox.root], { env, cwd: repo })
  }
  const { app, page } = await launchDesktop(env)
  fixture = { app, page, sandbox, mock, mockUrl: mock.url, cleanup: async () => {
    await app.close().catch(() => undefined)
    await mock.close()
  } }
  console.log('SANDBOX', sandbox.root)
  fs.writeFileSync(path.join(evidence, 'sandbox.txt'), sandbox.root)
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => { await fixture?.cleanup() })

async function openBot(name: string) {
  const page = fixture.page
  await page.getByRole('button', { name: 'Bots', exact: true }).or(page.getByRole('tab', { name: 'Bots', exact: true })).first().click()
  const row = page.getByRole('button', { name: new RegExp(`^${name === 'default' ? 'hermes' : name}\\b`, 'i') }).filter({ visible: true }).first()
  await expect(row).toBeVisible({ timeout: 30_000 })
  await row.click()
  const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()
  await expect(composer).toBeVisible({ timeout: 120_000 })
  return composer
}

function dbMessages(name: string) {
  const home = name === 'default' ? fixture.sandbox.hermesHome : path.join(fixture.sandbox.hermesHome, 'profiles', name)
  return JSON.parse(execFileSync(python, ['-c', 'import sqlite3,json,sys; c=sqlite3.connect(sys.argv[1]); print(json.dumps(c.execute("select role,content from messages").fetchall()))', path.join(home, 'state.db')], { env, cwd: repo, encoding: 'utf8' })) as string[][]
}


function py(code: string, args: string[] = []) {
  return JSON.parse(execFileSync(python, ['-c', code, ...args], { env, cwd: repo, encoding: 'utf8' }))
}
function owner(name: string) {
  return py('import json,sys; from tools.bot_live_delivery import find_canonical_live_owner; print(json.dumps(find_canonical_live_owner(sys.argv[1])))', [home(name)])
}
function home(name: string) {
  return name === 'default' ? fixture.sandbox.hermesHome : path.join(fixture.sandbox.hermesHome, 'profiles', name)
}
function snapshot(name: string) {
  return py('import json,sqlite3,sys; from hermes_cli.active_sessions import active_session_registry_snapshot; c=sqlite3.connect(sys.argv[1]+"/state.db"); c.row_factory=sqlite3.Row; print(json.dumps(dict(sessions=[dict(x) for x in c.execute("select * from sessions")],messages=[dict(x) for x in c.execute("select * from messages")],owners=active_session_registry_snapshot(registry_home=sys.argv[1])),default=str))', [home(name)])
}
async function capture(label: string) {
  fs.writeFileSync(path.join(evidence, `${label}.json`), JSON.stringify({ default: snapshot('default'), alpha: snapshot('alpha'), beta: snapshot('beta'), gamma: snapshot('gamma'), processes: Object.fromEntries(['default', 'alpha', 'beta', 'gamma'].map(n => { const file = path.join(home(n), 'processes.json'); return [n, fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null] })), prompts: fixture.mock.receivedPrompts, dom: await fixture.page.locator('body').innerText() }, null, 2))
  await fixture.page.screenshot({ path: path.join(evidence, `${label}.png`) })
}
async function unfold() {
  for (const toggle of await fixture.page.getByText('show message', { exact: true }).all()) {
    if (await toggle.isVisible()) await toggle.click()
  }
}

test('default live owner does not capture named unowned nested quiet CLI deliveries', async () => {
  test.setTimeout(300_000)
  const page = fixture.page
  const composer = await openBot('default')
  await expect(page.getByText('Say something to get started.').filter({ visible: true })).toBeVisible({ timeout: 120_000 })
  await composer.fill('initialize default owner')
  await page.keyboard.press('Enter')
  await expect(page.getByText(MOCK_REPLY).filter({ visible: true }).first()).toBeVisible({ timeout: 60_000 })
  const defaultOwner = owner('default')
  expect(defaultOwner).not.toBeNull()
  expect(owner('beta')).toBeNull()
  expect(owner('gamma')).toBeNull()
  // A Desktop turn executes the real message_agent, spawning quiet CLI B,
  // which executes the real message_agent again into unowned C.
  await composer.fill('E2E_DM(beta)[E2E_DM(gamma)[matrix-unowned-sentinel]]')
  await page.keyboard.press('Enter')
  try {
    await expect.poll(() => dbMessages('beta').filter(([r,t]) => r === 'user' && t.includes('matrix-unowned-sentinel')).length, { timeout: 35_000 }).toBe(1)
    await fixture.mock.waitForHeldCompletion()
    await expect.poll(() => dbMessages('beta').filter(([r]) => r === 'assistant').length, { timeout: 60_000 }).toBeGreaterThanOrEqual(2)
    expect(owner('default')).toEqual(defaultOwner)
    await capture('nested-held')
    const held = JSON.parse(fs.readFileSync(path.join(evidence, 'nested-held.json'), 'utf8'))
    const nestedProcess = held.processes.beta[0]
    expect(nestedProcess.session_key).toBe('matrix-beta')
    expect(nestedProcess.parent_session_id).toBe('matrix-beta')
    expect(nestedProcess.owner_task_id).toBeTruthy()
    expect(nestedProcess.owner_task_id).not.toBe(defaultOwner.session_id)
    expect(held.beta.owners[0].surface).toBe('cli')
    expect(held.gamma.owners[0].surface).toBe('cli')
    expect(held.beta.owners[0].pid).not.toBe(held.gamma.owners[0].pid)
    expect(held.beta.sessions.filter((s: any) => s.title === 'Bot Chat').map((s: any) => s.id)).toEqual(['matrix-beta'])
    expect(held.gamma.sessions.filter((s: any) => s.title === 'Bot Chat').map((s: any) => s.id)).toEqual(['matrix-gamma'])
    fixture.mock.releaseHeldStream()
    await expect.poll(() => dbMessages('beta').filter(([r,t]) => r === 'user' && t.includes('completed')).length, { timeout: 100_000 }).toBeGreaterThan(0)
    await expect.poll(() => dbMessages('default').filter(([r,t]) => r === 'user' && t.includes('completed')).length, { timeout: 100_000 }).toBeGreaterThan(0)
    expect(dbMessages('gamma').filter(([r,t]) => r === 'user' && t.includes('matrix-unowned-sentinel'))).toHaveLength(1)
    expect(dbMessages('default').filter(([r,t]) => r === 'user' && t.startsWith('Message from'))).toHaveLength(0)
    expect(owner('default')).toEqual(defaultOwner)
    await openBot('gamma')
    await unfold()
    await expect(page.getByText(/matrix-unowned-sentinel/).filter({ visible: true }).first()).toBeVisible({ timeout: 60_000 })
    await capture('nested-complete')
    expect(dbMessages('beta').filter(([r,t]) => r === 'user' && t.startsWith('[IMPORTANT: Background process') && t.includes(nestedProcess.session_id))).toHaveLength(1)
    expect(dbMessages('gamma').filter(([r,t]) => r === 'user' && t.startsWith('[IMPORTANT: Background process'))).toHaveLength(0)
    expect(fs.existsSync(path.join(evidence, 'wrong-path.log'))).toBe(false)
  } finally { fixture.mock.releaseHeldStream(); await capture('nested-final') }
})

test('incoming live sender card is rendered before any reload', async () => {
  test.setTimeout(240_000)
  const page = fixture.page
  const composer = await openBot('alpha')
  await expect(page.getByText('Say something to get started.').filter({ visible: true })).toBeVisible({ timeout: 120_000 })
  await composer.fill('initialize alpha owner')
  await page.keyboard.press('Enter')
  await expect(page.getByText(MOCK_REPLY).filter({ visible: true }).first()).toBeVisible({ timeout: 60_000 })
  expect(owner('alpha')).not.toBeNull()
  const output = fs.openSync(path.join(evidence, 'live-sender.log'), 'w')
  const child = spawn(python, ['-m', 'hermes_cli.main', '-p', 'beta', 'chat', '--in', '~', '-c', 'Bot Chat', '--create-if-missing', '-Q', '-q', 'E2E_DM(alpha)[matrix-live-sender-card]'], { cwd: repo, env, stdio: ['ignore', output, output] })
  try {
    await expect.poll(() => dbMessages('alpha').filter(([r,t]) => r === 'user' && t.includes('matrix-live-sender-card')).length, { timeout: 100_000 }).toBe(1)
    await expect.poll(() => child.exitCode, { timeout: 100_000 }).toBe(0)
    await capture('live-before-unfold')
    await expect(page.getByText('Message from beta', { exact: true }).filter({ visible: true }).first()).toBeVisible({ timeout: 30_000 })
    await expect(page.getByText('show message', { exact: true }).filter({ visible: true }).first()).toBeVisible({ timeout: 30_000 })
    await unfold()
    await capture('live-after-unfold')
    await expect(page.getByText(/matrix-live-sender-card/).filter({ visible: true }).first()).toBeVisible({ timeout: 30_000 })
  } finally { child.kill(); fs.closeSync(output); await capture('live-final') }
})

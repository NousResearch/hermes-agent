import { execFileSync, spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { buildAppEnv, createSandbox, launchDesktop, waitForAppReady, writeEnvFile, writeMockProviderConfig, type MockBackendFixture } from './fixtures'
import { MOCK_REPLY, startMockServer } from '../../../tests-js/scripts/mock-server'
import { expect, test } from './test'

const repo = path.resolve(import.meta.dirname, '../../..')
const python = path.join(process.env.VIRTUAL_ENV || path.join(repo, '.venv'), 'bin', 'python')
let fixture: MockBackendFixture
let env: Record<string, string>
const evidence = '/tmp/botmode-dm-recovery'

test.beforeAll(async () => {
  fs.mkdirSync(evidence, { recursive: true })
  const sandbox = createSandbox('dm-delivery')
  const mock = await startMockServer({ holdFirstCompletionContaining: 'CLI_OWNER_HOLD' })
  for (const name of ['default', 'alpha', 'beta']) {
    const home = name === 'default' ? sandbox.hermesHome : path.join(sandbox.hermesHome, 'profiles', name)
    fs.mkdirSync(home, { recursive: true })
    writeMockProviderConfig(home, mock.url)
    writeEnvFile(home)
    fs.writeFileSync(path.join(home, 'SOUL.md'), `# ${name}\nA Bot Mode teammate.\n`)
    fs.writeFileSync(path.join(home, 'profile.yaml'), 'name: ' + name + '\nui_meta:\n  hermes-bots: {}\n')
  }
  const bin = path.join(sandbox.root, 'bin')
  fs.mkdirSync(bin)
  fs.writeFileSync(path.join(bin, 'hermes'), `#!/bin/sh\ncd ${repo}\nexec ${python} -m hermes_cli.main "$@"\n`, { mode: 0o755 })
  env = buildAppEnv(sandbox, { HOME: sandbox.root, HERMES_DESKTOP_PYTHON: python,
    HERMES_DESKTOP_HERMES: path.join(bin, 'hermes'), PATH: `${bin}:${process.env.PATH}`,
    PYTHONPATH: repo, HERMES_SINGLE_QUERY_LINGER_SECONDS: '30' })
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

test('cron output waits for a CLI-only owner and arrives after owner release', async () => {
  test.setTimeout(240_000)
  const output = fs.openSync(path.join(evidence, 'cli-owner.log'), 'w')
  const child = spawn(python, ['-m', 'hermes_cli.main', '-p', 'beta', 'chat', '--in', '~', '-c', 'Bot Chat', '--create-if-missing', '-Q', '-q', 'CLI_OWNER_HOLD'], { cwd: repo, env, stdio: ['ignore', output, output] })
  const cronEnv = { ...env, HERMES_HOME: path.join(fixture.sandbox.hermesHome, 'profiles', 'beta') }
  try {
    await fixture.mock.waitForHeldCompletion()
    const script = 'import json; from cron.scheduler_delivery import _deliver_to_bot_chat; j={"id":"cli-residual","name":"CLI residual","execution_id":"fixed-execution"}; result=_deliver_to_bot_chat(j,"CLI_OWNER_CRON_SENTINEL",""); print(json.dumps({"result":result,"job":j}))'
    const result = JSON.parse(execFileSync(python, ['-c', script], { env: cronEnv, cwd: repo, encoding: 'utf8', timeout: 30_000 }))
    console.log('CLI_OWNER_CRON_ADMISSION', JSON.stringify(result))
    fs.writeFileSync(path.join(evidence, 'cli-owner-admission.json'), JSON.stringify(result, null, 2))
    fixture.mock.releaseHeldStream()
    await expect.poll(() => child.exitCode, { timeout: 60_000 }).toBe(0)
    const ticker = spawn(python, ['-c', 'import time; from cron.scheduler import tick; from cron.bot_chat_delivery import _running; tick(verbose=False);\nwhile _running: time.sleep(0.1)'], { env: cronEnv, cwd: repo, stdio: ['ignore', output, output] })
    await expect.poll(() => ticker.exitCode, { timeout: 90_000 }).toBe(0)
    await openBot('beta')
    expect(dbMessages('beta').filter(([role, text]) => role === 'user' && text.includes('CLI_OWNER_CRON_SENTINEL'))).toHaveLength(1)
    await expect(fixture.page.getByText(/CLI_OWNER_CRON_SENTINEL/).filter({ visible: true }).first()).toBeVisible({ timeout: 45_000 })
    await fixture.page.screenshot({ path: path.join(evidence, 'cli-owner-after-release.png') })
  } finally { fixture.mock.releaseHeldStream(); child.kill(); fs.closeSync(output) }
})

async function openBot(name: string) {
  const page = fixture.page
  await page.getByRole('button', { name: 'Bots', exact: true }).or(page.getByRole('tab', { name: 'Bots', exact: true })).first().click()
  const row = page.getByRole('button', { name: new RegExp(`^${name}\\b`, 'i') }).filter({ visible: true }).first()
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

test('named Bot Chat receives a nested one-shot message_agent delivery once', async () => {
  test.setTimeout(300_000)
  const page = fixture.page
  const composer = await openBot('alpha')
  await expect(page.getByText('Say something to get started.').filter({ visible: true })).toBeVisible({ timeout: 120_000 })
  await composer.fill('initialize alpha owner')
  await page.keyboard.press('Enter')
  await expect(page.getByText(MOCK_REPLY).filter({ visible: true }).first()).toBeVisible({ timeout: 60_000 })
  const output = fs.openSync(path.join(evidence, 'oneshot.log'), 'w')
  const child = spawn(python, ['-m', 'hermes_cli.main', '-p', 'beta', 'chat', '--in', '~', '-c', 'Bot Chat', '--create-if-missing', '-Q', '-q', 'E2E_DM(alpha)[nested-one-shot-sentinel]'], { cwd: repo, env, stdio: ['ignore', output, output] })
  try {
    await expect.poll(() => dbMessages('alpha').filter(([role, text]) => role === 'user' && text.includes('nested-one-shot-sentinel')).length, { timeout: 120_000 }).toBe(1)
    console.log('ALPHA_ROWS', JSON.stringify(dbMessages('alpha')))
    console.log('BETA_ROWS', JSON.stringify(dbMessages('beta')))
    await page.reload()
    await waitForAppReady(fixture, 120_000)
    await openBot('alpha')
    await expect(page.getByText('show message', { exact: true }).first()).toBeVisible({ timeout: 30_000 })
    for (const toggle of await page.getByText('show message', { exact: true }).all()) await toggle.click()
    await expect(page.getByText(/nested-one-shot-sentinel/).filter({ visible: true }).first()).toBeVisible({ timeout: 30_000 })
    await page.screenshot({ path: path.join(evidence, 'nested-delivery-desktop.png') })
    fs.writeFileSync(path.join(evidence, 'rows.json'), JSON.stringify({ alpha: dbMessages('alpha'), beta: dbMessages('beta'), prompts: fixture.mock.receivedPrompts }, null, 2))
  } finally { child.kill(); fs.closeSync(output) }
})

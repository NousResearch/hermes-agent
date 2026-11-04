import { execFileSync, spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { buildAppEnv, createSandbox, launchDesktop, waitForAppReady, writeEnvFile, writeMockProviderConfig, type MockBackendFixture } from './fixtures'
import { MOCK_REPLY, startMockServer } from '../../../tests-js/scripts/mock-server'
import { expect, test } from './test'

const repo = path.resolve(import.meta.dirname, '../../..')
const python = path.join(process.env.VIRTUAL_ENV || path.join(repo, '.venv'), 'bin', 'python')
const evidence = process.env.BOT_DM_EVIDENCE || path.join(os.tmpdir(), 'botmode-cron-root/native')
let fixture: MockBackendFixture
let env: Record<string, string>

test.beforeAll(async () => {
  fs.mkdirSync(evidence, { recursive: true })
  const sandbox = createSandbox('cron-root')
  const mock = await startMockServer({ holdFirstCompletionContaining: 'ORDINARY_CRON_SENTINEL' })
  for (const name of ['default', 'alpha', 'beta']) {
    const home = name === 'default' ? sandbox.hermesHome : path.join(sandbox.hermesHome, 'profiles', name)
    fs.mkdirSync(home, { recursive: true })
    writeMockProviderConfig(home, mock.url)
    writeEnvFile(home)
    fs.writeFileSync(path.join(home, 'SOUL.md'), `# ${name}\nA Bot Mode teammate.\n`)
    fs.writeFileSync(path.join(home, 'profile.yaml'), `name: ${name}\nui_meta:\n  hermes-bots: {}\n`)
  }
  const bin = path.join(sandbox.root, 'bin')
  fs.mkdirSync(bin)
  fs.writeFileSync(path.join(bin, 'hermes'), `#!/bin/sh\ncd ${repo}\nprintf '%s\\n' "$$ $HERMES_HOME $*" >> ${evidence}/children.log\nexec ${python} -m hermes_cli.main "$@"\n`, { mode: 0o755 })
  env = buildAppEnv(sandbox, { HOME: sandbox.root, HERMES_DESKTOP_PYTHON: python,
    HERMES_DESKTOP_HERMES: path.join(bin, 'hermes'), PATH: `${bin}:${process.env.PATH}`,
    PYTHONPATH: repo, HERMES_SINGLE_QUERY_LINGER_SECONDS: '1' })
  const { app, page } = await launchDesktop(env)
  fixture = { app, page, sandbox, mock, mockUrl: mock.url, cleanup: async () => {
    await app.close().catch(() => undefined)
    await mock.close()
  } }
  fs.writeFileSync(path.join(evidence, 'sandbox.txt'), sandbox.root)
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => { await fixture?.cleanup() })

function probe(script: string, extraEnv = {}) {
  return JSON.parse(execFileSync(python, ['-c', script], { env: { ...env, ...extraEnv }, cwd: repo, encoding: 'utf8', timeout: 30_000 }))
}

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

test('ordinary cron pins an unowned named target under a custom root', async () => {
  test.setTimeout(240_000)
  const page = fixture.page
  const composer = await openBot('Hermes')
  await expect(page.getByText('Say something to get started.').filter({ visible: true })).toBeVisible({ timeout: 120_000 })
  await composer.fill('initialize default Desktop owner')
  await page.keyboard.press('Enter')
  await expect(page.getByText(MOCK_REPLY).filter({ visible: true }).first()).toBeVisible({ timeout: 60_000 })
  const discovery = 'from pathlib import Path; import json,os; from tools.bot_live_delivery import find_canonical_live_owner; h=Path(os.environ["HERMES_HOME"]); print(json.dumps({"default":find_canonical_live_owner(h),"alpha":find_canonical_live_owner(h/"profiles"/"alpha")}))'
  const before = probe(discovery)
  expect(before.default.surface).toBe('desktop')
  expect(before.alpha).toBeNull()
  const output = fs.openSync(path.join(evidence, 'producer.log'), 'w')
  const resultPath = path.join(evidence, 'result.json')
  const script = `import json; from pathlib import Path; from cron.scheduler_delivery import _deliver_to_bot_chat; j={"id":"ordinary-cron","name":"Ordinary cron","execution_id":"never-deferred"}; result=_deliver_to_bot_chat(j,"ORDINARY_CRON_SENTINEL","alpha"); Path(${JSON.stringify(resultPath)}).write_text(json.dumps({"result":result,"job":j}))`
  const child = spawn(python, ['-c', script], { env, cwd: repo, stdio: ['ignore', output, output] })
  try {
    await expect.poll(() => fs.existsSync(resultPath) ? 'exited' : fixture.mock.receivedPrompts.some(p => p.includes('ORDINARY_CRON_SENTINEL')) ? 'held' : 'waiting', { timeout: 90_000 }).not.toBe('waiting')
    if (fs.existsSync(resultPath)) {
      console.log('EARLY_RESULT', fs.readFileSync(resultPath, 'utf8'))
      expect(JSON.parse(fs.readFileSync(resultPath, 'utf8')).result).toBeNull()
    }
    await fixture.mock.waitForHeldCompletion()
    const held = probe(discovery)
    expect(held.alpha.surface).toBe('cli')
    const launched = fs.readFileSync(path.join(evidence, 'children.log'), 'utf8').trim().split('\n').at(-1)!
    expect(Number(launched.split(' ')[0])).toBe(held.alpha.pid)
    expect(launched).toContain(path.join(fixture.sandbox.hermesHome, 'profiles', 'alpha'))
    expect(held.default.lease_id).toBe(before.default.lease_id)
    expect(fs.existsSync(path.join(fixture.sandbox.hermesHome, 'cron', 'bot_chat_pending'))).toBe(false)
    fs.writeFileSync(path.join(evidence, 'owners.json'), JSON.stringify({ before, held }, null, 2))
    fixture.mock.releaseHeldStream()
    await expect.poll(() => child.exitCode, { timeout: 60_000 }).toBe(0)
    expect(JSON.parse(fs.readFileSync(resultPath, 'utf8')).result).toBeNull()
    await openBot('alpha')
    await expect(page.getByText(/ORDINARY_CRON_SENTINEL/).filter({ visible: true }).first()).toBeVisible({ timeout: 60_000 })
    const rows = probe('import sqlite3,json,os; from pathlib import Path; h=Path(os.environ["HERMES_HOME"]); print(json.dumps({n:sqlite3.connect(h/"state.db" if n=="default" else h/"profiles"/n/"state.db").execute("select session_id,role,content from messages").fetchall() for n in ["default","alpha"]}))')
    expect(rows.alpha.filter((r: string[]) => r[1] === 'user' && r[2].includes('ORDINARY_CRON_SENTINEL'))).toHaveLength(1)
    expect(rows.default.filter((r: string[]) => r[2].includes('ORDINARY_CRON_SENTINEL'))).toHaveLength(0)
    fs.writeFileSync(path.join(evidence, 'rows.json'), JSON.stringify(rows, null, 2))
    await page.screenshot({ path: path.join(evidence, 'ordinary-recipient.png') })
    const missing = probe('import json,os; from pathlib import Path; from cron.scheduler_delivery import _deliver_to_bot_chat; h=Path(os.environ["HERMES_HOME"]); p=h/"profiles"/"beta"; p.rename(h/"profiles"/"beta-removed"); result=_deliver_to_bot_chat({"id":"removed","execution_id":"missing"},"MUST_NOT_RUN","beta"); print(json.dumps({"result":result,"recreated":p.exists(),"wrong_root":(Path.home()/".hermes").exists()}))')
    expect(missing.result).not.toBeNull()
    expect(missing.recreated).toBe(false)
    expect(missing.wrong_root).toBe(false)
    fs.writeFileSync(path.join(evidence, 'missing.json'), JSON.stringify(missing, null, 2))
  } finally { fixture.mock.releaseHeldStream(); child.kill(); fs.closeSync(output) }
})

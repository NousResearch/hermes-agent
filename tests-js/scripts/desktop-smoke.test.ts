import { spawn, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import yaml from 'js-yaml'
import { expect, test } from 'vitest'

import { candidateSmokeHermesHomes, resolveSmokeLaunch, runInstalledDesktopSmoke, smokeEnvironment } from '../../tests/install/e2e-assets/desktop-smoke.ts'

import { assertChatCommit, newCompletedPair, readMockPrompts, type TranscriptMessage } from './desktop-chat-smoke.ts'
import { assertBackendOrigin, localBackendProcess, readInstallationCommit } from './desktop-smoke-process.ts'
import { writeEnvFile, writeMockProviderConfig } from './mock-provider-config.ts'
import { MOCK_REPLY, startMockServer } from './mock-server.ts'

test('one server owns inference and a fresh, per-server prompt witness', async (): Promise<void> => {
  const first = await startMockServer()
  const second = await startMockServer()
  try {
    expect(await readMockPrompts(first.url)).toEqual([])
    const prompt = 'unique request from the OLD checkpoint'
    const response = await fetch(`${first.url}/v1/chat/completions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'mock-model', messages: [{ role: 'user', content: prompt }] }),
    })
    expect(await response.json()).toMatchObject({ choices: [{ message: { content: MOCK_REPLY } }] })
    expect(await readMockPrompts(first.url)).toEqual(first.receivedPrompts)
    expect(first.receivedPrompts).toEqual([prompt])
    await fetch(`${first.url}/v1/chat/completions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: [{ type: 'text', text: 'NEW multipart prompt' }] }] }),
    })
    expect(await readMockPrompts(first.url)).toEqual([prompt, 'NEW multipart prompt'])
    expect(await readMockPrompts(second.url)).toEqual([])
  } finally {
    await first.close()
    await second.close()
  }
  await expect(readMockPrompts(first.url)).rejects.toThrow()
})

test('provider reconfiguration preserves feed, plugins, history and explicit fixture options', (): void => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-config-'))
  try {
    const feed = { desktop_feed_base_url: 'http://127.0.0.1:8123/feed' }
    fs.writeFileSync(path.join(home, 'config.yaml'), yaml.dump({ updates: feed, plugins: { witness: true }, model: { temperature: 0.7 } }))
    fs.writeFileSync(path.join(home, 'state.db'), 'history witness')
    fs.writeFileSync(path.join(home, '.env'), 'OTHER_TEST_VALUE=kept\nMOCK_API_KEY=old\n')
    writeMockProviderConfig(home, 'http://127.0.0.1:9000')
    writeEnvFile(home)
    writeMockProviderConfig(home, 'http://127.0.0.1:9001', 'interim_assistant_messages: true', 'approvals:\n  mode: smart', 12000)
    writeEnvFile(home)
    expect(yaml.load(fs.readFileSync(path.join(home, 'config.yaml'), 'utf8'))).toMatchObject({
      updates: feed, plugins: { witness: true },
      model: { provider: 'mock', temperature: 0.7, context_length: 12000 },
      providers: { mock: { api: 'http://127.0.0.1:9001/v1', context_length: 64000 } },
      auxiliary: { title_generation: { enabled: false } }, approvals: { mode: 'smart' }, display: { interim_assistant_messages: true },
    })
    expect(fs.readFileSync(path.join(home, '.env'), 'utf8')).toBe('OTHER_TEST_VALUE=kept\nMOCK_API_KEY=e2e-mock-key\n')
    expect(fs.readFileSync(path.join(home, 'state.db'), 'utf8')).toBe('history witness')
  } finally { fs.rmSync(home, { recursive: true, force: true }) }
})

test('OLD history, partial/wrong replies, active streams and errors cannot satisfy NEW', (): void => {
  const message = (id: string, role: string, text: string): TranscriptMessage => ({ id, role, text, streaming: false, error: false })
  const old = [message('old-user', 'user', 'old'), message('old-reply', 'assistant', MOCK_REPLY)]
  const before = old.map((row: TranscriptMessage): string => row.id)
  const user = message('new-user', 'user', 'new nonce')
  const reply = message('new-reply', 'assistant', MOCK_REPLY)
  for (const rows of [old, [...old, user], [...old, user, { ...reply, id: 'old-reply' }],
    [...old, user, { ...reply, text: 'boot chain is working' }], [...old, user, { ...reply, text: 'wrong' }],
    [...old, user, { ...reply, streaming: true }], [...old, user, { ...reply, error: true }],
    [...old, user, message('other-user', 'user', 'unrelated'), reply]]) {
    expect(newCompletedPair(rows, before, 'new nonce')).toBeNull()
  }
  expect(newCompletedPair([...old, user, reply], before, 'new nonce')).toEqual({ user, assistant: reply })
})

test.runIf(process.platform !== 'win32')('shell wrapper exports the live witness URL without losing journey config', (): void => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-wrapper-'))
  const assets = path.resolve(import.meta.dirname, '../../tests/install/e2e-assets')
  try {
    fs.writeFileSync(path.join(home, 'config.yaml'), 'updates:\n  desktop_feed_base_url: http://127.0.0.1:1234/feed\n')
    const result = spawnSync('bash', ['-c', `
      set -eu
      ok() { printf '%s\\n' "$*"; }
      fail() { printf '%s\\n' "$*" >&2; exit 1; }
      log_group() { :; }
      source "$ASSETS/mock-provider.sh"
      trap mock_stop EXIT
      mock_start "$HERMES_HOME"
      node --input-type=module -e 'const r=await fetch(process.env.HERMES_E2E_MOCK_URL+"/__e2e__/prompts"); if(!r.ok || (await r.json()).receivedPrompts.length!==0)process.exit(1)'
    `], { env: { ...process.env, ASSETS: assets, HERMES_HOME: home, LOG_DIR: home }, encoding: 'utf8', timeout: 20_000 })
    expect(result.status, result.stdout + result.stderr).toBe(0)
    expect(yaml.load(fs.readFileSync(path.join(home, 'config.yaml'), 'utf8'))).toMatchObject({ updates: { desktop_feed_base_url: 'http://127.0.0.1:1234/feed' } })
  } finally { fs.rmSync(home, { recursive: true, force: true }) }
})

test.runIf(process.platform === 'linux')('origin proof finds the live child listener and rejects a package impostor', async (): Promise<void> => {
  const child = spawn(process.execPath, ['-e', 'const s=require("node:net").createServer();s.listen(0,"127.0.0.1",()=>console.log(s.address().port))'], { stdio: ['ignore', 'pipe', 'pipe'] })
  try {
    const port = await new Promise<number>((resolve, reject): void => {
      child.once('error', reject)
      child.stdout.once('data', (data: Buffer): void => resolve(Number(data.toString().trim())))
    })
    const backend = localBackendProcess(port, process.pid)
    expect(backend.pid).toBe(child.pid)
    expect(backend.executable).toBe(fs.realpathSync(process.execPath))
    expect((): void => assertBackendOrigin(backend, os.tmpdir(), 'bundled')).toThrow('agent-payload')
    expect((): void => assertBackendOrigin(backend, os.tmpdir(), 'source')).toThrow('source tree')
    expect((): void => { localBackendProcess(port, child.pid!) }).toThrow('found 0')
  } finally { child.kill(); await new Promise<void>((resolve): void => { child.once('exit', (): void => resolve()) }) }
})

test('source launch restores only an explicitly captured exact editable root', (): void => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-source-'))
  const specPath = path.join(home, 'launch.json')
  const childRoot = path.join(home, 'child')
  fs.mkdirSync(childRoot)
  const options = { exe: process.execPath, root: home, origin: 'source' as const, home,
    'user-data': path.join(home, 'user-data'), out: home, phase: 'old' as const,
    'expect-commit': 'a'.repeat(40), 'launch-spec': specPath }
  const writeSpec = (sourceRoot: string): void => {
    // oxlint-disable-next-line anti-slop/no-shape-in-symbol-names -- This is the existing launch-capture wire field.
    fs.writeFileSync(specPath, JSON.stringify({ argv: [process.execPath], cwd: home, matchedShape: 'packaged',
      env: { HERMES_DESKTOP_PYTHON: process.execPath, HERMES_PYTHON_SRC_ROOT: sourceRoot } }))
  }
  try {
    writeSpec(home)
    expect(resolveSmokeLaunch(options).env.HERMES_PYTHON_SRC_ROOT).toBe(home)
    for (const wrong of [childRoot, path.dirname(home), 'relative-root']) {
      writeSpec(wrong)
      expect((): void => { resolveSmokeLaunch(options) }).toThrow('expected source installation')
    }
    expect(smokeEnvironment({ HERMES_PYTHON_SRC_ROOT: home }, home, options['user-data']).HERMES_PYTHON_SRC_ROOT).toBeUndefined()
  } finally { fs.rmSync(home, { recursive: true, force: true }) }
})

test.runIf(process.platform === 'linux')('module-launched source listener proves its import root without an argv path', async (): Promise<void> => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-module-'))
  fs.mkdirSync(path.join(home, 'hermes_cli'))
  fs.writeFileSync(path.join(home, 'hermes_cli', '__init__.py'), '')
  fs.writeFileSync(path.join(home, 'hermes_cli', 'main.py'), 'import socket, time\ns = socket.socket()\ns.bind(("127.0.0.1", 0))\ns.listen()\nprint(s.getsockname()[1], flush=True)\ntime.sleep(60)\n')
  const child = spawn('python3', ['-m', 'hermes_cli.main'], {
    cwd: home, env: { ...process.env, HERMES_PYTHON_SRC_ROOT: home }, stdio: ['ignore', 'pipe', 'pipe'],
  })
  try {
    const port = await new Promise<number>((resolve, reject): void => {
      child.once('error', reject)
      child.stdout.once('data', (data: Buffer): void => resolve(Number(data.toString().trim())))
    })
    const backend = localBackendProcess(port, process.pid)
    expect(backend.sourceRoot).toBe(home)
    expect(backend.cwd).toBe(home)
    expect((): void => assertBackendOrigin(backend, home, 'source')).not.toThrow()
    expect((): void => assertBackendOrigin({ ...backend, sourceRoot: undefined }, home, 'source')).not.toThrow()
    expect((): void => assertBackendOrigin(backend, os.tmpdir(), 'source')).toThrow('source tree')
  } finally {
    child.kill()
    await new Promise<void>((resolve): void => { child.once('exit', (): void => resolve()) })
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('historical identity needs verified provenance and never overrides an app-reported mismatch', (): void => {
  const expected = 'a'.repeat(40)
  const identity = { appVersion: 'historical', commit: null, hermesRoot: '/unused', platform: process.platform }
  expect((): void => assertChatCommit(identity, expected)).toThrow('provenance is required')
  expect((): void => assertChatCommit(identity, expected, expected)).not.toThrow()
  expect((): void => assertChatCommit(identity, expected, 'b'.repeat(40))).toThrow('does not equal expected')
  expect((): void => assertChatCommit({ ...identity, commit: 'b'.repeat(40) }, expected, expected)).toThrow('Running commit')
  expect((): void => assertChatCommit({ ...identity, commit: expected }, expected)).not.toThrow()
  const resources = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-stamp-'))
  const payload = path.join(resources, 'agent-payload')
  fs.mkdirSync(payload)
  try {
    fs.writeFileSync(path.join(resources, 'install-stamp.json'), JSON.stringify({ payload: 'bundled', commit: expected }))
    expect(readInstallationCommit(payload, 'bundled')).toBe(expected)
    fs.writeFileSync(path.join(resources, 'install-stamp.json'), JSON.stringify({ payload: 'bundled', commit: 'malformed' }))
    expect((): void => { readInstallationCommit(payload, 'bundled') }).toThrow()
  } finally { fs.rmSync(resources, { recursive: true, force: true }) }
})

test('driver strips caller secrets and records missing executables as failure without launching', async (): Promise<void> => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-admission-'))
  try {
    const env = smokeEnvironment({ PATH: '/usr/bin', DISPLAY: ':1', OPENAI_API_KEY: 'secret', HERMES_DESKTOP_BOOT_FAKE: '1',
      HERMES_DESKTOP_HERMES_ROOT: '/wrong', PYTHONPATH: '/wrong', NODE_OPTIONS: '--inspect', HERMES_HOME: '/wrong' }, home, path.join(home, 'user-data'))
    expect(env).toMatchObject({ PATH: '/usr/bin', DISPLAY: ':1', HERMES_HOME: home })
    for (const key of ['OPENAI_API_KEY', 'HERMES_DESKTOP_BOOT_FAKE', 'HERMES_DESKTOP_HERMES_ROOT', 'PYTHONPATH', 'NODE_OPTIONS']) {
      expect(env[key]).toBeUndefined()
    }
    await expect(runInstalledDesktopSmoke({ exe: path.join(home, 'missing'), root: home, origin: 'bundled', home,
      'user-data': path.join(home, 'user-data'), out: home, phase: 'installed', 'expect-commit': 'a'.repeat(40) })).rejects.toThrow()
    expect(JSON.parse(fs.readFileSync(path.join(home, 'desktop-chat-installed.json'), 'utf8'))).toMatchObject({ status: 'failed', origin: 'bundled' })
  } finally { fs.rmSync(home, { recursive: true, force: true }) }
})

test('a bundle-env HERMES_HOME clear cannot strand the mock config outside the resolved home', async (): Promise<void> => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-bundle-clear-'))
  const home = path.join(root, 'home')
  const userData = path.join(root, 'root', 'user-data')
  // A real-but-dummy executable: admission passes, seeding runs, and the
  // Electron launch then fails fast because this is not an Electron app.
  const exe = path.join(root, 'root', 'fake-desktop')
  fs.mkdirSync(path.join(root, 'root'), { recursive: true })
  fs.writeFileSync(exe, '#!/bin/sh\nexit 1\n')
  fs.chmodSync(exe, 0o755)
  try {
    const mock = await startMockServer()
    try {
      // The bundled app's banner turns HERMES_HOME=null into HERMES_HOME='', so
      // resolveDesktopHermesHome falls to <userData>/hermes-home. The driver must
      // have seeded THAT home, not only the --home the caller named.
      await expect(runInstalledDesktopSmoke({ exe, root: path.join(root, 'root'), origin: 'bundled', home,
        'user-data': userData, out: root, phase: 'installed', 'expect-commit': 'a'.repeat(40) })).rejects.toThrow()
      for (const candidate of candidateSmokeHermesHomes(home, userData)) {
        expect(yaml.load(fs.readFileSync(path.join(candidate, 'config.yaml'), 'utf8'))).toMatchObject({ model: { provider: 'mock' } })
        expect(fs.readFileSync(path.join(candidate, '.env'), 'utf8')).toMatch(/MOCK_API_KEY=/)
      }
      // Electron resolves shell folders before 'ready'; the sandboxed AppData/XDG
      // dirs must exist or Windows applyDesktopIdentity crashes at launch.
      for (const dir of ['AppData/Roaming', 'AppData/Local', '.config', '.local/share', '.cache']) {
        expect(fs.statSync(path.join(home, '.desktop-smoke-home', ...dir.split('/'))).isDirectory()).toBe(true)
      }
    } finally {
      await mock.close()
    }
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
})

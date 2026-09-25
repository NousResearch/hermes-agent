import { spawn, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import yaml from 'js-yaml'
import { expect, test } from 'vitest'

import { candidateSmokeHermesHomes, predictSmokeHermesHome, resolveSmokeLaunch, runInstalledDesktopSmoke, smokeEnvironment } from '../../tests/install/e2e-assets/desktop-smoke.ts'
import { sourceRuntimeSettleCommand } from '../../tests/install/e2e-assets/source-runtime-settle.mjs'
import { assertUpdateWindowBackendOrigin, assertUpdateWindowProcess } from '../../tests/install/e2e-assets/update-window-chat.mjs'

import { assertChatCommit, newCompletedPair, readMockPrompts, type TranscriptMessage } from './desktop-chat-smoke.ts'
import { assertBackendOrigin, localBackendProcess, readBundledBundleEnv, readInstallationCommit } from './desktop-smoke-process.ts'
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
    // A body shape the witness does not recognize records nothing rather than a
    // half-parsed string — that is the empty-witness case the desktop smoke
    // reports as a poll timeout, so the mock logs the parsed shape on this path.
    for (const content of [{ type: 'text', text: 'unrecognized shape' }, 42]) {
      await fetch(`${first.url}/v1/chat/completions`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: [{ role: 'user', content }] }),
      })
    }
    // No user turn at all leaves `lastUserMessage` undefined — the shape the
    // diagnostic must still describe without throwing.
    await fetch(`${first.url}/v1/chat/completions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'assistant', content: 'no user turn' }] }),
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
      model: { provider: 'custom', base_url: 'http://127.0.0.1:9001/v1', temperature: 0.7, context_length: 12000 },
      providers: {},
      // Exactly one entry, repointed at the second URL: reconfiguring must not
      // stack duplicates. The readiness ladder reads model.base_url (above),
      // not this list; the entry names the same endpoint for `custom:<name>`.
      custom_providers: [{ name: 'Mock', base_url: 'http://127.0.0.1:9001/v1', key_env: 'OPENAI_API_KEY' }],
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

test('a backend bound to the tree by environment needs no root in argv', (): void => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-env-origin-'))
  try {
    const root = path.join(home, 'hermes-agent')
    fs.mkdirSync(path.join(root, 'venv'), { recursive: true })
    // The macOS shape: a venv's interpreter is a SYMLINK to the framework
    // binary, and the app resolves it before spawning, so argv names that binary
    // and never the installation root.
    const resolvedPython = fs.realpathSync(process.execPath)
    const interpreter = path.join(home, 'python3.11')
    fs.symlinkSync(resolvedPython, interpreter)
    const base = { pid: process.pid, parentPid: 1, executable: interpreter, cwd: home,
      command: `"${resolvedPython}" "-m" "hermes_cli.main" serve --host 127.0.0.1 --port 0` }
    // Control: with no environment evidence this is still a different tree.
    expect((): void => { assertBackendOrigin(base, root, 'source') }).toThrow('source tree')
    expect((): void => {
      assertBackendOrigin({ ...base, pythonPath: `${root}${path.delimiter}/elsewhere` }, root, 'source')
    }).not.toThrow()
    expect((): void => {
      assertBackendOrigin({ ...base, virtualEnv: path.join(root, 'venv') }, root, 'source')
    }).not.toThrow()
    // Evidence naming some OTHER tree is no evidence for this one.
    expect((): void => {
      assertBackendOrigin({ ...base, pythonPath: path.join(home, 'elsewhere') }, root, 'source')
    }).toThrow('source tree')
    expect((): void => {
      assertBackendOrigin({ ...base, virtualEnv: path.join(home, 'other-venv') }, root, 'source')
    }).toThrow('source tree')
  } finally { fs.rmSync(home, { recursive: true, force: true }) }
})

test('a platform that cannot read the backend environment proves ownership by the app', (): void => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-owner-origin-'))
  try {
    const root = path.join(home, 'hermes-agent')
    const other = path.join(home, 'other-tree')
    fs.mkdirSync(root, { recursive: true })
    fs.mkdirSync(other, { recursive: true })
    // The Windows shape: the app's venv launcher hands the interpreter over as a system
    // python, so argv never names the tree, and this platform exposes neither the
    // backend's cwd nor its environment.
    const backend = {
      pid: process.pid,
      parentPid: 1,
      executable: path.join(home, 'python.exe'),
      command: `"${path.join(home, 'python.exe')}" "-m" "hermes_cli.main" serve --host 127.0.0.1 --port 0`,
    }
    // Control row: with nothing readable and no report from the app, this is still a
    // different tree.
    expect((): void => { assertBackendOrigin(backend, root, 'source') }).toThrow('source tree')
    // The app reported resolving this root (the driver asserts that against options.root
    // before calling), and the listener was already tied to that same app process.
    expect((): void => {
      assertBackendOrigin(backend, root, 'source', { appReportedRoot: root })
    }).not.toThrow()
    // A report of some other tree is no evidence for this one.
    expect((): void => {
      assertBackendOrigin(backend, root, 'source', { appReportedRoot: other })
    }).toThrow('source tree')
    // Readable process evidence keeps the strict path: an environment naming another
    // tree is not rescued by the app's own report.
    expect((): void => {
      assertBackendOrigin({ ...backend, cwd: other, pythonPath: other }, root, 'source', { appReportedRoot: root })
    }).toThrow('source tree')
  } finally { fs.rmSync(home, { recursive: true, force: true }) }
})

test('OLD update-window source provenance carries its verified app identity to the listener check', (): void => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-old-owner-origin-'))
  try {
    const root = path.join(home, 'hermes-agent')
    const other = path.join(home, 'other-tree')
    fs.mkdirSync(root, { recursive: true })
    fs.mkdirSync(other, { recursive: true })
    const backend = { pid: 2, parentPid: 1, executable: path.join(home, 'python.exe'),
      command: `"${path.join(home, 'python.exe')}" -m hermes_cli.main dashboard --port 0` }
    expect((): void => {
      assertUpdateWindowBackendOrigin(backend, { hermesRoot: root }, root, 'source')
    }).not.toThrow()
    expect((): void => {
      assertUpdateWindowBackendOrigin(backend, { hermesRoot: other }, root, 'source')
    }).toThrow('resolved another source installation')
  } finally { fs.rmSync(home, { recursive: true, force: true }) }
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

test('a module launch proves its tree without leaning on the app-owned cwd', (): void => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-venv-root-'))
  try {
    const python = path.join(root, 'venv', 'bin', 'python')
    const launched = (executable: string, command: string): Parameters<typeof assertBackendOrigin>[0] =>
      ({ pid: 1, parentPid: 1, executable, command, cwd: path.join(os.tmpdir(), 'app-owned-cwd') })
    const venv = launched(python, `"${python}" "-m" "hermes_cli.main" "serve" --host 127.0.0.1 --port 0`)
    // The app owns the backend's cwd; the installation's own venv interpreter is the evidence.
    expect((): void => assertBackendOrigin(venv, root, 'source')).not.toThrow()
    // A foreign interpreter whose command names no tree is still rejected.
    expect((): void => assertBackendOrigin(launched('/usr/bin/python3', '"python3" "-m" "hermes_cli.main" "serve"'), root, 'source')).toThrow('source tree')
    // A captured root disagrees: authoritative, even when the command names the expected tree.
    expect((): void => assertBackendOrigin({ ...venv, sourceRoot: os.tmpdir() }, root, 'source')).toThrow('source tree')
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
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
    expect(env).toMatchObject({ PATH: '/usr/bin', DISPLAY: ':1', HERMES_HOME: home,
      HERMES_DESKTOP_SKIP_QUIT_CONFIRM: '1' })
    for (const key of ['OPENAI_API_KEY', 'HERMES_DESKTOP_BOOT_FAKE', 'HERMES_DESKTOP_HERMES_ROOT', 'PYTHONPATH', 'NODE_OPTIONS']) {
      expect(env[key]).toBeUndefined()
    }
    await expect(runInstalledDesktopSmoke({ exe: path.join(home, 'missing'), root: home, origin: 'bundled', home,
      'user-data': path.join(home, 'user-data'), out: home, phase: 'installed', 'expect-commit': 'a'.repeat(40) })).rejects.toThrow()
    expect(JSON.parse(fs.readFileSync(path.join(home, 'desktop-chat-installed.json'), 'utf8'))).toMatchObject({ status: 'failed', origin: 'bundled' })
  } finally { fs.rmSync(home, { recursive: true, force: true }) }
})

test.runIf(process.platform !== 'win32')('OLD and NEW source smokes settle the clean runtime before Electron launch', async (): Promise<void> => {
  for (const phase of ['old', 'new'] as const) {
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-source-settle-'))
  const root = path.join(workspace, 'source')
  const home = path.join(workspace, 'home')
  const out = path.join(workspace, 'out')
  const userData = path.join(workspace, 'user-data')
  const exe = path.join(root, 'fake-desktop')
  const launcher = path.join(root, '.hermes', 'bin', 'hermes')
  const witness = path.join(workspace, 'settled')

  try {
    fs.mkdirSync(path.dirname(launcher), { recursive: true })
    fs.writeFileSync(exe, '#!/bin/sh\nexit 1\n')
    fs.chmodSync(exe, 0o755)
    fs.writeFileSync(launcher, `#!/bin/sh
set -eu
[ "$1" = status ]
[ "$HERMES_HOME" = ${JSON.stringify(home)} ]
[ "$HOME" = ${JSON.stringify(path.join(home, '.desktop-smoke-home'))} ]
[ -z "\${PM_E2E_LEAK-}" ]
printf 'clean source runtime settled\\n'
: > ${JSON.stringify(witness)}
`)
    fs.chmodSync(launcher, 0o755)

    const refuseLaunch = async (): Promise<never> => {
      expect(fs.existsSync(witness)).toBe(true)
      throw new Error('launch observed settled runtime')
    }

    const prior = process.env.PM_E2E_LEAK
    process.env.PM_E2E_LEAK = 'must be stripped'

    try {
      await expect(runInstalledDesktopSmoke({ exe, root, origin: 'source', home, 'user-data': userData,
        out, phase, 'expect-commit': 'a'.repeat(40) }, refuseLaunch)).rejects.toThrow('launch observed settled runtime')
    } finally {
      if (prior === undefined) { delete process.env.PM_E2E_LEAK } else { process.env.PM_E2E_LEAK = prior }
    }

    expect(fs.readFileSync(path.join(out, `desktop-source-settle-${phase}.log`), 'utf8')).toContain('clean source runtime settled')
  } finally { fs.rmSync(workspace, { recursive: true, force: true }) }
  }
})

test('Windows source settle bypasses the current cmd launcher beside a stale historical exe', (): void => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-windows-settle-'))
  try {
    const bin = path.join(root, '.hermes', 'bin')
    fs.mkdirSync(bin, { recursive: true })
    const current = path.join(bin, 'hermes.cmd')
    const python = path.join(root, 'managed python', 'python.exe')
    fs.mkdirSync(path.dirname(python), { recursive: true })
    fs.writeFileSync(python, '')
    const prepareLaunch = path.join(root, 'hermes_cli', 'venv_sync.py')
    fs.mkdirSync(path.dirname(prepareLaunch), { recursive: true })
    fs.writeFileSync(prepareLaunch, '')
    fs.writeFileSync(current, `@"${python}" -I -c "import base64; exec(base64.b64decode('eA=='))" %*\r\n`)
    fs.writeFileSync(path.join(bin, 'hermes.exe'), 'locked historical launcher')
    const invocation = sourceRuntimeSettleCommand(root, { ComSpec: 'C:\\Windows\\System32\\cmd.exe' }, 'win32')
    expect(invocation).toEqual({
      launcher: current,
      command: python,
      args: ['-I', '-B', '-c', `import pathlib, sys; sys.path.insert(0, ${JSON.stringify(root)}); from hermes_cli.venv_sync import prepare_launch; prepare_launch(pathlib.Path(${JSON.stringify(root)}), ['status'])`],
      windowsVerbatimArguments: false,
    })
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
})

test('Windows source settle bypasses the generated cmd command line', (): void => {
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-windows-settle-live-'))
  const root = path.join(workspace, 'source with spaces')
  try {
    const bin = path.join(root, '.hermes', 'bin')
    const witness = path.join(workspace, 'settled.txt')
    fs.mkdirSync(bin, { recursive: true })
    const pythonProbe = spawnSync('python', ['-c', 'import sys; print(sys.executable)'], { encoding: 'utf8' })
    expect(pythonProbe.status, pythonProbe.stderr || String(pythonProbe.error)).toBe(0)
    const python = pythonProbe.stdout.trim()
    fs.writeFileSync(path.join(bin, 'hermes.cmd'), `@"${python}" -I -c "import base64; exec(base64.b64decode('eA=='))" %*\r\n`)
    const prepareLaunch = path.join(root, 'hermes_cli', 'venv_sync.py')
    fs.mkdirSync(path.dirname(prepareLaunch), { recursive: true })
    fs.writeFileSync(prepareLaunch, `from pathlib import Path\ndef prepare_launch(root, args):\n    Path(${JSON.stringify(witness)}).write_text(str(root) + '\\n' + '\\n'.join(args))\n`)
    fs.writeFileSync(path.join(bin, 'hermes.exe'), 'locked historical launcher')
    const invocation = sourceRuntimeSettleCommand(root, process.env, 'win32')
    const result = spawnSync(invocation.command, invocation.args, {
      cwd: root, env: process.env, encoding: 'utf8', windowsHide: true,
      windowsVerbatimArguments: invocation.windowsVerbatimArguments,
    })
    expect(result.status, result.stderr || String(result.error)).toBe(0)
    expect(fs.readFileSync(witness, 'utf8').split(/\r?\n/)).toEqual([
      root,
      'status',
    ])
  } finally { fs.rmSync(workspace, { recursive: true, force: true }) }
})

test('update-window process checks use the isolated launch environment, not the driver environment', (): void => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-update-window-process-'))
  try {
    const executable = path.join(root, 'Hermes')
    const isolated = path.join(root, 'isolated-user-data')
    const driver = path.join(root, 'driver-user-data')
    fs.writeFileSync(executable, '')
    fs.mkdirSync(isolated)
    fs.mkdirSync(driver)
    const prior = process.env.HERMES_DESKTOP_USER_DATA_DIR
    process.env.HERMES_DESKTOP_USER_DATA_DIR = driver
    try {
      expect(() => assertUpdateWindowProcess(
        { executable, resources: root, userData: isolated },
        { executable, root, origin: 'source', userData: isolated },
      )).not.toThrow()
      expect(() => assertUpdateWindowProcess(
        { executable, resources: root, userData: isolated },
        { executable, root, origin: 'source', userData: driver },
      )).toThrow('OLD update window did not honor isolated userData')
    } finally {
      if (prior === undefined) {
        delete process.env.HERMES_DESKTOP_USER_DATA_DIR
      } else {
        process.env.HERMES_DESKTOP_USER_DATA_DIR = prior
      }
    }
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
})

test('predictSmokeHermesHome replays the bundle banner through the shared resolver', (): void => {
  const launchEnv = { HERMES_HOME: '/pinned/home', HERMES_DESKTOP_USER_DATA_DIR: '/pinned/userdata', LOCALAPPDATA: 'C:/Users/runner/AppData/Local' }
  // No baked env: the driver's own HERMES_HOME pin wins.
  expect(predictSmokeHermesHome(launchEnv, {}, 'linux', '/real/home')).toBe('/pinned/home')
  // HERMES_HOME cleared -> the <userData>/hermes-home fallback.
  expect(predictSmokeHermesHome(launchEnv, { HERMES_HOME: null }, 'linux', '/real/home')).toBe('/pinned/userdata/hermes-home')
  // Both cleared + baked suffix -> the platform default with that suffix.
  expect(predictSmokeHermesHome(launchEnv, { HERMES_HOME: null, HERMES_DESKTOP_USER_DATA_DIR: null, HERMES_DATA_DIR_SUFFIX: '-magic' }, 'linux', '/real/home')).toBe('/real/home/.hermes-magic')
  // On Windows the default derives from the sandboxed LOCALAPPDATA, not the OS home.
  expect(predictSmokeHermesHome(launchEnv, { HERMES_HOME: null, HERMES_DESKTOP_USER_DATA_DIR: null, HERMES_DATA_DIR_SUFFIX: '-magic' }, 'win32', 'C:/Users/real')).toBe('C:\\Users\\runner\\AppData\\Local\\hermes-magic')
})

test('readBundledBundleEnv reads the stamped defaults/clears and is absent when unstamped', (): void => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-bundlenv-'))
  try {
    expect(readBundledBundleEnv(path.join(root, 'agent-payload'))).toBeUndefined()
    fs.writeFileSync(path.join(root, 'install-stamp.json'), JSON.stringify({ payload: 'bundled', commit: 'a'.repeat(40), bundleEnv: { HERMES_HOME: null, SUFFIX: 'x' } }))
    expect(readBundledBundleEnv(path.join(root, 'agent-payload'))).toEqual({ HERMES_HOME: null, SUFFIX: 'x' })
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
})

test('a bundle-env HERMES_HOME clear cannot strand the mock config outside the resolved home', async (): Promise<void> => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-bundle-clear-'))
  const home = path.join(root, 'home')
  const userData = path.join(root, 'root', 'user-data')
  // A real-but-dummy executable so admission passes and seeding runs; the
  // launcher itself is substituted below, so Playwright never spawns it (a
  // process that exits at once leaves Playwright with dangling rejections).
  const exe = path.join(root, 'root', 'fake-desktop')
  fs.mkdirSync(path.join(root, 'root'), { recursive: true })
  fs.writeFileSync(exe, '#!/bin/sh\nexit 1\n')
  fs.chmodSync(exe, 0o755)
  const refuseLaunch = async (): Promise<never> => { throw new Error('launch refused by test') }
  try {
    const mock = await startMockServer()
    try {
      // The bundled app's banner turns HERMES_HOME=null into HERMES_HOME='', so
      // resolveDesktopHermesHome falls to <userData>/hermes-home. The driver must
      // have seeded THAT home, not only the --home the caller named.
      await expect(runInstalledDesktopSmoke({ exe, root: path.join(root, 'root'), origin: 'bundled', home,
        'user-data': userData, out: root, phase: 'installed', 'expect-commit': 'a'.repeat(40) }, refuseLaunch)).rejects.toThrow('launch refused by test')
      for (const candidate of candidateSmokeHermesHomes(home, userData)) {
        expect(yaml.load(fs.readFileSync(path.join(candidate, 'config.yaml'), 'utf8'))).toMatchObject({ model: { provider: 'custom' } })
        const env = fs.readFileSync(path.join(candidate, '.env'), 'utf8')
        expect(env).toMatch(/MOCK_API_KEY=/)
        expect(env).toMatch(/OPENAI_API_KEY=/)
        expect(env).toMatch(/OPENAI_BASE_URL=http:\/\/127\.0\.0\.1:\d+\/v1/)
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

test('writeEnvFile without a URL keeps the journey provider endpoint', (): void => {
  // The install e2e configures the mock provider through the CLI, which writes
  // OPENAI_BASE_URL/OPENAI_API_KEY. The desktop smoke then calls writeEnvFile with
  // no URL, and stripping the pair there deleted the endpoint the snapshot was
  // meant to record -- the upgrade's own .env sync later put it back, and the leg
  // failed as "the upgrade changed the user's own state".
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-env-'))
  try {
    writeEnvFile(home, 'e2e-mock-key', 'http://127.0.0.1:9000')
    expect(fs.readFileSync(path.join(home, '.env'), 'utf8')).toContain('OPENAI_BASE_URL=http://127.0.0.1:9000/v1')

    writeEnvFile(home)

    const after = fs.readFileSync(path.join(home, '.env'), 'utf8')
    expect(after).toContain('OPENAI_BASE_URL=http://127.0.0.1:9000/v1')
    expect(after).toContain('OPENAI_API_KEY=e2e-mock-key')
    expect(after).toContain('MOCK_API_KEY=e2e-mock-key')
  } finally { fs.rmSync(home, { recursive: true, force: true }) }
})

test('re-writing the same provider config is byte-identical', (): void => {
  // Filtering the keys out and re-appending them moved a journey's own .env
  // entries on every call: same keys, same values, different bytes -- which the
  // user-state verifier reported as "0 deleted, 1 modified ... no key differs".
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-env-order-'))
  try {
    // The key is deliberately NOT last, with a comment after it: filtering the
    // managed keys out and re-appending them moved them past the journey's own
    // entries, which is the "no key differs, comments/order/blanks only" the
    // user-state verifier reported.
    const before = 'OPENAI_BASE_URL=http://127.0.0.1:9000/v1\n# keep me\nOTHER_TEST_VALUE=kept\n'
    fs.writeFileSync(path.join(home, '.env'), before)

    writeEnvFile(home, 'e2e-mock-key', 'http://127.0.0.1:9000')

    expect(fs.readFileSync(path.join(home, '.env'), 'utf8')).toBe(
      'OPENAI_BASE_URL=http://127.0.0.1:9000/v1\n# keep me\nOTHER_TEST_VALUE=kept\n'
      + 'MOCK_API_KEY=e2e-mock-key\nOPENAI_API_KEY=e2e-mock-key\n')
  } finally { fs.rmSync(home, { recursive: true, force: true }) }
})

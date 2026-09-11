import assert from 'node:assert/strict'
import childProcess from 'node:child_process'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'

import { Arch, MacPackager } from 'app-builder-lib'
import { MacTargetHelper } from 'app-builder-lib/internal'
import { test, vi } from 'vitest'

const require = createRequire(import.meta.url)
const config = require('../electron-builder.config.cjs')
const repo = path.resolve(import.meta.dirname, '../../..')
const desktop = path.resolve(import.meta.dirname, '..')
const python = process.env.HERMES_PYTHON || process.env.UV_PYTHON || 'python'

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mac-digest-order-'))
  const app = path.join(root, 'Hermes.app')
  const payload = path.join(app, 'Contents', 'Resources', 'agent-payload')
  const tools = path.join(payload, 'tools')
  const nested = path.join(tools, 'chromium', 'Browser.app')
  const binaries = [path.join(app, 'Contents', 'MacOS', 'Hermes'),
    path.join(tools, 'python', 'bin', 'python3'),
    path.join(nested, 'Contents', 'MacOS', process.platform === 'win32' ? 'chrome.exe' : 'Chromium')]
  for (const binary of binaries) {
    fs.mkdirSync(path.dirname(binary), { recursive: true })
    const bytes = Buffer.alloc(64)
    bytes.writeUInt32LE(0xfeedfacf)
    bytes.writeUInt32LE(0x100000c, 4)
    fs.writeFileSync(binary, bytes)
  }
  const ignored = path.join(tools, 'python', 'non-macho.bin')
  fs.writeFileSync(ignored, Buffer.alloc(64))
  const env = { ...process.env, HERMES_HOME: path.join(root, 'home'),
    HERMES_RUNTIME_DIR: path.join(root, 'state'), UV_PYTHON: python,
    UV_OFFLINE: '1', UV_PYTHON_DOWNLOADS: 'never', UV_CACHE_DIR: path.join(root, 'cache') }
  delete env.PYTHONHOME
  delete env.PYTHONPATH
  const py = code => childProcess.execFileSync(python, ['-c', code, payload], {
    cwd: repo, env, encoding: 'utf8', timeout: 60000
  }).trim()
  // The digest hook only needs the manifest to mark the payload as present.
  fs.writeFileSync(path.join(payload, 'manifest.json'), '{}')
  py('from pathlib import Path; import sys; from scripts.bundles.payload import record_tools; from pm.store import current_target; root=Path(sys.argv[1]); record_tools(root,Path("pm/lock.json"),current_target(),{"python":"python","chromium":"chromium"})')
  const facts = path.join(tools, 'facts.json')
  const initial = JSON.parse(fs.readFileSync(facts, 'utf8'))
  return { root, app, payload, tools, nested, binaries, ignored, env, py, facts, initial }
}

async function signThroughBuilder(f) {
  const packager = {
    config,
    appInfo: { type: 'module' },
    info: { getWorkspaceRoot: async () => repo },
    buildResourcesDir: path.join(desktop, 'build'),
    resourceList: Promise.resolve([]),
    getResource: async value => path.resolve(desktop, value),
    expandArch: value => [value]
  }
  packager.helper = new MacTargetHelper(packager)
  const identity = { name: 'Developer ID Application: Test Fixture (ABC123)' }
  const declared = typeof config.mac.sign === 'object' ? config.mac.sign : undefined
  const opts = await packager.helper.buildSignOptions(
    f.app, identity, declared, path.join(f.root, 'fixture.keychain-db'), Arch.arm64, 'mac')
  // This is the builder's real dispatch, not a test-only invocation of a factory.
  await MacPackager.prototype.doSign.call(packager, opts, config.mac.sign, identity)
}

function armEnvironment(f) {
  for (const name of ['HERMES_HOME', 'HERMES_RUNTIME_DIR', 'UV_PYTHON', 'UV_OFFLINE', 'UV_PYTHON_DOWNLOADS', 'UV_CACHE_DIR']) {
    vi.stubEnv(name, f.env[name])
  }
  vi.stubEnv('PYTHONHOME', undefined)
  vi.stubEnv('PYTHONPATH', undefined)
}

test('builder dispatch refreshes real facts after every payload signature and before the outer seal', async () => {
  const f = fixture()
  const signed = []
  const commands = []
  let outerFacts
  let digestsAtSeal
  armEnvironment(f)
  // Execute the installed walk and argument builders; intercept only native codesign.
  const native = vi.spyOn(childProcess, 'execFile').mockImplementation((file, args, options, callback) => {
    try {
      assert.equal(file, 'codesign', 'no other native signing or keychain command is permitted')
      if (args.includes('--sign')) {
        const target = args.at(-1)
        commands.push(args)
        if (target === f.app) {
          digestsAtSeal = JSON.parse(f.py('from pathlib import Path; import sys,json; from pm.store import tree_digest; root=Path(sys.argv[1]); print(json.dumps({name:tree_digest(root/"tools"/name) for name in ("python","chromium")}))'))
          outerFacts = JSON.parse(fs.readFileSync(f.facts, 'utf8'))
        } else if (fs.statSync(target).isDirectory()) {
          const signature = path.join(target, 'Contents', '_CodeSignature', 'CodeResources')
          fs.mkdirSync(path.dirname(signature), { recursive: true })
          fs.writeFileSync(signature, 'fixture byte mutation, not a native signature')
        } else {
          fs.appendFileSync(target, 'fixture byte mutation, not a native signature')
        }
        signed.push(target)
      }
      callback(null, '', '')
    } catch (error) {
      callback(error, '', '')
    }
    return undefined
  })
  try {
    await signThroughBuilder(f)
    assert.equal(signed.at(-1), f.app)
    assert.deepEqual(new Set(signed.slice(0, -1)), new Set([...f.binaries, f.nested]))
    assert.ok(!signed.includes(f.ignored))
    for (const args of commands) {
      assert.equal(args[args.indexOf('--sign') + 1], 'Developer ID Application: Test Fixture (ABC123)')
      assert.equal(args[args.indexOf('--keychain') + 1], path.join(f.root, 'fixture.keychain-db'))
      assert.equal(path.resolve(args[args.indexOf('--entitlements') + 1]),
        path.join(desktop, 'electron', args.at(-1) === f.app ? 'entitlements.mac.plist' : 'entitlements.mac.inherit.plist'))
      if (process.platform === 'darwin') assert.ok(args.includes('--options') && args[args.indexOf('--options') + 1].includes('runtime'))
    }
    for (const name of Object.keys(digestsAtSeal)) {
      assert.notEqual(digestsAtSeal[name], f.initial.packages[name].digest)
      assert.deepEqual(outerFacts.packages[name], { ...f.initial.packages[name], digest: digestsAtSeal[name] })
    }
  } finally {
    native.mockRestore()
    vi.unstubAllEnvs()
    fs.rmSync(f.root, { recursive: true, force: true })
  }
}, 30000)

test('builder dispatch refuses corrupt payload facts before the outer seal but signs a payload-free app', async () => {
  const f = fixture()
  const signed = []
  armEnvironment(f)
  fs.writeFileSync(f.facts, 'invalid-json')
  const native = vi.spyOn(childProcess, 'execFile').mockImplementation((file, args, options, callback) => {
    assert.equal(file, 'codesign')
    if (args.includes('--sign')) signed.push(args.at(-1))
    callback(null, '', '')
    return undefined
  })
  try {
    await assert.rejects(signThroughBuilder(f))
    assert.ok(signed.includes(f.binaries[1]))
    assert.ok(!signed.includes(f.app))
    assert.equal(fs.readFileSync(f.facts, 'utf8'), 'invalid-json')
    fs.rmSync(f.payload, { recursive: true })
    signed.length = 0
    await signThroughBuilder(f)
    assert.deepEqual(signed, [f.binaries[0], f.app])
    assert.ok(!fs.existsSync(f.payload))
  } finally {
    native.mockRestore()
    vi.unstubAllEnvs()
    fs.rmSync(f.root, { recursive: true, force: true })
  }
}, 30000)

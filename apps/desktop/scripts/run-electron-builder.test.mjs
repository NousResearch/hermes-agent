import assert from 'node:assert/strict'
import { test } from 'vitest'
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import { runElectronBuilder } from './run-electron-builder.mjs'
import fs from 'node:fs'
import os from 'node:os'
import { publishPackagingInputs } from './prepared-packaging.mjs'
import { recordNativeInputs } from './prepared-native-deps.mjs'

test('validate-only admits real prepared inputs without launching tools and rejects unsafe arguments', async () => {
  const source = path.resolve(import.meta.dirname, '../../..')
  const out = fs.mkdtempSync(path.join(os.tmpdir(), 'builder-validation-'))
  try {
    const electron = path.join(out, 'electron.zip')
    fs.writeFileSync(electron, 'fixture archive')
    const toolsets = { sevenZip: path.join(out, 'sevenZip'), icons: path.join(out, 'icons') }
    for (const dir of Object.values(toolsets)) fs.mkdirSync(dir)
    const manifest = await publishPackagingInputs({ source, out, target: `${process.platform}-${process.arch}`, formats: ['dir'], electron, toolsets })
    const nativeDeps = path.join(out, 'native')
    fs.mkdirSync(path.join(nativeDeps, 'node-pty'), { recursive: true })
    fs.writeFileSync(path.join(nativeDeps, 'node-pty/package.json'), '{}')
    recordNativeInputs({ source, out: nativeDeps, platform: process.platform, arch: process.arch })
    const args = ['--validate-only', '--prepared', manifest, '--native-deps', nativeDeps, '--dir']
    const options = { spawn: () => { throw new Error('validation must not launch tools') } }
    assert.equal(runElectronBuilder(args, options), 0)
    assert.throws(() => runElectronBuilder([...args, '-c.npmRebuild=true'], options), /not admitted/)
    assert.throws(() => runElectronBuilder(['--validate-only'], options), /--prepared/)
    const cli = spawnSync(process.execPath, [path.join(import.meta.dirname, 'run-electron-builder.mjs'), ...args], { encoding: 'utf8' })
    assert.equal(cli.status, 0, cli.stderr)
  } finally {
    fs.rmSync(out, { recursive: true, force: true })
  }
})

test('source multiarch prepares isolated native and packaging inputs before each strict invocation', () => {
  const calls = []
  const spawn = (_node, args) => { calls.push(args); return { status: 0 } }
  assert.equal(runElectronBuilder(['--mac', '--x64', '--arm64', '--dir'], { spawn }), 0)
  for (const arch of ['x64', 'arm64']) {
    const native = calls.find(args => args[0].endsWith('stage-native-deps.mjs') && args.includes(arch))
    assert.ok(native)
    assert.equal(native[native.indexOf('--platform') + 1], 'darwin')
    const prepare = calls.find(args => args[0].endsWith('prepare-packaging-tools.mjs') && args.includes(`darwin-${arch}`))
    assert.ok(prepare)
    const strict = calls.find(args => args[0].endsWith('run-electron-builder.mjs') && args.includes(`--${arch}`))
    assert.ok(strict)
    assert.equal(strict[strict.indexOf('--prepared') + 1], path.join(prepare[prepare.indexOf('--out') + 1], 'prepared.json'))
    assert.equal(strict[strict.indexOf('--native-deps') + 1], native[native.indexOf('--out') + 1])
    assert.equal(strict.filter(arg => ['--x64', '--arm64'].includes(arg)).length, 1)
  }
  assert.equal(calls.length, 6)
  assert.notEqual(calls[0][calls[0].indexOf('--out') + 1], calls[3][calls[3].indexOf('--out') + 1])
  calls.length = 0
  assert.equal(runElectronBuilder(['--dir'], { spawn }), 0)
  assert.equal(calls.filter(args => args[0].endsWith('prepare-packaging-tools.mjs')).length, 1)
  assert.equal(calls.filter(args => args[0].endsWith('run-electron-builder.mjs')).length, 1)
  assert.throws(() => runElectronBuilder(['--mac', '--universal'], { spawn }), /No prepared universal native payload/)
  assert.equal(runElectronBuilder(['--mac', '--x64', '--arm64'], { spawn: () => ({ status: 7 }) }), 7)
})

test('strict builder refuses absent inputs before loading electron-builder', () => {
  const result = spawnSync(process.execPath, [path.join(import.meta.dirname, 'run-electron-builder.mjs'),
    '--prepared', path.join(import.meta.dirname, 'missing-prepared.json'), '--native-deps', 'missing', '--dir'], { encoding: 'utf8' })
  assert.notEqual(result.status, 0)
  assert.match(result.stderr, /run preparation again/)
  assert.doesNotMatch(result.stdout, /electron-builder\s+version/)
})

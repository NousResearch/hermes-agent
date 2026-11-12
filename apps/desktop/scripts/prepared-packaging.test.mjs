import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, test } from 'vitest'
import * as prepared from './prepared-packaging.mjs'
import { validatePreparedBuilderArgs } from './run-electron-builder.mjs'
import { spawn, spawnSync } from 'node:child_process'
import http from 'node:http'

/** @type {string[]} */
const roots = []
afterEach(() => roots.splice(0).forEach(root => fs.rmSync(root, { recursive: true, force: true })))

/** @returns {{ source: string, out: string, electron: string, sevenZip: string, icons: string }} */
function fixture() {
  const source = fs.mkdtempSync(path.join(os.tmpdir(), 'prepared-packager-'))
  roots.push(source)
  const out = path.join(source, 'work')
  fs.mkdirSync(path.join(source, 'apps/desktop'), { recursive: true })
  fs.writeFileSync(path.join(source, 'pnpm-lock.yaml'), "lockfileVersion: '9.0'\n")
  fs.writeFileSync(path.join(source, 'apps/desktop/package.json'), '{}')
  fs.writeFileSync(path.join(source, 'apps/desktop/electron-builder.config.cjs'), 'module.exports = {}')
  const electron = path.join(out, 'electron.zip')
  const sevenZip = path.join(out, 'sevenZip')
  const icons = path.join(out, 'icons')
  fs.mkdirSync(path.join(sevenZip, 'bin'), { recursive: true })
  fs.mkdirSync(icons)
  fs.writeFileSync(electron, 'verified archive fixture')
  fs.writeFileSync(path.join(sevenZip, 'bin', process.platform === 'win32' ? '7za.exe' : '7za'), 'archive utility fixture')
  fs.writeFileSync(path.join(icons, 'icon-tool.js'), 'icon tool fixture')
  return { source, out, electron, sevenZip, icons }
}

test('a failed preparation invalidates an earlier completion claim before loading suppliers', () => {
  const { source, out } = fixture()
  const manifest = path.join(out, 'prepared.json')
  fs.writeFileSync(manifest, '{}')
  const result = spawnSync(process.execPath, [path.join(import.meta.dirname, 'prepare-packaging-tools.mjs'),
    '--source', source, '--out', out, '--cache', path.join(source, 'cache')], { encoding: 'utf8' })
  assert.notEqual(result.status, 0)
  assert.equal(fs.existsSync(manifest), false)
  assert.match(result.stderr, /lock|install|pinned/i)
})

test('prepared inputs are path-bound and reject changed or missing bytes without repair', async () => {
  const inputs = fixture()
  const manifest = await prepared.publishPackagingInputs({
    ...inputs, target: 'linux-x64', formats: ['dir'],
    toolsets: { sevenZip: inputs.sevenZip, icons: inputs.icons },
  })
  const result = prepared.readPackagingInputs(manifest, inputs.source, 'linux-x64')
  assert.equal(result.electron, inputs.electron)
  assert.throws(() => validatePreparedBuilderArgs([], result), /run preparation again/)
  assert.throws(() => validatePreparedBuilderArgs(['--dir', '-c.electronDist=/another.zip'], result), /run preparation again/)
  validatePreparedBuilderArgs(['--dir', '-c.extraMetadata.version=1.2.3'], result)
  fs.writeFileSync(inputs.electron, 'corrupt')
  assert.throws(() => prepared.readPackagingInputs(manifest, inputs.source, 'linux-x64'), /run preparation again/i)
  assert.equal(fs.readFileSync(inputs.electron, 'utf8'), 'corrupt')
  fs.rmSync(inputs.electron)
  assert.throws(() => prepared.readPackagingInputs(manifest, inputs.source, 'linux-x64'), /run preparation again/i)
})

// Windows dir preparation still acquires the signing kit (winCodeSign), so only POSIX skips every download.
test.runIf(process.platform !== 'win32')('an unpacked build prepares from the lock-pinned install with every release host unreachable', async () => {
  const source = path.resolve(import.meta.dirname, '../../..')
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'prepared-dir-offline-'))
  roots.push(root)
  /** @type {string[]} */
  const requests = []
  const server = http.createServer((request, response) => {
    requests.push(String(request.url))
    response.writeHead(504).end()
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(undefined)))
  const mirror = `http://127.0.0.1:${/** @type {import('node:net').AddressInfo} */ (server.address()).port}/`
  try {
    const out = path.join(root, 'packager')
    const child = spawn(process.execPath, [path.join(import.meta.dirname, 'prepare-packaging-tools.mjs'),
      '--source', source, '--out', out, '--cache', path.join(root, 'cache'), '--format', 'dir'], {
      env: { ...process.env, ELECTRON_BUILDER_BINARIES_MIRROR: mirror, ELECTRON_BUILDER_DANGEROUSLY_ALLOW_HTTP: 'true',
        ELECTRON_MIRROR: mirror, HTTPS_PROXY: mirror, HTTP_PROXY: mirror, NO_PROXY: '' },
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let stderr = ''
    child.stderr.on('data', chunk => { stderr += chunk })
    const status = await new Promise(resolve => child.on('close', resolve))
    assert.equal(status, 0, stderr)
    assert.deepEqual(requests, [])
    const inputs = prepared.readPackagingInputs(path.join(out, 'prepared.json'), source)
    assert.deepEqual(inputs.toolsets, {})
    assert.ok(fs.statSync(inputs.electron).isDirectory())
  } finally {
    await new Promise(resolve => server.close(resolve))
  }
})

test('packaged formats still require the archive and icon tools', async () => {
  const inputs = fixture()
  const manifest = await prepared.publishPackagingInputs({
    ...inputs, target: 'linux-x64', formats: ['AppImage'], toolsets: { sevenZip: inputs.sevenZip },
  })
  assert.throws(() => prepared.readPackagingInputs(manifest, inputs.source, 'linux-x64'), /archive or icon tool/)
})

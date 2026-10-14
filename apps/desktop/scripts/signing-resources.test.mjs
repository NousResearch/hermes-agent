import assert from 'node:assert/strict'
import { execFileSync, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { test } from 'vitest'

const require = createRequire(import.meta.url)
const adapter = path.join(import.meta.dirname, 'patch-electron-builder-mac-binary.mjs')
const supplier = path.dirname(require.resolve('@electron/osx-sign'))
const walkUrl = pathToFileURL(path.join(supplier, 'util.js')).href
const signUrl = pathToFileURL(path.join(supplier, 'index.js')).href

function run(root, body, { patched = true, limit = 64 } = {}) {
  const script = path.join(root, 'probe.mjs')
  fs.writeFileSync(script, body)
  const args = [...(patched ? ['--import', adapter] : []), script]
  const result = spawnSync('/bin/sh', ['-c',
    `ulimit -S -n ${limit} && ulimit -H -n ${limit} && exec "$@"`, 'signing-probe', process.execPath, ...args], {
    encoding: 'utf8', env: { ...process.env, NODE_OPTIONS: '', DEBUG: '' }, timeout: 30000
  })
  assert.ifError(result.error)
  return result
}

function fixture(root) {
  const app = path.join(root, 'Probe.app')
  const child = path.join(app, 'Contents/Frameworks/Child.app')
  const framework = path.join(child, 'Contents/Frameworks/Demo.framework')
  const binaries = [path.join(app, 'Contents/MacOS/Probe'), path.join(child, 'Contents/MacOS/Child'),
    path.join(framework, 'Versions/A/Demo')]
  for (const binary of binaries) {
    fs.mkdirSync(path.dirname(binary), { recursive: true })
    fs.writeFileSync(binary, Buffer.from([0xcf, 0xfa, 0xed, 0xfe, 0]))
  }
  fs.symlinkSync('A', path.join(framework, 'Versions/Current'))
  fs.symlinkSync('Versions/Current/Demo', path.join(framework, 'Demo'))
  fs.symlinkSync('Frameworks/Child.app', path.join(app, 'Contents/child-link'))
  fs.symlinkSync('missing', path.join(app, 'Contents/broken-link'))
  for (let i = 0; i < 3000; i++) {
    const dir = path.join(app, 'Contents/Resources', `pkg${i % 40}`, `mod${i % 15}`)
    fs.mkdirSync(dir, { recursive: true })
    fs.writeFileSync(path.join(dir, `f${i}.py`), '# text resource\n')
  }
  // Exercise the exact supplier classifier, including its 3-byte UTF-8 reserve.
  const resources = path.join(app, 'Contents/Resources')
  fs.writeFileSync(path.join(resources, 'utf8.txt'), 'a'.repeat(511) + '😀')
  fs.writeFileSync(path.join(resources, 'empty'), '')
  fs.writeFileSync(path.join(resources, 'pdf'), '%PDF-1.0\n')
  if (process.platform === 'darwin') {
    const plist = require('plist')
    fs.writeFileSync(path.join(root, 'main.c'), 'int main(void) { return 0; }\n')
    for (const binary of binaries) execFileSync('/usr/bin/clang', [path.join(root, 'main.c'),
      ...(binary === binaries[2] ? ['-dynamiclib'] : []), '-o', binary])
    for (const [dir, name, type] of [[app, 'Probe', 'APPL'], [child, 'Child', 'APPL'], [framework, 'Demo', 'FMWK']]) {
      const info = type === 'APPL' ? path.join(dir, 'Contents/Info.plist') : path.join(dir, 'Versions/A/Resources/Info.plist')
      fs.mkdirSync(path.dirname(info), { recursive: true })
      fs.writeFileSync(info, plist.build({ CFBundleExecutable: name, CFBundleIdentifier: `org.hermes.fixture.${name}`,
        CFBundleVersion: '1', CFBundlePackageType: type }))
    }
    fs.symlinkSync('Versions/Current/Resources', path.join(framework, 'Resources'))
  }
  return app
}

const tracking = `
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
let live = 0, peak = 0
const opened = () => { live++; peak = Math.max(peak, live) }
const open = fs.open, close = fs.close, promisesOpen = fs.promises.open
fs.open = function (...args) {
  const cb = args.pop()
  return open.call(fs, ...args, (error, fd) => { if (!error) opened(); cb(error, fd) })
}
fs.close = function (fd, cb) {
  return close.call(fs, fd, error => { if (!error) live--; cb(error) })
}
fs.promises.open = async function (...args) {
  const handle = await promisesOpen.apply(fs.promises, args)
  opened()
  const closeHandle = handle.close.bind(handle)
  handle.close = async () => { await closeHandle(); live-- }
  return handle
}
const { walk } = await import(${JSON.stringify(walkUrl)})
`

test.skipIf(process.platform === 'win32')('bounded probes preserve real supplier candidates and native nested signatures at a low fd limit', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'signing-resources-'))
  try {
    const app = fixture(root)
    const body = `${tracking}
const app = ${JSON.stringify(app)}, contents = path.join(app, 'Contents')
fs.writeFileSync(path.join(contents, 'old.cstemp'), 'stale signature')
const selected = await walk(contents)
while (live) await new Promise(resolve => setImmediate(resolve))
assert.ok(!fs.existsSync(path.join(contents, 'old.cstemp')))
const signed = []
if (process.platform === 'darwin') {
  // The walk excludes dangling links; strict codesign correctly rejects them.
  fs.rmSync(path.join(contents, 'broken-link'), { force: true })
  const { sign } = await import(${JSON.stringify(signUrl)})
  await sign({ app, platform: 'darwin', identity: '-', identityValidation: false,
    preAutoEntitlements: false, preEmbedProvisioningProfile: false, strictVerify: true,
    batchCodesignCalls: false, ignore: file => file.endsWith('/pdf'),
    optionsForFile(file) { signed.push(file); return { timestamp: 'none', hardenedRuntime: true } } })
}
console.log(JSON.stringify({ selected, signed, peak, live }))
`
    const raw = run(root, body, { patched: false })
    assert.notEqual(raw.status, 0)
    assert.match(raw.stderr, /EMFILE/)
    const reference = run(root, body, { patched: false, limit: 4096 })
    assert.equal(reference.status, 0, reference.stderr)
    // Remove signatures from the native reference before the identical second run.
    fs.rmSync(app, { recursive: true })
    fixture(root)
    const limited = run(root, body)
    assert.equal(limited.status, 0, limited.stderr)
    const before = JSON.parse(reference.stdout), after = JSON.parse(limited.stdout)
    assert.deepEqual(after.selected, before.selected)
    assert.deepEqual(after.signed, before.signed)
    assert.ok(after.selected.some(file => file.endsWith('/Demo.framework')))
    assert.ok(after.selected.some(file => file.endsWith('/Child.app')))
    assert.ok(after.selected.every(file => !file.includes('link') && !file.includes('utf8.txt')))
    assert.ok(after.peak <= 16, JSON.stringify(after))
    assert.ok(before.peak > after.peak)
    assert.equal(after.live, 0)
    if (process.platform === 'darwin') execFileSync('/usr/bin/codesign', ['--verify', '--deep', '--strict', app])
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
}, 60000)

test.skipIf(process.platform === 'win32')('probe errors release complete operations and concurrent walks share the bound without patching fs', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'signing-errors-'))
  try {
    const tree = path.join(root, 'tree')
    fs.mkdirSync(tree)
    for (let i = 0; i < 100; i++) fs.writeFileSync(path.join(tree, `f${i}`), 'text')
    const result = run(root, `
import assert from 'node:assert/strict'
import fs from 'node:fs'
const originalOpen = fs.open, originalPromisesOpen = fs.promises.open
await import(${JSON.stringify(pathToFileURL(adapter).href)})
assert.equal(fs.open, originalOpen)
assert.equal(fs.promises.open, originalPromisesOpen)
${tracking.replaceAll("import assert from 'node:assert/strict'", '').replaceAll("import fs from 'node:fs'", '')}
const tree = ${JSON.stringify(tree)}
await assert.rejects(walk(tree + '/missing'), { code: 'ENOENT' })
const measuredOpen = fs.promises.open
let failed = false
fs.promises.open = async (...args) => {
  if (!failed) { failed = true; throw Object.assign(new Error('open failure'), { code: 'EACCES' }) }
  return measuredOpen(...args)
}
await assert.rejects(walk(tree), { code: 'EACCES' })
// Drain the rest of Promise.all by completing a second walk through the same queue.
assert.deepEqual(await walk(tree), [])
failed = false
fs.promises.open = async (...args) => {
  const handle = await measuredOpen(...args)
  if (!failed) { failed = true; handle.read = async () => { throw Object.assign(new Error('read failure'), { code: 'EIO' }) } }
  const closeHandle = handle.close.bind(handle)
  handle.close = async () => { await new Promise(resolve => setTimeout(resolve, 2)); await closeHandle() }
  return handle
}
await assert.rejects(walk(tree), { code: 'EIO' })
assert.deepEqual(await Promise.all([walk(tree), walk(tree), walk(tree)]), [[], [], []])
assert.equal(live, 0)
assert.ok(peak <= 16, String(peak))
console.log('errors released, late closes bounded, fs unchanged')
`, { patched: false })
    assert.equal(result.status, 0, result.stderr)
    assert.match(result.stdout, /late closes bounded/)
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
}, 60000)

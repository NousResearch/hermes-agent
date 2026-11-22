import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { Arch, Packager, Platform, WinPackager } from 'app-builder-lib'
import { afterEach, test } from 'vitest'

import builderConfig from '../electron-builder.config.cjs'

import {
  batchSignAppTree,
  chunk,
  customSign,
  getBinaries
} from './batch-sign-binaries.mjs'

const tmpDirs = []

afterEach(() => {
  for (const dir of tmpDirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

function tmpTree() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'batch-sign-test-'))
  tmpDirs.push(dir)
  return dir
}

test('getBinaries collects .exe and .dll recursively, case-insensitive, sorted', () => {
  const root = tmpTree()
  fs.mkdirSync(path.join(root, 'resources', 'agent-payload', 'tools', 'python', 'bin'), { recursive: true })
  fs.mkdirSync(path.join(root, 'resources', 'agent-payload', 'tools', 'chromium-1', 'meep'), { recursive: true })
  fs.writeFileSync(path.join(root, 'Hermes.exe'), 'x')
  fs.writeFileSync(path.join(root, 'FFMPEG.DLL'), 'x')
  fs.writeFileSync(path.join(root, 'resources', 'agent-payload', 'tools', 'python', 'bin', 'node.exe'), 'x')
  fs.writeFileSync(path.join(root, 'resources', 'agent-payload', 'tools', 'chromium-1', 'meep', 'chrome.dll'), 'x')
  fs.writeFileSync(path.join(root, 'resources', 'README.md'), 'x')

  const files = getBinaries(root).map(f => path.relative(root, f))

  assert.deepEqual(files, [
    path.join('FFMPEG.DLL'),
    path.join('Hermes.exe'),
    path.join('resources', 'agent-payload', 'tools', 'chromium-1', 'meep', 'chrome.dll'),
    path.join('resources', 'agent-payload', 'tools', 'python', 'bin', 'node.exe')
  ])
})

test('getBinaries skips symlinks and honors the skip predicate (product exe)', () => {
  const root = tmpTree()
  fs.mkdirSync(path.join(root, 'tools'), { recursive: true })
  const target = path.join(root, 'tools', 'real.exe')
  fs.writeFileSync(target, 'x')
  fs.writeFileSync(path.join(root, 'Hermes.exe'), 'x')
  const link = path.join(root, 'tools', 'link.exe')
  try {
    fs.symlinkSync(target, link)
  } catch {
    // Windows without symlink privilege: the collection contract under test
    // is the skip predicate; symlink skipping is covered on the other OS.
  }

  const exe = path.join(root, 'Hermes.exe')
  const files = getBinaries(root, { skip: file => path.resolve(file) === exe })

  assert.equal(files.includes(exe), false)
  assert.equal(files.includes(target), true)
  if (fs.existsSync(link)) {
    assert.equal(files.includes(link), false)
  }
})

test('getBinaries tolerates a missing directory', () => {
  assert.deepEqual(getBinaries(path.join(tmpTree(), 'nope')), [])
})

test('chunk splits into ~100-file batches with no leftovers', () => {
  assert.deepEqual(chunk([], 100), [])
  const two50 = Array.from({ length: 250 }, (_, i) => `f${i}.exe`)
  const batches = chunk(two50)
  assert.equal(batches.length, 3)
  assert.deepEqual(batches.map(b => b.length), [100, 100, 50])
  assert.deepEqual(batches.flat(), two50)

  const exact = Array.from({ length: 200 }, (_, i) => `f${i}.exe`)
  assert.deepEqual(chunk(exact).map(b => b.length), [100, 100])
})

test('customSign skips Store- submission packages (Partner Center signs)', async () => {
  const result = await customSign(
    { path: 'C:/out/Store-HermesBundled-0.28.0-win-x64.msix' },
    { appInfo: { productFilename: 'Hermes' } },
    { signMsix: async () => { throw new Error('must not be called') } }
  )
  assert.equal(result, true)
})

test('customSign delegates only the msix package and root product exe to the Azure signer', async () => {
  /** @type {string[][]} */
  const delegated = []
  /** @type {Parameters<typeof customSign>[2]} */
  const deps = {
    signMsix: async configuration => { delegated.push(['msix', configuration.path]) },
    azureSignFile: async file => { delegated.push(['exe', file]) }
  }
  const output = path.join(tmpTree(), '${os}', '${arch}')
  const info = new Packager({
    projectDir: path.resolve(import.meta.dirname, '..'),
    targets: Platform.WINDOWS.createTarget(['msix'], Arch.x64, Arch.arm64),
    config: {
      ...builderConfig,
      // validateConfig normalizes these file sets in place.
      files: structuredClone(builderConfig.files),
      extraResources: structuredClone(builderConfig.extraResources),
      directories: { output },
      win: { ...builderConfig.win, executableName: 'hermes-collision' }
    }
  })
  await info.validateConfig()
  const packager = new WinPackager(info)
  const msix = path.join(tmpTree(), 'HermesBundled-0.28.0-win-x64.msix')
  assert.equal(await customSign({ path: msix }, packager, deps), true)
  assert.deepEqual(delegated, [['msix', msix]])

  for (const arch of [Arch.x64, Arch.arm64]) {
    const root = packager['computeAppOutDir'](packager.expandMacro(output, Arch[arch]), arch)
    const exeName = `${packager.appInfo.productFilename}.exe`
    const exe = path.join(root, exeName)
    delegated.length = 0
    // Same basename is not enough: payload copies wait for the afterPack batch.
    for (const file of [path.join(root, 'resources', 'agent-payload', 'bin', exeName), path.join(`${root}-other`, exeName), path.join(root, 'resources', 'Hermes-helper.exe')]) {
      assert.equal(await customSign({ path: file }, packager, deps), true)
    }
    assert.deepEqual(delegated, [])
    // Returning true suppresses electron-builder's fallback per-file signer.
    assert.equal(await customSign({ path: exe }, packager, deps), true)
    assert.equal(await customSign({ path: exe.toUpperCase() }, packager, deps), true)
    assert.deepEqual(delegated, [['exe', exe], ['exe', exe.toUpperCase()]])
  }
})

test('batchSignAppTree is a no-op (skipped=true) without the Azure env, and signs via chunked argv-array invocations when set', async () => {
  const root = tmpTree()
  fs.mkdirSync(path.join(root, 'tools'), { recursive: true })
  const exe = path.join(root, 'Hermes.exe')
  fs.writeFileSync(exe, 'x')
  fs.writeFileSync(path.join(root, 'tools', 'node.exe'), 'x')
  fs.writeFileSync(path.join(root, 'tools', 'FFMPEG.DLL'), 'x')
  // The shipped uv-cache holds inert sdist/archive artifacts — the arch
  // audit exempts it and batch-sign must never sign it (wasted Azure
  // round-trips; locally it can even reference files since removed).
  fs.mkdirSync(path.join(root, 'uv-cache', 'archive-v0', 'setuptools'), { recursive: true })
  fs.writeFileSync(path.join(root, 'uv-cache', 'archive-v0', 'setuptools', 'cli.exe'), 'x')

  // Unsigned lane: loud no-op, nothing invoked.
  for (const env of [{}, { AZURE_SIGN_ENDPOINT: 'https://test.invalid' }, { AZURE_SIGN_ENDPOINT: 'https://test.invalid', AZURE_SIGN_ACCOUNT: 'account' }]) {
    assert.deepEqual(await batchSignAppTree(root, exe, { env }), { signed: 0, chunks: 0, skipped: true })
  }

  // Signed lane: product exe excluded, chunked argv arrays. Two passes per
  // chunk: 'sign' (Azure, no timestamp) then 'timestamp' (RFC3161, no dlib).
  const invocations = []
  const fakeExec = (tool, args) => {
    invocations.push({ tool, args })
    return Buffer.from('')
  }
  const result = await batchSignAppTree(root, exe, {
    env: {
      AZURE_SIGN_ENDPOINT: 'https://cus.codesigning.azure.net',
      AZURE_SIGN_ACCOUNT: 'codesign2',
      AZURE_SIGN_PROFILE: 'hermesagent'
    },
    exec: fakeExec,
    mkdtemp: () => root,
    signtool: 'signtool.exe',
    dlib: 'azure.codesigning.dlib.dll'
  })

  assert.deepEqual(result, { signed: 2, chunks: 1, skipped: false })
  assert.equal(invocations.length, 2, 'one sign + one timestamp pass for a single chunk')
  const [signInv, tsInv] = invocations
  assert.equal(signInv.tool, 'signtool.exe')
  assert.ok(Array.isArray(signInv.args), 'argv array, never a shell string')
  assert.equal(signInv.args[0], 'sign')
  assert.equal(tsInv.args[0], 'timestamp')
  const args = signInv.args
  assert.ok(!args.some(arg => typeof arg === 'string' && arg.includes(' ')))
  assert.equal(args.includes(exe), false, 'product exe excluded — signed per-file after rcedit')
  assert.equal(args.includes(path.join(root, 'tools', 'node.exe')), true)
  assert.equal(args.includes(path.join(root, 'tools', 'FFMPEG.DLL')), true)
  assert.equal(args[args.indexOf('/dlib') + 1], 'azure.codesigning.dlib.dll')
  assert.ok(args[args.indexOf('/dmdf') + 1].endsWith('batch-sign.json'))
  assert.equal(args[args.indexOf('/fd') + 1], 'SHA256')
  // Sign pass carries NO timestamp flags — timestamping is the separate pass.
  assert.equal(args.includes('/tr'), false)
  assert.equal(args.includes('/td'), false)
  // Timestamp pass carries /tr + /td but NO dlib / dmdf.
  const tsArgs = tsInv.args
  assert.equal(tsArgs[tsArgs.indexOf('/tr') + 1], 'http://timestamp.digicert.com')
  assert.equal(tsArgs[tsArgs.indexOf('/td') + 1], 'SHA256')
  assert.equal(tsArgs.includes('/dlib'), false)
  assert.equal(tsArgs.includes('/dmdf'), false)
  assert.ok(tsArgs.includes(path.join(root, 'tools', 'node.exe')), 'timestamp pass covers the same files')
  assert.ok(tsArgs.includes(path.join(root, 'tools', 'FFMPEG.DLL')))
})

test('batchSignAppTree chunks large trees into ~100-file signtool invocations, sign + timestamp passes', async () => {
  const root = tmpTree()
  fs.mkdirSync(path.join(root, 'tools'), { recursive: true })
  for (let i = 0; i < 250; i += 1) {
    fs.writeFileSync(path.join(root, 'tools', `bin${i}.exe`), 'x')
  }

  const invocations = []
  const result = await batchSignAppTree(root, path.join(root, 'Hermes.exe'), {
    env: {
      AZURE_SIGN_ENDPOINT: 'https://cus.codesigning.azure.net',
      AZURE_SIGN_ACCOUNT: 'codesign2',
      AZURE_SIGN_PROFILE: 'hermesagent'
    },
    exec: (tool, args) => {
      invocations.push(args)
      return Buffer.from('')
    },
    mkdtemp: () => root,
    signtool: 'signtool.exe',
    dlib: 'azure.codesigning.dlib.dll'
  })

  assert.deepEqual(result, { signed: 250, chunks: 3, skipped: false })
  // 3 chunks × 2 passes (sign + timestamp).
  assert.equal(invocations.length, 6)
  const signBatches = invocations.filter(a => a[0] === 'sign')
  const tsBatches = invocations.filter(a => a[0] === 'timestamp')
  assert.equal(signBatches.length, 3)
  assert.equal(tsBatches.length, 3)
  assert.deepEqual(
    signBatches.map(batch => batch.filter(arg => arg.endsWith('.exe')).length).sort((a, b) => a - b),
    [50, 100, 100]
  )
  const flat = signBatches.flat().filter(arg => arg.endsWith('.exe'))
  assert.equal(new Set(flat).size, 250, 'every binary signed exactly once')
  const tsFlat = tsBatches.flat().filter(arg => arg.endsWith('.exe'))
  assert.equal(new Set(tsFlat).size, 250, 'every binary timestamped exactly once')
  assert.deepEqual(new Set(tsFlat), new Set(flat), 'timestamp pass covers the same files as the sign pass')
})

test('sign chunks run concurrently, capped at the configured concurrency', async () => {
  const root = tmpTree()
  fs.mkdirSync(path.join(root, 'tools'), { recursive: true })
  for (let i = 0; i < 12; i += 1) {
    fs.writeFileSync(path.join(root, 'tools', `bin${i}.exe`), 'x')
  }

  // 12 files at chunk size 1 → 12 chunks, each a sign + timestamp pass. The
  // fake holds its "child" open across an await so the pool's concurrency is
  // actually observable (a sync fake would report max 1 by construction).
  let inFlight = 0
  let maxInFlight = 0
  const fakeExec = async () => {
    inFlight += 1
    maxInFlight = Math.max(maxInFlight, inFlight)
    await new Promise(resolve => setTimeout(resolve, 5))
    inFlight -= 1
  }

  await batchSignAppTree(root, path.join(root, 'Hermes.exe'), {
    env: {
      AZURE_SIGN_ENDPOINT: 'https://cus.codesigning.azure.net',
      AZURE_SIGN_ACCOUNT: 'codesign2',
      AZURE_SIGN_PROFILE: 'hermesagent'
    },
    exec: fakeExec,
    mkdtemp: () => root,
    signtool: 'signtool.exe',
    dlib: 'azure.codesigning.dlib.dll',
    chunkSize: 1,
    concurrency: 3
  })

  assert.equal(maxInFlight, 3, 'never more than the configured concurrency in flight')
})

test.each([false, true])('timestamp retry is bounded; permanent failure=%s', async permanent => {
  const root = tmpTree()
  fs.mkdirSync(path.join(root, 'tools'), { recursive: true })
  fs.writeFileSync(path.join(root, 'tools', 'node.exe'), 'x')

  let calls = 0
  const fakeExec = (tool, args) => {
    if (args[0] === 'sign') return
    calls += 1
    if (permanent || calls < 3) throw new Error('Invalid Time Stamp Request Length:-1')
  }

  const signing = batchSignAppTree(root, path.join(root, 'Hermes.exe'), {
    env: {
      AZURE_SIGN_ENDPOINT: 'https://cus.codesigning.azure.net',
      AZURE_SIGN_ACCOUNT: 'codesign2',
      AZURE_SIGN_PROFILE: 'hermesagent'
    },
    exec: fakeExec,
    mkdtemp: () => root,
    signtool: 'signtool.exe',
    dlib: 'azure.codesigning.dlib.dll',
    timestampRetryDelayMs: 0
  })

  if (permanent) await assert.rejects(signing, /Invalid Time Stamp/); else await signing
  assert.equal(calls, 3, 'timestamp pass retried until the server succeeded')
})

test('batch signing passes its paired toolchain and runtime to both child phases', async () => {
  const root = tmpTree()
  fs.writeFileSync(path.join(root, 'node.exe'), 'x')
  const tools = {
    signtool: path.join(root, 'SDK with spaces', 'signtool.exe'),
    dlib: path.join(root, 'ATS with spaces', 'Azure.CodeSigning.Dlib.dll'),
    dotnetRoot: path.join(root, 'dotnet'),
  }
  const invocations = []
  const result = await batchSignAppTree(root, path.join(root, 'Hermes.exe'), {
    ...tools,
    env: {
      AZURE_SIGN_ENDPOINT: 'https://test.invalid',
      AZURE_SIGN_ACCOUNT: 'account', AZURE_SIGN_PROFILE: 'profile',
      DOTNET_ROOT: 'unrelated machine runtime', TEMP: root,
    },
    exec: (tool, args, options) => { invocations.push({ tool, args, options }) },
    cache: null,
  })
  assert.equal(result.signed, 1)
  assert.deepEqual(invocations.map(call => call.args[0]), ['sign', 'timestamp'])
  for (const call of invocations) {
    assert.equal(call.tool, tools.signtool)
    assert.equal(call.options.env.DOTNET_ROOT, tools.dotnetRoot)
  }
  const [sign, timestamp] = invocations
  assert.equal(sign.args[sign.args.indexOf('/dlib') + 1], tools.dlib)
  assert.equal(timestamp.args.includes('/dlib'), false)
})

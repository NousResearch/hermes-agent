import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { afterEach, test } from 'vitest'
import { stageGetWindows, stageNodePtyInto } from './stage-native-deps.mjs'
import { runDesktopTypecheck, typecheckNodeOptions } from './typecheck.mjs'

const repo = resolve(import.meta.dirname, '../../..')
const roots = []
const FLOOR_BYTES = 4096 * 1024 * 1024

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function put(file, text) {
  mkdirSync(dirname(file), { recursive: true })
  writeFileSync(file, text)
}

function makeProbe(root) {
  const probe = join(root, 'sp  ace', 'tsc-heap-probe.cjs')
  const log = join(root, 'tsc-heap.jsonl')
  put(
    probe,
    [
      "const fs = require('node:fs')",
      "const v8 = require('node:v8')",
      "if (/(?:^|[/\\\\])tsc(?:\\.js)?$/.test(process.argv[1] ?? '')) {",
      '  fs.appendFileSync(',
      '    process.env.HERMES_TSC_HEAP_LOG,',
      '    `${JSON.stringify({ args: process.argv.slice(2), heapSizeLimit: v8.getHeapStatistics().heap_size_limit })}\\n`,',
      '  )',
      '}',
      '',
    ].join('\n'),
  )
  return { probe, log }
}

function readProbeLog(log) {
  if (!existsSync(log)) return []
  return readFileSync(log, 'utf8').trim().split('\n').filter(Boolean).map(line => JSON.parse(line))
}

function stageTypecheckFixture() {
  const root = mkdtempSync(join(tmpdir(), 'desktop-typecheck-'))
  roots.push(root)
  const source = join(root, 'source')
  const app = join(source, 'apps/desktop')
  cpSync(join(repo, 'apps/desktop/package.json'), join(app, 'package.json'))
  cpSync(join(repo, 'scripts/build/frontend-common.mjs'), join(source, 'scripts/build/frontend-common.mjs'))
  if (existsSync(join(repo, 'apps/desktop/scripts/typecheck.mjs'))) {
    cpSync(join(repo, 'apps/desktop/scripts/typecheck.mjs'), join(app, 'scripts/typecheck.mjs'))
  }
  symlinkSync(join(repo, 'node_modules'), join(app, 'node_modules'), 'junction')
  const baseTsconfig = { compilerOptions: { noEmit: true, skipLibCheck: true, types: [] } }
  put(join(app, 'tsconfig.json'), JSON.stringify({ ...baseTsconfig, include: ['src/*.ts'] }))
  put(join(app, 'tsconfig.electron.json'), JSON.stringify({ extends: './tsconfig.json', include: ['electron/*.ts'] }))
  put(join(app, 'tsconfig.e2e.json'), JSON.stringify({ extends: './tsconfig.json', include: ['e2e/*.ts'] }))
  put(join(app, 'src/ok.ts'), 'export const rendererOk: string = "ok"\n')
  put(join(app, 'electron/ok.ts'), 'export const electronOk: string = "ok"\n')
  put(join(app, 'e2e/ok.ts'), 'export const e2eOk: string = "ok"\n')
  put(join(app, 'electron-builder.config.cjs'), '/** @type {{ appId: string }} */\nmodule.exports = { appId: "com.nousresearch.hermes" }\n')
  return { root, source, app }
}

test('npm run typecheck raises every tsc child above the 8 GB default heap while preserving inherited NODE_OPTIONS (#144459)', () => {
  const { root, app } = stageTypecheckFixture()
  const { probe, log } = makeProbe(root)
  const windows = process.platform === 'win32'
  const runNpmTypecheck = (nodeOptions) =>
    spawnSync('npm', ['run', 'typecheck', '--silent', '--ignore-scripts'], {
      cwd: app,
      encoding: 'utf8',
      shell: windows,
      env: { ...process.env, HERMES_TSC_HEAP_LOG: log, NODE_OPTIONS: nodeOptions },
    })

  // Simulate an 8 GB host's ~2 GiB ceiling plus a quoted preload path containing multiple spaces.
  const ok = runNpmTypecheck(`--require=${JSON.stringify(probe)} --max-old-space-size=2048`)
  assert.equal(ok.status, 0, ok.stderr || ok.stdout)
  const runs = readProbeLog(log)
  assert.equal(runs.length, 4)
  for (const entry of runs) {
    assert.ok(entry.heapSizeLimit >= FLOOR_BYTES, `expected >= 4096 MiB heap, got ${entry.heapSizeLimit}`)
  }

  // An explicit caller ceiling higher than the floor is respected rather than downgraded.
  rmSync(log, { force: true })
  const raised = runNpmTypecheck(`--require=${JSON.stringify(probe)} --max-old-space-size=8192`)
  assert.equal(raised.status, 0, raised.stderr || raised.stdout)
  for (const entry of readProbeLog(log)) {
    assert.ok(entry.heapSizeLimit >= 8192 * 1024 * 1024, `expected >= 8192 MiB heap, got ${entry.heapSizeLimit}`)
  }

  // A type error in an intermediate config fails the command and short-circuits later steps.
  rmSync(log, { force: true })
  put(join(app, 'electron/ok.ts'), 'export const broken: string = 123\n')
  const failed = runNpmTypecheck(`--require=${JSON.stringify(probe)} --max-old-space-size=2048`)
  assert.notEqual(failed.status, 0)
  assert.equal(readProbeLog(log).length, 2)
})

test('buildDesktop({ typecheck: true }) runs renderer tsc with the raised V8 heap ceiling (#144459)', async () => {
  const { buildDesktop } = await import('../../../scripts/build/desktop.mjs')
  const { root, source, app } = stageTypecheckFixture()
  const { probe, log } = makeProbe(root)
  put(join(app, 'vite.config.mjs'), 'export default { base: "./", build: { minify: false } }\n')
  put(join(app, 'index.html'), '<html><div id="app"></div><script type="module" src="/src/index.js"></script></html>\n')
  put(join(app, 'src/index.js'), 'document.getElementById("app").textContent = "built renderer"\n')
  put(join(app, 'electron/entry.ts'), "await import('./main')\n")
  put(join(app, 'electron/main.ts'), 'console.log(JSON.stringify({ stamp: __HERMES_INSTALL_STAMP__, identity: __HERMES_PRODUCT_IDENTITY__ }))\n')
  put(join(app, 'electron/preload.ts'), 'globalThis.fixturePreload = "compiled preload"\n')
  put(join(app, 'electron/preview-guest-preload-entry.ts'), 'globalThis.fixtureGuestPreload = "compiled guest preload"\n')
  cpSync(join(repo, 'apps/desktop/product-identity.cjs'), join(app, 'product-identity.cjs'))
  cpSync(join(repo, 'apps/desktop/electron/native'), join(app, 'electron/native'), { recursive: true })
  for (const helper of ['scripts/msix-shared.mjs', 'scripts/release-content-types.json']) {
    cpSync(join(repo, helper), join(source, helper))
  }
  const icons = join(root, 'icons')
  put(join(icons, 'apps/desktop/public/apple-touch-icon.png'), 'fresh icon')
  put(join(app, 'public/apple-touch-icon.png'), 'stale icon')
  const nativeDeps = join(root, 'native')
  stageNodePtyInto(join(repo, 'node_modules/node-pty'), join(nativeDeps, 'node-pty'))
  stageGetWindows({ source: repo, out: nativeDeps })
  const stamp = join(root, 'install-stamp.json')
  put(stamp, JSON.stringify({
    schemaVersion: 1,
    payload: 'light',
    updateMechanism: 'external',
    commit: 'a'.repeat(40),
    tag: 'v1.2.3',
    builtAt: '2026-01-01T00:00:00.000Z',
  }))

  const prevNodeOptions = process.env.NODE_OPTIONS
  const prevLog = process.env.HERMES_TSC_HEAP_LOG
  process.env.NODE_OPTIONS = `--require=${JSON.stringify(probe)} --max-old-space-size=2048`
  process.env.HERMES_TSC_HEAP_LOG = log
  try {
    await buildDesktop({ source, out: join(root, 'out'), icons, nativeDeps, stamp, typecheck: true })
  } finally {
    if (prevNodeOptions === undefined) delete process.env.NODE_OPTIONS
    else process.env.NODE_OPTIONS = prevNodeOptions
    if (prevLog === undefined) delete process.env.HERMES_TSC_HEAP_LOG
    else process.env.HERMES_TSC_HEAP_LOG = prevLog
  }

  const runs = readProbeLog(log)
  assert.equal(runs.length, 1)
  assert.ok(runs[0].heapSizeLimit >= FLOOR_BYTES, `expected >= 4096 MiB heap, got ${runs[0].heapSizeLimit}`)
})

test('typecheckNodeOptions preserves quoted flags and higher ceilings while runDesktopTypecheck fails on signal aborts', () => {
  assert.equal(typecheckNodeOptions(''), '--max-old-space-size=4096')
  assert.equal(typecheckNodeOptions('--max-old-space-size=2048'), '--max-old-space-size=2048 --max-old-space-size=4096')
  assert.equal(
    typecheckNodeOptions('--max-old-space-size=8192 --max-old-space-size=2048'),
    '--max-old-space-size=8192 --max-old-space-size=2048 --max-old-space-size=4096',
  )
  assert.equal(typecheckNodeOptions('--max-old-space-size=8192'), '--max-old-space-size=8192')
  const quoted = `--require=${JSON.stringify(join(tmpdir(), 'sp  ace', 'preload.cjs'))}`
  assert.ok(typecheckNodeOptions(quoted).startsWith(quoted))
  assert.match(typecheckNodeOptions(quoted), /--max-old-space-size=4096$/)

  const { source } = stageTypecheckFixture()
  let calls = 0
  const status = runDesktopTypecheck({
    source,
    env: { ...process.env, NODE_OPTIONS: '' },
    spawn: () => {
      calls++
      return { status: null, signal: 'SIGABRT' }
    },
  })
  assert.equal(status, 1)
  assert.equal(calls, 1)
})

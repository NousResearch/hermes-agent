import assert from 'node:assert/strict'
import { execFileSync, spawnSync } from 'node:child_process'
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'

import { macosSysroot, xcrunClangArgv } from '../../scripts/macos-sysroot.mjs'

// Deliberately opt-in: this opens two isolated native windows, never the app,
// gateway or installed profile. Other hosts cannot certify Cocoa hit testing.
if (process.platform !== 'darwin') {
  console.log('SKIP titlebar native clicks: macOS Cocoa/WindowServer required; no native verification performed')
  process.exit(0)
}

const require = createRequire(import.meta.url)
const { build } = await import('esbuild')
const electron = require('electron')
const fixture = import.meta.dirname
const desktop = resolve(fixture, '../..')
const rootModule = join(desktop, 'src/components/pane-shell/tree/renderer/index.tsx')
const seam = join(fixture, 'shell-seam.mjs')
const headers = process.env.NODE_INCLUDE_DIR ?? resolve(dirname(realpathSync(process.execPath)), '../include/node')
assert.ok(existsSync(join(headers, 'node_api.h')), 'Use a Node distribution with include/node, or set NODE_INCLUDE_DIR')
const runtime = mkdtempSync(join(tmpdir(), 'hermes-titlebar-native-'))
const assets = join(runtime, 'assets')
mkdirSync(assets)
console.log(`Native titlebar artifacts: ${runtime}`)

execFileSync(
  'xcrun',
  [
    ...xcrunClangArgv(macosSysroot()),
    '-x',
    'objective-c++',
    '-fobjc-arc',
    '-bundle',
    '-undefined',
    'dynamic_lookup',
    '-framework',
    'Cocoa',
    '-I',
    headers,
    join(fixture, 'native-pointer.mm'),
    '-o',
    join(assets, 'native-pointer.node')
  ],
  { stdio: 'pipe' }
)

const shellImports = new Set([
  '../../edit-mode',
  '../../geometry',
  '../store',
  '../tab-key-hint-state',
  '../zone-editor',
  './edit-bar',
  './floating-panes',
  './keep-alive-panes',
  './narrow-overlays',
  './tree-node'
])
await build({
  entryPoints: [join(fixture, 'renderer.mjs')],
  outfile: join(assets, 'renderer.js'),
  bundle: true,
  format: 'iife',
  jsx: 'automatic',
  alias: { '@': join(desktop, 'src') },
  define: { 'process.env.NODE_ENV': '"production"' },
  plugins: [
    {
      name: 'isolate-unrelated-shell-services',
      setup(builder) {
        builder.onResolve({ filter: /.*/ }, args => {
          if ((args.importer === rootModule && shellImports.has(args.path)) || args.path === '@/store/session') {
            return { path: seam }
          }
        })
      }
    }
  ]
})
copyFileSync(join(fixture, 'fixture.html'), join(assets, 'fixture.html'))

const runs = []
for (let run = 1; run <= 2; run++) {
  const directory = join(runtime, `run-${run}`)
  mkdirSync(directory)
  mkdirSync(join(directory, 'home'))
  for (const name of ['native-pointer.node', 'renderer.js', 'fixture.html']) {
    copyFileSync(join(assets, name), join(directory, name))
  }

  const env = {
    ...process.env,
    HOME: join(directory, 'home'),
    HERMES_HOME: join(directory, 'home/hermes'),
    XDG_CONFIG_HOME: join(directory, 'home/config'),
    XDG_CACHE_HOME: join(directory, 'home/cache')
  }
  delete env.ELECTRON_RUN_AS_NODE
  const result = spawnSync(electron, [join(fixture, 'main.cjs'), directory], {
    env,
    encoding: 'utf8',
    timeout: 45_000
  })
  writeFileSync(join(directory, 'stdout.log'), result.stdout ?? '')
  writeFileSync(join(directory, 'stderr.log'), result.stderr ?? '')
  assert.equal(result.status, 0, `Native run ${run} failed (${directory}): ${result.error ?? result.stderr}`)
  const data = JSON.parse(readFileSync(join(directory, 'results.json'), 'utf8'))
  runs.push({
    run,
    swaps: data.swaps,
    successfulNativeClicks: data.successfulNativeClicks,
    negativeControl: data.negativeControl,
    focusAndCursorPreserved: data.focusAndCursorPreserved
  })
}

const summary = { status: 'passed', rootModule, artifacts: runtime, runs }
writeFileSync(join(runtime, 'summary.json'), JSON.stringify(summary, null, 2))
console.log(JSON.stringify(summary, null, 2))

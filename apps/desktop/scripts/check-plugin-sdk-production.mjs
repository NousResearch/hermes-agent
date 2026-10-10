// Run from any directory: node apps/desktop/scripts/check-plugin-sdk-production.mjs
// Requires the installed desktop Vite and Playwright dependencies and Chromium.
// --baseline substitutes only the old eager namespace capture, in memory.
// --channel=chrome (or msedge) uses an installed browser without downloading one.
// --screenshot emits a PNG data URL to stdout before deleting all temp artifacts.
import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

export const app = fileURLToPath(new URL('../', import.meta.url))
export const runtime = fileURLToPath(new URL('../src/sdk/runtime.ts', import.meta.url))

// Portable pre-fix control. Never used for the positive test: that loads runtime.ts.
// This preserves the old module-scope read and shim construction without Git history.
export const eagerBaseline = `
import * as sdk from './index'
import * as React from 'react'
import * as jsxRuntime from 'react/jsx-runtime'
import * as jsxDevRuntime from 'react/jsx-dev-runtime'
const GLOBALS = {
  __HERMES_PLUGIN_SDK__: sdk, __HERMES_REACT__: React,
  __HERMES_REACT_JSX__: jsxRuntime, __HERMES_REACT_JSX_DEV__: jsxDevRuntime
}
export function installPluginSdk() { Object.assign(globalThis, GLOBALS) }
function shimUrl(key) {
  const names = Object.keys(GLOBALS[key]).filter(name =>
    name !== 'default' && /^[A-Za-z_$][\\w$]*$/.test(name))
  const source = 'const m = globalThis.' + key + ';\\n' +
    'export default m.default ?? m;\\n' +
    (names.length ? 'export const { ' + names.join(', ') + ' } = m;\\n' : '')
  return URL.createObjectURL(new Blob([source], { type: 'text/javascript' }))
}
let cached
export function sdkImportMap() {
  return cached ??= {
    '@hermes/plugin-sdk': shimUrl('__HERMES_PLUGIN_SDK__'),
    'react/jsx-dev-runtime': shimUrl('__HERMES_REACT_JSX_DEV__'),
    'react/jsx-runtime': shimUrl('__HERMES_REACT_JSX__'), react: shimUrl('__HERMES_REACT__')
  }
}
`

// A second entry shares the real app graph/chunks, but does not mount the app,
// discover disk plugins, or connect to a backend. No SDK/React/loader mocks.
const probe = `
import * as sdk from '@/sdk/index'
import * as React from 'react'
import * as jsx from 'react/jsx-runtime'
import * as jsxDev from 'react/jsx-dev-runtime'
import { createRoot } from 'react-dom/client'
import { installPluginSdk, sdkImportMap } from '@/sdk/runtime'
import { loadRuntimePlugin, unloadRuntimePlugin } from '@/contrib/runtime-loader'
import { $pluginRecords, dropPlugin } from '@/contrib/plugins-store'
import { registry } from '@/contrib/registry'
globalThis.sdkSmoke = async () => {
  const check = (ok, message) => { if (!ok) throw new Error(message) }
  const receipt = []
  const area = 'sdk-smoke-' + crypto.randomUUID()
  const ids = []
  const sources = [
    ['', 'null'],
    ["import { host } from '@hermes/plugin-sdk'", 'host'],
    ["import React from 'react'; import { jsx } from 'react/jsx-runtime'; import { jsxDEV } from 'react/jsx-dev-runtime'",
      '{ React, jsx, jsxDEV, node: jsx("strong", { children: "SDK React JSX registered" }) }']
  ]
  try {
    for (const [imports, value] of sources) {
      const id = crypto.randomUUID()
      ids.push(id)
      const source = imports + '\\nexport default { id: ' + JSON.stringify(id) +
        ', register(ctx) { const data = { value: ' + value + ', disposed: false }; ' +
        'ctx.onDispose(() => { data.disposed = true }); ctx.register({ id: "entry", area: ' + JSON.stringify(area) +
        ', data }) } }'
      check(await loadRuntimePlugin(source, id) === id,
        'plugin load failed: ' + ($pluginRecords.get()[id]?.error ?? id))
      check($pluginRecords.get()[id]?.status === 'loaded', 'plugin did not register')
      const count = registry.getArea(area).length
      const previous = registry.getArea(area).find(item => item.id === id + ':entry').data
      const map = sdkImportMap()
      check(await loadRuntimePlugin(source, id) === id, 'plugin reload failed')
      check(registry.getArea(area).length === count, 'reload duplicated contributions')
      check(previous.disposed, 'reload did not dispose previous plugin')
      check(sdkImportMap() === map, 'shim cache changed on reload')
    }
    const values = registry.getArea(area).map(item => item.data.value)
    check(values.length === sources.length, 'missing registrations')
    check(values[1] === sdk.host, 'SDK named import lost host identity')
    const reactValue = values[2]
    check(reactValue.React === (React.default ?? React), 'React default is not the app singleton')
    check(reactValue.jsx === jsx.jsx && reactValue.jsxDEV === jsxDev.jsxDEV, 'JSX runtime identity mismatch')
    check(React.isValidElement(reactValue.node), 'JSX did not create a React element')
    check(globalThis.__HERMES_PLUGIN_SDK__ === sdk, 'SDK namespace identity mismatch')
    const map = sdkImportMap()
    for (const [specifier, key, namespace] of [
      ['@hermes/plugin-sdk', '__HERMES_PLUGIN_SDK__', sdk],
      ['react', '__HERMES_REACT__', React],
      ['react/jsx-runtime', '__HERMES_REACT_JSX__', jsx],
      ['react/jsx-dev-runtime', '__HERMES_REACT_JSX_DEV__', jsxDev]
    ]) {
      const installed = globalThis[key]
      installPluginSdk()
      check(globalThis[key] === installed, specifier + ' namespace changed on reinstall')
      const shim = await import(/* @vite-ignore */ map[specifier])
      check(shim.default === (installed.default ?? installed), specifier + ' default identity mismatch')
      // CJS interop may create separate namespace wrappers across chunks.
      // The underlying React singleton and every exported value must be shared.
      for (const name of Object.keys(namespace).filter(name => name !== 'default' && /^[A-Za-z_$][\\w$]*$/.test(name))) {
        check(installed[name] === namespace[name], specifier + ' installed identity mismatch: ' + name)
        check(shim[name] === namespace[name], specifier + ' named identity mismatch: ' + name)
      }
    }
    check(sdkImportMap() === map, 'reinstall invalidated shim cache')
    receipt.push('no-import, SDK named import and React JSX plugins registered',
      'reload disposes previous contributions; shim map is stable',
      'SDK, React and both JSX namespaces retain all export identities')
    for (const [imports, expected] of [
      ["import 'sdk-smoke-unsupported-package'", /unsupported import/],
      ["import { __sdkSmokeMissingExport } from '@hermes/plugin-sdk'", /does not provide an export/]
    ]) {
      const id = crypto.randomUUID()
      ids.push(id)
      const source = imports + '\\nexport default { id: ' + JSON.stringify(id) +
        ', register() { throw new Error("invalid plugin evaluated") } }'
      check(await loadRuntimePlugin(source, id) === null, 'invalid import was accepted')
      const record = $pluginRecords.get()[id]
      check(record?.status === 'error' && expected.test(record.error),
        'wrong import error: ' + record?.error)
    }
    receipt.push('unsupported package and missing named export remain errors')
    const root = createRoot(document.getElementById('rendered'))
    root.render(reactValue.node)
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))
    check(document.getElementById('rendered').textContent === 'SDK React JSX registered', 'React render failed')
    receipt.push('React JSX rendered in Chromium')
    document.getElementById('receipt').textContent = receipt.join('\\n')
    return receipt
  } finally {
    for (const id of ids) { unloadRuntimePlugin(id); dropPlugin(id) }
    check(registry.getArea(area).length === 0, 'unload leaked contributions')
  }
}
`

async function main() {
  const args = process.argv.slice(2)
  assert(args.every(arg => ['--baseline', '--screenshot', '--channel=chrome', '--channel=msedge'].includes(arg)), 'Unknown argument')
  const baseline = args.includes('--baseline')
  const channel = args.find(arg => arg.startsWith('--channel='))?.split('=')[1]
  const { build, preview } = await import('vite')
  const { chromium } = await import('@playwright/test')
  const scratch = await mkdtemp(path.join(tmpdir(), 'hermes-sdk-production-'))
  let server
  let browser
  const oldDirname = globalThis.__dirname
  try {
    // The runner loads the existing TS config without writing a config bundle
    // into node_modules (which may be shared by several worktrees).
    globalThis.__dirname = app
    const result = await build({
      root: app,
      configFile: path.join(app, 'vite.config.ts'),
      configLoader: 'runner',
      envDir: false,
      logLevel: 'warn',
      cacheDir: path.join(scratch, 'cache'),
      plugins: [{
        name: 'sdk-production-probe',
        resolveId(id) { if (id === 'virtual:sdk-smoke') return '\0sdk-smoke' },
        load(id) {
          if (id === '\0sdk-smoke') return probe
          if (baseline && id.endsWith('/sdk/runtime.ts') && path.normalize(id) === runtime) return eagerBaseline
        }
      }],
      build: {
        outDir: path.join(scratch, 'dist'),
        emptyOutDir: true,
        rolldownOptions: { input: { app: path.join(app, 'index.html'), smoke: 'virtual:sdk-smoke' } }
      }
    })
    const outputs = (Array.isArray(result) ? result : [result]).flatMap(result => result.output)
    const modules = new Set(outputs.filter(output => output.type === 'chunk').flatMap(output => Object.keys(output.modules)))
    assert([...modules].some(id => path.normalize(id) === runtime), 'Actual runtime module missing from app graph')
    console.log(`Production build: ${modules.size} modules; ${baseline ? 'old eager control' : 'actual runtime.ts'}`)
    const entry = outputs.find(output => output.type === 'chunk' && output.isEntry && output.name === 'smoke')
    assert(entry, 'Smoke entry was not emitted')
    await writeFile(path.join(scratch, 'dist', 'sdk-smoke.html'), `<!doctype html>
      <html><head><meta charset="utf-8"><title>SDK production smoke</title></head>
      <body><h1>SDK production smoke</h1><div id="rendered"></div><pre id="receipt"></pre>
      <script type="module" src="./${entry.fileName}"></script></body></html>`)
    server = await preview({
      root: app, configFile: false, envDir: false,
      build: { outDir: path.join(scratch, 'dist') },
      preview: { host: '127.0.0.1', port: 0, open: false }
    })
    const address = server.httpServer.address()
    const origin = `http://127.0.0.1:${address.port}`
    browser = await chromium.launch({ headless: true, channel })
    const page = await browser.newPage({ viewport: { width: 1100, height: 600 }, serviceWorkers: 'block' })
    // Only this generated preview is reachable; no backend, CDN or user data.
    await page.route('**/*', route => {
      const url = new URL(route.request().url())
      return url.origin === origin || ['blob:', 'data:'].includes(url.protocol)
        ? route.continue() : route.abort()
    })
    await page.routeWebSocket('**/*', socket => socket.close())
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    await page.goto(`${origin}/sdk-smoke.html`, { waitUntil: 'load', timeout: 120_000 })
    await page.waitForFunction(() => typeof globalThis.sdkSmoke === 'function')
    const receipt = await page.evaluate(() => globalThis.sdkSmoke())
    assert.deepEqual(errors, [], 'Uncaught browser errors')
    // Screenshots are temporary like the build. The textual receipt survives
    // in stdout; no files are retained in the checkout or personal directories.
    const screenshot = await page.screenshot({ path: path.join(scratch, 'sdk-smoke.png'), fullPage: true })
    console.log(JSON.stringify({ control: baseline ? 'old-eager' : 'current-runtime',
      browser: browser.version(), receipt, pageErrors: errors, screenshot: 'captured (temporary)' }, null, 2))
    if (args.includes('--screenshot')) console.log('SDK_SMOKE_SCREENSHOT=data:image/png;base64,' + screenshot.toString('base64'))
  } finally {
    await browser?.close()
    await server?.close()
    if (oldDirname === undefined) delete globalThis.__dirname
    else globalThis.__dirname = oldDirname
    // scratch is exactly the newly allocated temp directory owned by this run.
    await rm(scratch, { recursive: true, force: true })
  }
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  main().catch(error => { console.error(error); process.exitCode = 1 })
}

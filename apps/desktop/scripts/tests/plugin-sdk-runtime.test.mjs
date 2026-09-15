import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { test } from 'node:test'
import { pathToFileURL } from 'node:url'
import vm from 'node:vm'
import { eagerBaseline, runtime } from '../check-plugin-sdk-production.mjs'

// Use Vite's bundler even when the package manager does not hoist it.
const requireFromVite = createRequire(import.meta.resolve('vite'))
const { rolldown } = await import(pathToFileURL(requireFromVite.resolve('rolldown')).href)

// Deliberate RED: node apps/desktop/scripts/tests/plugin-sdk-runtime.test.mjs --baseline
// Normal acceptance: node --test apps/desktop/scripts/tests/plugin-sdk-runtime.test.mjs

async function bundledRuntime(baseline) {
  const bundle = await rolldown({
    input: 'fixture:entry',
    plugins: [{
      name: 'sdk-cycle-fixture',
      resolveId(id, importer) {
        if (id.startsWith('fixture:')) return '\0' + id
        if (id === './index' && importer === runtime) return '\0fixture:sdk'
        if (id === 'runtime') return runtime
        if (id.startsWith('react')) return '\0fixture:' + id
      },
      load(id) {
        if (id === runtime && baseline) return eagerBaseline
        if (id === '\0fixture:entry') return `
          import * as sdk from 'fixture:sdk';
          import { installPluginSdk, sdkImportMap } from 'runtime';
          export { sdk, installPluginSdk, sdkImportMap };
        `
        // The barrel enters runtime before its namespace is initialized, as the
        // app's sdk -> contrib -> runtime-loader -> runtime cycle does.
        if (id === '\0fixture:sdk') return `
          export { installPluginSdk, sdkImportMap } from 'runtime';
          export const marker = {};
        `
        if (id.startsWith('\0fixture:react')) return 'export const marker = {}'
      }
    }]
  })
  try {
    const { output } = await bundle.generate({ format: 'iife', name: 'fixture', minify: true })
    const urls = new Map()
    const context = vm.createContext({ Blob, URL: {
      createObjectURL(blob) {
        const url = `blob:fixture/${urls.size}`
        urls.set(url, blob)
        return url
      }
    } })
    vm.runInContext(output[0].code, context)
    return { context, urls, ...context.fixture }
  } finally {
    await bundle.close()
  }
}

async function evaluateShim(blob, context) {
  const source = await blob.text()
  const bundle = await rolldown({
    input: 'shim',
    plugins: [{
      name: 'generated-shim',
      resolveId(id) { if (id === 'shim') return '\0shim' },
      load(id) { if (id === '\0shim') return source }
    }]
  })
  try {
    const { output } = await bundle.generate({ format: 'iife', name: 'shim', exports: 'named' })
    vm.runInContext(output[0].code, context)
    return context.shim
  } finally {
    await bundle.close()
  }
}

test('bundled SDK resolves initialized namespaces and reuses its shim map', async () => {
  const fixture = await bundledRuntime(process.argv.includes('--baseline'))
  fixture.installPluginSdk()
  const map = fixture.sdkImportMap()
  assert.equal(fixture.context.__HERMES_PLUGIN_SDK__, fixture.sdk)
  assert.equal(fixture.context.__HERMES_PLUGIN_SDK__.marker, fixture.sdk.marker)
  fixture.installPluginSdk()
  assert.equal(fixture.sdkImportMap(), map)
  assert.equal(fixture.urls.size, Object.keys(map).length)
  for (const [specifier, key] of [
    ['@hermes/plugin-sdk', '__HERMES_PLUGIN_SDK__'],
    ['react', '__HERMES_REACT__'],
    ['react/jsx-runtime', '__HERMES_REACT_JSX__'],
    ['react/jsx-dev-runtime', '__HERMES_REACT_JSX_DEV__']
  ]) {
    const namespace = fixture.context[key]
    const shim = await evaluateShim(fixture.urls.get(map[specifier]), fixture.context)
    assert.equal(shim.default, namespace)
    for (const name of Object.keys(namespace)) assert.equal(shim[name], namespace[name])
  }
})

test('old eager capture fails in the same bundled cycle before plugin evaluation', async () => {
  const fixture = await bundledRuntime(true)
  fixture.installPluginSdk()
  assert.equal(fixture.context.__HERMES_PLUGIN_SDK__, undefined)
  assert.throws(() => fixture.sdkImportMap(), {
    name: 'TypeError', message: 'Cannot convert undefined or null to object'
  })
  assert.equal(fixture.urls.size, 0)
})

#!/usr/bin/env node
// bundle-electron-main.mjs — bundles electron/main.ts and electron/preload.ts
// into self-contained js files in dist/ so the packaged app doesn't need
// node_modules/ or tsx at runtime.
//
// Output:
//   dist/electron-main.mjs    (MJS bundle — entry point for packaged app)
//   dist/electron-preload.js (CJS bundle — loaded via BrowserWindow preload)
//
// `electron` and `node-pty` are external (provided by the runtime / staged
// separately via stage-native-deps).
import { build } from 'esbuild'
import { createRequire } from 'node:module'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { mkdirSync, readFileSync } from 'node:fs'

const require = createRequire(import.meta.url)

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
const distDir = resolve(root, 'dist')
mkdirSync(distDir, { recursive: true })

const mainEntry = resolve(root, 'electron/main.ts')
const mainOut = resolve(distDir, 'electron-main.mjs')
const preloadEntry = resolve(root, 'electron/preload.ts')
const preloadOut = resolve(distDir, 'electron-preload.js')

const external = ['electron', 'node-pty', 'get-windows', 'fs']
// Production bundles bake packaged=true so unpackaged `electron .` still
// behaves like a packaged build. Dev bundles (`--dev`) leave the env alone
// so HERMES_DESKTOP_DEV_SERVER / source-tree resolution keep working.
const isDev = process.argv.includes('--dev')

// The install stamp is baked INTO the bundle: the define below sets the
// __HERMES_INSTALL_STAMP__ global to the stamp OBJECT literal (the JSON
// text is a valid JS expression, so no string round-trip). A baked
// constant cannot be missing, stale, or edited after signing.
// `npm run build` writes build/install-stamp.json immediately before this
// script runs, so a missing file here is a broken build, not a thin one.
// Dev bundles bake nothing — install-stamp.ts's typeof guard yields null,
// because a dev run has no artifact to be truthful about (a stale stamp
// from a previous build would lie about provenance).
function bakedInstallStamp() {
  const raw = readFileSync(resolve(root, 'build/install-stamp.json'), 'utf8')
  JSON.parse(raw) // fail the build on malformed output, not first launch
  return raw
}

const define = isDev
  ? {}
  : {
      'process.env.HERMES_DESKTOP_IS_PACKAGED': JSON.stringify(true),
      '__HERMES_INSTALL_STAMP__': bakedInstallStamp(),
      // The product identity (name object, appId, deep-link scheme)
      // baked the same way electron-builder gets it
      '__HERMES_PRODUCT_IDENTITY__': JSON.stringify(require('../product-identity.cjs')),
    }

// Bundle main.ts → dist/electron-main.mjs
await build({
  entryPoints: [mainEntry],
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node20',
  outfile: mainOut,
  external,
  banner: {
    js: "import { createRequire } from 'module'; const require = createRequire(import.meta.url);",
  },
  define,
  logLevel: 'info',
})
console.log(`bundled ${mainOut}${isDev ? ' (dev)' : ''}`)

// Bundle preload.ts → dist/electron-preload.js
await build({
  entryPoints: [preloadEntry],
  bundle: true,
  platform: 'node',
  format: 'cjs',
  target: 'node20',
  outfile: preloadOut,
  external,
  define,
  logLevel: 'info',
})
console.log(`bundled ${preloadOut}${isDev ? ' (dev)' : ''}`)

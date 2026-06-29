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
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { mkdirSync } from 'node:fs'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
const distDir = resolve(root, 'dist')
mkdirSync(distDir, { recursive: true })

const mainEntry = resolve(root, 'electron/main.ts')
const mainOut = resolve(distDir, 'electron-main.mjs')
const preloadEntry = resolve(root, 'electron/preload.ts')
const preloadOut = resolve(distDir, 'electron-preload.js')

const external = ['electron', 'node-pty', 'fs']
  const define = {
    'process.env.HERMES_DESKTOP_IS_PACKAGED': JSON.stringify(true)
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
  banner:  {
    js: "import { createRequire } from 'module'; const require = createRequire(import.meta.url);",
  },
  define,
  logLevel: 'info',
})
console.log(`bundled ${mainOut}`)

// Bundle preload.ts → dist/electron-preload.cjs
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
console.log(`bundled ${preloadOut}`)

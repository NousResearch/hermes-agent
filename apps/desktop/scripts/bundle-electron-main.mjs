#!/usr/bin/env node
// bundle-electron-main.mjs — bundles electron/main.ts and electron/preload.ts
// into self-contained js files in dist/ so the packaged app doesn't need
// node_modules/ or tsx at runtime.
//
// Output:
//   dist/electron-main.mjs    (MJS bundle — entry point for packaged app)
//   dist/electron-preload.js (CJS bundle — loaded via BrowserWindow preload)
//
// The esbuild options live in electron-bundle-options.mjs so the pre-build
// check in assert-root-install.mjs resolves the same bundles.
import { build } from 'esbuild'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { mkdirSync } from 'node:fs'
import { electronBundleOptions } from './electron-bundle-options.mjs'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
mkdirSync(resolve(root, 'dist'), { recursive: true })

const isDev = process.argv.includes('--dev')

for (const options of electronBundleOptions(root, { isDev })) {
  await build({ ...options, logLevel: 'info' })
  console.log(`bundled ${options.outfile}${isDev ? ' (dev)' : ''}`)
}

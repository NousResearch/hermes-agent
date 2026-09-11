#!/usr/bin/env node
import { execFileSync } from 'node:child_process'
import { mkdirSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'
import { isMain, workspaceTool } from '../../../scripts/build/frontend-common.mjs'

const repoRoot = resolve(import.meta.dirname, '../../..')

// Evaluate the one identity module in a fresh process. Its CJS cache and the
// caller's environment must not carry a previous variant into this artifact.
function productIdentity(source, stamp) {
  const variant = stamp.updateMechanism === 'microsoft-store' ? 'store'
    : stamp.payload === 'bootstrap' ? '' : stamp.payload
  if (!['', 'bundled', 'light', 'store'].includes(variant)) {
    throw new Error(`Invalid desktop stamp payload: ${stamp.payload}`)
  }
  return execFileSync(process.execPath, ['-e', 'console.log(JSON.stringify(require(process.argv[1])))',
    join(source, 'apps/desktop/product-identity.cjs')], {
    env: { ...process.env, HERMES_DESKTOP_VARIANT: variant, HERMES_PAYLOAD_TAG: stamp.tag || '' },
    encoding: 'utf8',
  }).trim()
}

/** Bundle main/preload using only the prepared workspace's compiler. */
export async function bundleElectronMain({ source, out, stamp, dev = false }) {
  source = resolve(source)
  out = resolve(out)
  const { build } = await import(pathToFileURL(workspaceTool(source, 'apps/desktop', 'esbuild')).href)
  const define = {}
  if (!dev) {
    if (!stamp) throw new Error('A prepared install stamp is required')
    const raw = readFileSync(stamp, 'utf8')
    const metadata = JSON.parse(raw)
    define['process.env.HERMES_DESKTOP_IS_PACKAGED'] = JSON.stringify(true)
    define.__HERMES_INSTALL_STAMP__ = raw
    define.__HERMES_PRODUCT_IDENTITY__ = productIdentity(source, metadata)
  }
  mkdirSync(out, { recursive: true })
  const common = {
    absWorkingDir: join(source, 'apps/desktop'),
    bundle: true,
    platform: 'node',
    target: 'node20',
    external: ['electron', 'node-pty', 'get-windows', 'fs'],
    define,
    logLevel: 'info',
  }
  await build({
    ...common,
    entryPoints: [join(source, 'apps/desktop/electron/main.ts')],
    format: 'esm',
    outfile: join(out, 'electron-main.mjs'),
    banner: { js: "import { createRequire } from 'module'; const require = createRequire(import.meta.url);" },
  })
  await build({
    ...common,
    entryPoints: [join(source, 'apps/desktop/electron/preload.ts')],
    format: 'cjs',
    outfile: join(out, 'electron-preload.js'),
  })
}

if (isMain(import.meta.url)) {
  const { values } = parseArgs({ options: {
    source: { type: 'string', default: repoRoot },
    out: { type: 'string' }, stamp: { type: 'string' }, dev: { type: 'boolean' },
  } })
  await bundleElectronMain({
    ...values,
    out: values.out || join(values.source, 'apps/desktop/dist'),
    stamp: values.stamp || join(values.source, 'apps/desktop/build/install-stamp.json'),
  })
}

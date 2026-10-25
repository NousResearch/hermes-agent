#!/usr/bin/env node
// Pure browser-renderer compilation. The workspace is prepared by the caller
// (apps/desktop/scripts/build-webapp.mjs); the receipt certifies the checkout.
import { existsSync, readdirSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { productOutput, withProduct, workspaceTool } from './frontend-common.mjs'
import { buildInputs, recordProduct, sourceHash } from './freshness.mjs'

export async function buildWebapp({ source, workspace, out }) {
  if (!workspace) throw new Error('A prepared Webapp workspace is required')
  const app = 'apps/desktop'
  workspace = resolve(workspace)
  ;({ source, out } = productOutput(source, out, [
    // The output lives inside the Desktop workspace: protect its siblings, not the directory.
    ...readdirSync(join(resolve(source), app)).filter(name => !['dist', 'dist-webapp', 'build'].includes(name)).map(name => `${app}/${name}`),
    'apps/shared', 'package.json', 'package-lock.json', 'scripts/build', 'node_modules',
    relative(resolve(source), workspace),
  ]))
  const inputs = buildInputs(source, 'webapp')
  // The workspace is what compiles; a stale mirror must not publish under a current receipt.
  if (sourceHash(workspace, 'webapp') !== inputs.sourceHash) {
    throw new Error(`Prepared Webapp workspace ${workspace} does not match ${source}; prepare it again`)
  }
  const { build } = await import(pathToFileURL(workspaceTool(workspace, app, 'vite')).href)
  await withProduct(out, async (product, scratch) => {
    await build({
      root: join(workspace, app),
      configLoader: 'runner',
      cacheDir: join(scratch, 'vite-cache'),
      build: { outDir: product, emptyOutDir: true },
    })
    if (!existsSync(join(product, 'index.html'))) throw new Error('Webapp build did not produce index.html')
    recordProduct({ source, product: 'webapp', out: product, inputs })
  }, { source })
  return { out, index: join(out, 'index.html') }
}

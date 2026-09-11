#!/usr/bin/env node
import { execFileSync } from 'node:child_process'
import { cpSync, existsSync, readdirSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { bundleElectronMain } from '../../apps/desktop/scripts/bundle-electron-main.mjs'
import { checkDistBuilt } from '../../apps/desktop/scripts/assert-dist-built.mjs'
import { classifyNativeBinary } from '../../apps/desktop/scripts/stage-native-deps.mjs'
import { frontendArgs, isMain, productOutput, withProduct, workspaceTool } from './frontend-common.mjs'

function validateNativeTree(nativeDeps, platform) {
  const pty = join(nativeDeps, 'node-pty')
  if (!existsSync(join(pty, 'package.json'))) throw new Error(`Missing prepared node-pty: ${pty}`)
  const bindings = readdirSync(pty, { recursive: true }).filter(name => name.endsWith('.node'))
  if (!bindings.length) throw new Error(`Prepared node-pty has no native binding: ${pty}`)
  for (const binding of bindings) {
    if (classifyNativeBinary(join(pty, binding)) !== platform) {
      throw new Error(`Prepared node-pty binding does not match ${platform}: ${binding}`)
    }
  }
  if (platform === 'darwin' && !existsSync(join(nativeDeps, 'get-windows/main'))) {
    throw new Error('Prepared get-windows is missing its macOS helper')
  }
}

/** Pure desktop compilation. out is the dist directory, not the source app. */
export async function buildDesktop({ source, out, icons, stamp, nativeDeps, typecheck = false, platform = process.platform }) {
  if (!icons || !stamp || !nativeDeps) throw new Error('icons, stamp and nativeDeps are required prepared inputs')
  const app = 'apps/desktop'
  ;({ source, out } = productOutput(source, out, [
    ...readdirSync(join(resolve(source), app)).filter(name => name !== 'dist').map(name => `${app}/${name}`),
    `${app}/scripts`, 'scripts/build', 'package.json', 'package-lock.json',
    'apps/shared', 'node_modules', ...[join(resolve(icons), app, 'public'), stamp, nativeDeps].map(input => relative(resolve(source), resolve(input))),
  ]))
  const publicIcons = join(resolve(icons), app, 'public')
  if (!existsSync(join(publicIcons, 'apple-touch-icon.png'))) throw new Error(`Missing desktop icon: ${join(publicIcons, 'apple-touch-icon.png')}`)
  validateNativeTree(resolve(nativeDeps), platform)
  await withProduct(out, async (product, scratch) => {
    const publicDir = join(scratch, 'public')
    const sourcePublic = join(source, app, 'public')
    if (existsSync(sourcePublic)) cpSync(sourcePublic, publicDir, { recursive: true })
    cpSync(publicIcons, publicDir, { recursive: true })
    if (typecheck) {
      const ts = workspaceTool(source, app, 'typescript')
      execFileSync(process.execPath, [join(dirname(ts), 'tsc.js'), '-p', join(source, app, 'tsconfig.json'),
        '--noEmit', '--incremental', '--tsBuildInfoFile', join(scratch, 'renderer.tsbuildinfo')], { cwd: join(source, app), stdio: 'inherit' })
    }
    const { build } = await import(pathToFileURL(workspaceTool(source, app, 'vite')).href)
    await build({
      root: join(source, app),
      configLoader: 'runner',
      publicDir,
      cacheDir: join(scratch, 'vite-cache'),
      build: { outDir: product, emptyOutDir: true },
    })
    await bundleElectronMain({ source, out: product, stamp })
    cpSync(resolve(nativeDeps), join(product, 'node_modules'), { recursive: true, dereference: true })
    const result = checkDistBuilt(product)
    if (!result.ok) throw new Error(result.error)
  }, { source })
  return { out }
}

if (isMain(import.meta.url)) {
  const args = frontendArgs(process.argv.slice(2), {
    icons: { type: 'string' }, stamp: { type: 'string' }, 'native-deps': { type: 'string' },
    typecheck: { type: 'boolean' }, platform: { type: 'string' },
  })
  await buildDesktop({ ...args, nativeDeps: args['native-deps'] })
}

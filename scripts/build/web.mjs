#!/usr/bin/env node
// Pure dashboard compilation; icon/dependency preparation belongs to callers.
import { cpSync, existsSync, mkdirSync, statSync } from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { frontendArgs, isMain, productOutput, repoRoot, withProduct, workspaceTool } from './frontend-common.mjs'

function typecheck(ts, root, scratch) {
  const diagnostics = []
  const host = ts.createSolutionBuilderHost(ts.sys, undefined, diagnostic => diagnostics.push(diagnostic))
  // Use the existing tsc -b project graph, with build state isolated from the
  // prepared source. force also prevents stale source buildinfo skipping checks.
  const buildInfo = new Map()
  host.writeFile = (file, data) => {
    if (!file.endsWith('.tsbuildinfo')) throw new Error(`Unexpected TypeScript emit: ${file}`)
    if (!buildInfo.has(file)) buildInfo.set(file, path.join(scratch, `ts-${buildInfo.size}.tsbuildinfo`))
    ts.sys.writeFile(buildInfo.get(file), data)
  }
  const status = ts.createSolutionBuilder(host, [path.join(root, 'tsconfig.json')], { force: true }).build()
  if (status !== 0) {
    const message = ts.formatDiagnosticsWithColorAndContext(diagnostics, {
      getCurrentDirectory: () => root,
      getCanonicalFileName: file => file,
      getNewLine: () => '\n'
    })
    throw new Error(`TypeScript build failed (${status})\n${message}`)
  }
}

export async function buildWeb(options) {
  const { source, out } = productOutput(options.source, options.out, ['web', 'apps/shared', 'node_modules'])
  if (!options.icons) throw new Error('Prepared icons are required (--icons)')
  const publicIcons = path.resolve(options.icons, 'web/public')
  const favicon = path.join(publicIcons, 'favicon.ico')
  if (!existsSync(favicon) || !statSync(favicon).isFile()) throw new Error(`Missing prepared icon: ${favicon}`)
  // Icon inputs can be outside source but are still read-only build inputs.
  productOutput(options.icons, out, ['web/public'])
  const root = path.join(source, 'web')
  const tsModule = await import(pathToFileURL(workspaceTool(source, 'web', 'typescript')).href)
  const { build } = await import(pathToFileURL(workspaceTool(source, 'web', 'vite')).href)
  await withProduct(out, async (product, scratch) => {
    typecheck(tsModule.default ?? tsModule, root, scratch)
    const publicDir = path.join(scratch, 'public')
    mkdirSync(publicDir)
    if (existsSync(path.join(root, 'public'))) cpSync(path.join(root, 'public'), publicDir, { recursive: true })
    cpSync(publicIcons, publicDir, { recursive: true })
    await build({
      root,
      configFile: path.join(root, 'vite.config.ts'),
      // Vite's default config bundler writes into source node_modules/.vite-temp.
      configLoader: 'runner',
      cacheDir: path.join(scratch, 'vite-cache'),
      publicDir,
      build: { outDir: product, emptyOutDir: true }
    })
    if (!existsSync(path.join(product, 'index.html'))) throw new Error('Web build did not produce index.html')
  }, { source })
  return { out, index: path.join(out, 'index.html') }
}

if (isMain(import.meta.url)) {
  try {
    const options = process.argv.length === 2
      ? { source: repoRoot, icons: repoRoot, out: path.join(repoRoot, 'hermes_cli/web_dist') }
      : frontendArgs(process.argv.slice(2), { icons: { type: 'string' } })
    const result = await buildWeb(options)
    console.log(`built ${result.index}`)
  } catch (error) {
    console.error(error)
    process.exitCode = 1
  }
}

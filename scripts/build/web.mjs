#!/usr/bin/env node
// Pure dashboard compilation; icon/dependency preparation belongs to callers.
import { cpSync, existsSync, mkdirSync, readFileSync, statSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { frontendArgs, isMain, productOutput, repoRoot, withProduct, workspaceTool } from './frontend-common.mjs'
import { recordProduct, buildInputs } from './freshness.mjs'

// --- Bounded build (#63338) -------------------------------------------------
// `npm run build` drives Vite 8's Rust-native bundler (Rolldown), whose rayon
// thread pool saturates every CPU on small hosts (200%+ `top` readings, OOMs
// and frozen SSH sessions on 2–4 vCPU VPS boxes). Vite exposes no worker knob,
// but Rolldown's native binding honours RAYON_NUM_THREADS, and V8's heap is
// capped with --max-old-space-size. `--max-old-space-size` cannot affect an
// already-running process, so the default heap cap is enforced with a single
// re-exec (spawnSync → exit); the thread cap is set before vite is imported.
// User-set env (NODE_OPTIONS heap flag, RAYON_NUM_THREADS) always wins, and
// HERMES_WEB_BUILD_MAX_OLD_SPACE_SIZE / HERMES_WEB_BUILD_THREADS tailor the
// caps; HERMES_WEB_BUILD_LIGHT=1 tightens them (1 thread, minimum heap) for
// hosts that cannot spare full CPU during the build.
const MIN_HEAP_MB = 1024
const MAX_HEAP_MB = 4096
const MAX_THREADS = 8
const TRUTHY = new Set(['1', 'true', 'yes', 'on'])

const webBuildLight = () => TRUTHY.has(String(process.env.HERMES_WEB_BUILD_LIGHT ?? '').toLowerCase())

function availableCores() {
  try {
    return Math.max(1, os.availableParallelism?.() ?? (os.cpus().length || 1))
  } catch {
    return 1
  }
}

// Half the cores (the dashboard graph is small), clamped to [1, MAX_THREADS].
function boundedThreads() {
  return Math.max(1, Math.min(MAX_THREADS, Math.ceil(availableCores() / 2)))
}

// cgroup v2/v1 memory limit in MB, or null when unconstrained.
function readCgroupLimitMb() {
  for (const cgroupPath of ['/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes']) {
    try {
      const raw = String(readFileSync(cgroupPath, 'utf-8')).trim()
      if (raw === 'max' || !/^\d+$/.test(raw)) continue
      const value = Number(raw)
      if (value <= 0 || value >= 2 ** 50) return null // the v1 "unlimited" sentinel
      return Math.floor(value / (1024 * 1024))
    } catch { /* unconstrained or unsupported */ }
  }
  return null
}

// 75% of the cgroup limit when constrained (mirrors the TUI launcher's heap
// sizing), else the conservative ceiling. Never below the GC-thrash floor.
function boundedHeapMb() {
  const limitMb = readCgroupLimitMb()
  const sized = limitMb ? Math.floor(limitMb * 0.75) : MAX_HEAP_MB
  return Math.max(MIN_HEAP_MB, Math.min(MAX_HEAP_MB, sized))
}

function applyThreadCap() {
  if (process.env.RAYON_NUM_THREADS) return
  const explicit = Number.parseInt(process.env.HERMES_WEB_BUILD_THREADS ?? '', 10)
  const threads = Number.isFinite(explicit) && explicit > 0 ? explicit
    : (webBuildLight() ? 1 : boundedThreads())
  process.env.RAYON_NUM_THREADS = String(Math.max(1, threads))
}

// Re-exec with the heap cap appended to NODE_OPTIONS unless the user already
// set one. Runs only for direct invocation (isMain), before vite is imported.
async function ensureHeapCapForFreshProcess() {
  const tokens = (process.env.NODE_OPTIONS ?? '').split(/\s+/).filter(Boolean)
  if (tokens.some(token => token.startsWith('--max-old-space-size='))) return
  const explicit = Number.parseInt(process.env.HERMES_WEB_BUILD_MAX_OLD_SPACE_SIZE ?? '', 10)
  const heapMb = Number.isFinite(explicit) && explicit > 0
    ? Math.max(MIN_HEAP_MB, explicit)
    : (webBuildLight() ? MIN_HEAP_MB : boundedHeapMb())
  tokens.push(`--max-old-space-size=${heapMb}`)
  const { spawnSync } = await import('node:child_process')
  const result = spawnSync(process.execPath, process.argv.slice(1), {
    stdio: 'inherit',
    env: { ...process.env, NODE_OPTIONS: tokens.join(' ') },
  })
  process.exit(result.status ?? 1)
}

if (isMain(import.meta.url)) {
  applyThreadCap()
  await ensureHeapCapForFreshProcess()
}

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
  const inputs = buildInputs(source, 'web', { icons: publicIcons })
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
    recordProduct({ source, product: 'web', out: product, inputs })
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

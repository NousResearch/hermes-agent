// batch-sign-binaries.mjs — Authenticode-sign every standalone binary inside
// the packed Windows app tree (Hermes.exe's sibling DLLs and everything under
// resources/agent-payload/tools/: node.exe, ffmpeg.exe, chromium, the minted
// CLI launcher exes, pm tool binaries, …).
//
// Why a batch: Windows validates only the MSIX package signature
// (AppxSignature.p7x over AppxBlockMap.xml), so inner binaries have never been
// signed. But an MSIX whose payload carries unsigned PEs trips SmartScreen /
// enterprise WDAC policies that scan inner files, and signature-verification
// tooling reports the app as mixed signed/unsigned. Task 0 (pm-clean fix
// plan): sign the whole tree, once, in chunked signtool invocations.
//
// Ordering (electron-builder win32 pipeline, pinned by app-builder-lib source):
//   beforePack → pack → afterPack (this module's caller) → electron fuses
//   → signAndEditResources (rcedit on the product exe) → per-file sign hook.
// Consequences:
//   - The batch runs in afterPack AFTER sanitize-pe-signatures.mjs (a dangling
//     certificate table makes signtool fail 0x800700C1) and AFTER the rcedit
//     identity stamp, so neither can invalidate what we sign.
//   - The product exe (`<productName>.exe`) is EXCLUDED from the batch:
//     rcedit edits its resources (and the electron fuses flip) after
//     afterPack, which invalidates any signature. It is signed per-file by the
//     customSign hook AFTER rcedit, on the exact Azure mechanism sign-msix.mjs
//     uses for the package itself.
//   - The customSign hook returns true (does nothing) for every file the batch
//     already covered, so electron-builder never re-signs one-by-one.
//
// Sign-nested-chromium.mjs stays as-is: it is macOS-only (codesign --deep over
// .app bundles inside the payload for Apple notarization) and is invoked from
// the darwin branch of after-pack.mjs. There is no overlap with this Windows
// Authenticode batch.
//
// Gating matches the rest of the pipeline: this module only signs when the
// Azure Trusted Signing variables are present (the same AZURE_SIGN_* set the
// release-signing workflow arms provide). Without them — local builds, forks,
// unsigned canary lanes — it is a no-op with a loud warning, exactly like
// stage-msixbundle.mjs. The dlib + signtool resolution reuses the
// electron-builder cache walk from scripts/stage-msixbundle.mjs; nothing is
// hardcoded to C:\Tools.

import { execFile } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { isMain } from './utils.mjs'

export const CHUNK_SIZE = 100
// How many signtool children may run at once. Azure Trusted Signing and the
// timestamp server are both network round-trips per file, so N concurrent
// children multiply throughput ~Nx. Keep it modest — the timestamp server
// rate-limits aggressive bursters.
export const DEFAULT_CONCURRENCY = 4
// Separate timestamp pass URL. `timestamp.acs.microsoft.com` intermittently
// fails with "Invalid Time Stamp Request Length:-1" (documented widely);
// digicert's RFC3161 endpoint has been reliable in the release pipeline.
const TIMESTAMP_URL = 'http://timestamp.digicert.com'

/**
 * execFile as a promise, so chunks can run concurrently.
 * @returns {Promise<void>} rejects with the child's error on non-zero exit.
 */
function execFileAsync(cmd, args, options) {
  return new Promise((resolve, reject) => {
    execFile(cmd, args, options, (error) => {
      if (error) reject(error)
      else resolve()
    })
  })
}

/**
 * Retry a fallible op (bounded). Used for the timestamp pass, where the
 * external server flakes intermittently — a retried timestamp beats a whole
 * rebuild. Exponential backoff (1s, 2s, ...) between attempts.
 *
 * @param {() => Promise<void>} fn
 * @param {{ attempts?: number, baseDelayMs?: number }} [opts]
 */
export async function withRetry(fn, opts = {}) {
  const attempts = opts.attempts ?? 3
  const baseDelayMs = opts.baseDelayMs ?? 1000
  let lastError
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await fn()
    } catch (error) {
      lastError = error
      if (attempt < attempts) {
        await new Promise((resolve) => setTimeout(resolve, baseDelayMs * attempt))
      }
    }
  }
  throw lastError
}

/**
 * Run up to `concurrency` async workers over items. Each worker pulls the
 * next item as it frees up, so unevenly-sized work balances itself.
 *
 * @param {T[]} items
 * @param {number} concurrency
 * @param {(item: T, index: number) => Promise<void>} worker
 * @template T
 */
export async function runPool(items, concurrency, worker) {
  let index = 0
  const next = async () => {
    while (index < items.length) {
      const i = index
      index += 1
      await worker(items[i], i)
    }
  }
  const workers = Array.from(
    { length: Math.min(Math.max(1, concurrency), items.length) },
    () => next()
  )
  await Promise.all(workers)
}

/**
 * Recursively collect every .exe/.dll under dir. Symbolic links are skipped
 * (the payload's materialized-link layout can contain them; the target is
 * collected on its own walk). Sorted for deterministic chunking.
 *
 * @param {string} dir
 * @param {{ skip?: (file: string) => boolean }} [opts]
 * @returns {string[]}
 */
export function getBinaries(dir, opts = {}) {
  const out = []
  const walk = (current) => {
    let entries
    try {
      entries = fs.readdirSync(current, { withFileTypes: true })
    } catch {
      return
    }
    for (const entry of entries) {
      if (entry.isSymbolicLink()) continue
      const full = path.join(current, entry.name)
      if (entry.isDirectory()) {
        walk(full)
        continue
      }
      if (!entry.isFile()) continue
      const lower = entry.name.toLowerCase()
      if (lower.endsWith('.exe') || lower.endsWith('.dll')) {
        if (opts.skip && opts.skip(full)) continue
        out.push(full)
      }
    }
  }
  walk(dir)
  return out.sort()
}

/** Split a file list into signtool-sized batches. */
export function chunk(items, size = CHUNK_SIZE) {
  const chunks = []
  for (let i = 0; i < items.length; i += size) {
    chunks.push(items.slice(i, i + size))
  }
  return chunks
}

/** True when the Azure Trusted Signing variables are all present. */
export function azureSigningConfigured(env = process.env) {
  return Boolean(env.AZURE_SIGN_ENDPOINT && env.AZURE_SIGN_ACCOUNT && env.AZURE_SIGN_PROFILE)
}

// The electron-builder toolset cache roots, in precedence order — the same
// walk scripts/stage-msixbundle.mjs does (configured ELECTRON_BUILDER_CACHE
// beats stray defaults).
function cacheRoots(env) {
  return [
    env.ELECTRON_BUILDER_CACHE || '',
    path.join(env.LOCALAPPDATA || '', 'electron-builder', 'Cache'),
    path.join(env.USERPROFILE || '', 'AppData', 'Local', 'electron-builder', 'Cache')
  ].filter(Boolean)
}

/**
 * Host arch for the signing toolset. The ATS bundle ships the dlib for
 * x64/x86 only (its arm64 dir has no dlib), and a 32-bit signtool cannot
 * load a 64-bit dlib — so the signtool + dlib pair must be same-arch. On
 * arm64 hosts use the x64 pair (x64 signtool runs under Windows-on-ARM
 * emulation). Mirrors app-builder-lib's WindowsSignAzureManager
 * (`process.arch === "ia32" ? Arch.ia32 : Arch.x64`).
 */
export function signingArch() {
  return process.arch === 'ia32' ? 'x86' : 'x64'
}

/**
 * True when a resolved path sits in the `<arch>/` dir (e.g. `.../x64/signtool.exe`
 * or `.../ats-bundle-.../x64/Azure.CodeSigning.Dlib.dll`), matching both the
 * modern windows-kits-bundle layout and the legacy windows-10/<arch> layout.
 */
function archMatch(file, arch) {
  return path.basename(path.dirname(file)).toLowerCase() === arch
}

/**
 * Find azure.codesigning.dlib.dll under the electron-builder cache, preferring
 * the one matching the host arch (x64 unless the host is ia32) so signtool can
 * load it. Returns an absolute path or null (caller decides loud-fail vs warn).
 */
export function resolveTrustedSigningDlib(env = process.env) {
  const arch = signingArch()
  for (const root of cacheRoots(env)) {
    if (!fs.existsSync(root)) continue
    const found = []
    const walk = (p) => {
      let entries
      try {
        entries = fs.readdirSync(p, { withFileTypes: true })
      } catch {
        return
      }
      for (const entry of entries) {
        const full = path.join(p, entry.name)
        if (entry.isDirectory()) walk(full)
        else if (entry.name.toLowerCase() === 'azure.codesigning.dlib.dll') found.push(full)
      }
    }
    for (const entry of fs.readdirSync(root)) {
      walk(path.join(root, entry))
    }
    if (found.length > 0) {
      const matched = found.filter(f => archMatch(f, arch))
      const pool = matched.length > 0 ? matched : found
      pool.sort()
      return pool[pool.length - 1]
    }
  }
  return null
}

/**
 * Find a host signtool.exe under the electron-builder cache (the winCodeSign
 * toolset bundles the Windows Kits tools for arm64/x64/x86), or honor
 * SIGNTOOL_PATH. Prefers the signtool matching the host arch — a 32-bit
 * signtool cannot load the x64 ATS dlib — and among arch-matched candidates
 * the newest SDK build (sorted paths put the highest version last).
 * Returns an absolute path or null.
 */
export function resolveSigntool(env = process.env) {
  if (env.SIGNTOOL_PATH && fs.existsSync(env.SIGNTOOL_PATH)) return env.SIGNTOOL_PATH
  const arch = signingArch()
  for (const root of cacheRoots(env)) {
    if (!fs.existsSync(root)) continue
    const found = []
    for (const entry of fs.readdirSync(root)) {
      const dir = path.join(root, entry)
      if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) continue
      const walk = (p, depth) => {
        if (depth > 5) return
        let entries
        try {
          entries = fs.readdirSync(p, { withFileTypes: true })
        } catch {
          return
        }
        for (const sub of entries) {
          if (sub.isSymbolicLink()) continue
          const full = path.join(p, sub.name)
          if (sub.isDirectory()) walk(full, depth + 1)
          else if (sub.name.toLowerCase() === 'signtool.exe') found.push(full)
        }
      }
      walk(dir, 0)
    }
    if (found.length > 0) {
      const matched = found.filter(f => archMatch(f, arch))
      const pool = matched.length > 0 ? matched : found
      pool.sort()
      return pool[pool.length - 1]
    }
  }
  return null
}

/**
 * Find the bundled .NET 8 runtime dir (win-codesign@&lt;ver&gt;/dotnet-runtime-*).
 * The ATS dlib is a .NET assembly loaded via Ijwhost.dll, which locates
 * hostfxr.dll through DOTNET_ROOT — without it the dlib cannot initialize even
 * with a same-arch signtool/dlib pair. Returns an absolute path or null.
 */
export function resolveDotnetRuntimeDir(env = process.env) {
  for (const root of cacheRoots(env)) {
    if (!fs.existsSync(root)) continue
    const found = []
    const walk = (p, depth) => {
      if (depth > 3) return
      let entries
      try {
        entries = fs.readdirSync(p, { withFileTypes: true })
      } catch {
        return
      }
      for (const entry of entries) {
        if (!entry.isDirectory()) continue
        const full = path.join(p, entry.name)
        if (/^dotnet-runtime-/.test(entry.name)) found.push(full)
        else walk(full, depth + 1)
      }
    }
    for (const entry of fs.readdirSync(root)) {
      walk(path.join(root, entry), 0)
    }
    if (found.length > 0) {
      found.sort()
      return found[found.length - 1]
    }
  }
  return null
}

/**
 * Sign one chunk of binaries with a single signtool invocation (argv array,
 * never a shell string — the joined list can be long and must not interpolate
 * through a shell). Azure-only: NO timestamp here (see timestampChunk — the
 * sign pass would otherwise hold each file hostage to the flaky timestamp
 * server, and a timestamp failure would force a full re-sign).
 *
 * @param {string[]} files
 * @param {{ signtool: string, dlib: string, metadataPath: string, exec?: typeof execFile, execOptions?: import('node:child_process').ExecFileOptions }} opts
 */
export async function signChunk(files, opts) {
  const args = [
    'sign',
    '/fd', 'SHA256',
    '/dlib', opts.dlib,
    '/dmdf', opts.metadataPath,
    ...files
  ]
  if (opts.exec) {
    await opts.exec(opts.signtool, args, opts.execOptions)
    return
  }
  await execFileAsync(opts.signtool, args, opts.execOptions ?? { stdio: 'inherit' })
}

/**
 * RFC3161-timestamp one chunk of ALREADY-SIGNED files. No /dlib, no /dmdf —
 * this is pure timestamping, so it neither re-auths against Azure nor
 * re-initializes the .NET dlib. The timestamp server is the flaky external
 * dependency, so this pass retries per chunk (a retried timestamp beats a
 * whole re-sign) and runs concurrently like the sign pass.
 *
 * @param {string[]} files
 * @param {{ signtool: string, timestampUrl?: string, exec?: typeof execFile, execOptions?: import('node:child_process').ExecFileOptions, timestampAttempts?: number, timestampRetryDelayMs?: number }} opts
 */
export async function timestampChunk(files, opts) {
  const args = [
    'timestamp',
    '/tr', opts.timestampUrl ?? TIMESTAMP_URL,
    '/td', 'SHA256',
    ...files
  ]
  const run = async () => {
    if (opts.exec) {
      await opts.exec(opts.signtool, args, opts.execOptions)
      return
    }
    await execFileAsync(opts.signtool, args, opts.execOptions ?? { stdio: 'inherit' })
  }
  await withRetry(run, {
    attempts: opts.timestampAttempts,
    baseDelayMs: opts.timestampRetryDelayMs
  })
}

/**
 * Batch-sign every binary under a tree.
 *
 * Two passes: (1) Azure Authenticode sign — concurrent signtool children,
 * no timestamp; (2) RFC3161 timestamp — concurrent, no Azure/dlib, retried
 * per chunk. Parallelism is the whole speed story: both Azure and the
 * timestamp server are per-file network round-trips, so N concurrent children
 * multiply throughput ~Nx.
 *
 * @param {string[]} binaries file list from getBinaries
 * @param {{ env?: NodeJS.ProcessEnv, exec?: typeof execFile, chunkSize?: number, concurrency?: number, mkdtemp?: typeof fs.mkdtempSync, signtool?: string, dlib?: string, timestampUrl?: string, timestampAttempts?: number, timestampRetryDelayMs?: number }} [opts]
 * @returns {Promise<{ signed: number, chunks: number, skipped: boolean }>}
 *   skipped=true when Azure signing is not configured (caller warns).
 */
export async function batchSignBinaries(binaries, opts = {}) {
  const env = opts.env ?? process.env
  if (!azureSigningConfigured(env)) {
    return { signed: 0, chunks: 0, skipped: true }
  }
  if (binaries.length === 0) {
    return { signed: 0, chunks: 0, skipped: false }
  }
  const dlib = opts.dlib ?? resolveTrustedSigningDlib(env)
  if (!dlib) {
    throw new Error('batch-sign-binaries: azure.codesigning.dlib.dll not found under the electron-builder cache')
  }
  const signtool = opts.signtool ?? resolveSigntool(env)
  if (!signtool) {
    throw new Error('batch-sign-binaries: signtool.exe not found under the electron-builder cache (or SIGNTOOL_PATH)')
  }
  // The ATS dlib is a .NET assembly; Ijwhost.dll finds hostfxr.dll via
  // DOTNET_ROOT. Mirror app-builder-lib's WindowsSignAzureManager and point it
  // at the bundled runtime so the dlib initializes (a missing runtime reads as
  // "no certificates found"). Only merged when a runtime dir exists.
  const signEnv = { ...env }
  const dotnetRoot = opts.dotnetRoot ?? resolveDotnetRuntimeDir(env)
  if (dotnetRoot) signEnv.DOTNET_ROOT = dotnetRoot
  const mkdtemp = opts.mkdtemp ?? fs.mkdtempSync
  const tmpDir = mkdtemp(path.join(env.TEMP || env.TMP || '.', 'batch-sign-'))
  const metadataPath = path.join(tmpDir, 'batch-sign.json')
  fs.writeFileSync(metadataPath, JSON.stringify({
    Endpoint: env.AZURE_SIGN_ENDPOINT,
    CodeSigningAccountName: env.AZURE_SIGN_ACCOUNT,
    CertificateProfileName: env.AZURE_SIGN_PROFILE
  }))
  const concurrency = opts.concurrency ?? DEFAULT_CONCURRENCY
  const batches = chunk(binaries, opts.chunkSize ?? CHUNK_SIZE)
  const execOptions = { stdio: 'inherit', env: signEnv }
  try {
    // Pass 1: Azure Authenticode sign — concurrent, no timestamp.
    await runPool(batches, concurrency, (batch) =>
      signChunk(batch, { signtool, dlib, metadataPath, exec: opts.exec, execOptions })
    )
    // Pass 2: RFC3161 timestamp — concurrent, no Azure/dlib, retried.
    await runPool(batches, concurrency, (batch) =>
      timestampChunk(batch, {
        signtool,
        timestampUrl: opts.timestampUrl,
        exec: opts.exec,
        execOptions,
        timestampAttempts: opts.timestampAttempts,
        timestampRetryDelayMs: opts.timestampRetryDelayMs
      })
    )
    return { signed: binaries.length, chunks: batches.length, skipped: false }
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true })
  }
}

/**
 * afterPack-side entry: batch-sign the packed tree, excluding the product exe
 * (it is rcedit-ed and signed per-file after this hook — see the header).
 * Callers must have run sanitize-pe-signatures.mjs first.
 *
 * @param {string} appOutDir
 * @param {string} productExePath absolute path of the main product exe
 * @param {{ env?: NodeJS.ProcessEnv, exec?: typeof execFile, chunkSize?: number, concurrency?: number, timestampUrl?: string, timestampAttempts?: number, timestampRetryDelayMs?: number }} [opts]
 */
export async function batchSignAppTree(appOutDir, productExePath, opts = {}) {
  const env = opts.env ?? process.env
  if (!azureSigningConfigured(env)) {
    console.warn(
      '[batch-sign] AZURE_SIGN_* not set — payload binaries will be UNSIGNED ' +
      '(package block-map still covers them; release lanes must set the signing env)'
    )
    return { signed: 0, chunks: 0, skipped: true }
  }
  const productExe = productExePath ? path.resolve(productExePath) : null
  // uv-cache/ is the pm bundle's deliberately-shipped build cache (inert
  // sdist/archive artifacts, never loaded at runtime — the arch audit
  // exempts it for the same reason). Signing it wastes Azure round-trips
  // on dead weight and can FAIL: locally the cache may hold files that
  // were removed between pm bundle staging and the afterPack walk.
  const inCache = (file) => /(^|[\\/])uv-cache[\\/]/.test(path.resolve(file))
  const binaries = getBinaries(appOutDir, {
    skip: (file) =>
      inCache(file) || (productExe ? path.resolve(file) === productExe : false)
  })
  if (binaries.length === 0) return { signed: 0, chunks: 0, skipped: false }
  const result = await batchSignBinaries(binaries, opts)
  console.log(
    `[batch-sign] signed ${result.signed} payload binaries in ${result.chunks} signtool batch(es)` +
    ` (product exe excluded — signed per-file after rcedit)`
  )
  return result
}

// ── electron-builder custom win.sign hook ───────────────────────────────────
//
// Per app-builder-lib's signtoolBaseSignManager, the custom `sign` hook is
// invoked once per signable file (the product exe after rcedit, top-level
// exes, asar.unpacked natives, extraResource exes). The hook's return value is
// ignored by the manager, but per the Task 0 contract it resolves true for
// files the afterPack batch already signed — a no-op — and delegates the two
// artifacts that genuinely need per-file signing to the sign-msix.mjs Azure
// machinery: the .msix/.msixbundle package and the product exe.

const SIGNABLE_PACKAGE_EXTENSIONS = ['.msix', '.msixbundle']
const STORE_ARTIFACT_PREFIX = 'Store-'

/**
 * The electron-builder custom win.sign hook.
 *
 * @param {{ path: string }} configuration
 * @param {any} packager
 * @param {{ signMsix?: (configuration: any, packager: any) => Promise<void>, azureSignFile?: (file: string, packager: any) => Promise<void> }} [deps]
 * @returns {Promise<boolean>} true when this hook handled the file
 *   (batch-signed: nothing to do) — electron-builder must not re-sign it.
 */
export async function customSign(configuration, packager, deps = {}) {
  const file = configuration.path
  const base = path.basename(file)
  // Store-submission packages are Partner Center's to sign (see sign-msix.mjs).
  if (base.startsWith(STORE_ARTIFACT_PREFIX)) return true
  const lower = file.toLowerCase()
  if (SIGNABLE_PACKAGE_EXTENSIONS.some(ext => lower.endsWith(ext))) {
    const { default: signMsix } = await import('./sign-msix.mjs')
    await (deps.signMsix ?? signMsix)(configuration, packager)
    return true
  }
  // The product exe was rcedit-ed after the batch ran, so it is signed here,
  // after its resources are final, on the same Azure manager sign-msix uses.
  const productName = packager?.appInfo?.productFilename
  if (productName && base.toLowerCase() === `${productName.toLowerCase()}.exe`) {
    const { azureSignFile } = await import('./sign-msix.mjs')
    await (deps.azureSignFile ?? azureSignFile)(file, packager)
    return true
  }
  // Everything else the hook is offered was already batch-signed in afterPack.
  return true
}

async function main() {
  const root = process.argv[2]
  if (!root) {
    console.error('usage: batch-sign-binaries.mjs <dir>')
    process.exit(2)
  }
  const result = await batchSignAppTree(root, process.env.HERMES_PRODUCT_EXE || path.join(root, 'Hermes.exe'))
  if (result.skipped) process.exit(0)
}

if (isMain(import.meta.url)) {
  main()
}

// electron-builder's resolveFunction prefers a named export matching the hook
// name ("sign") and falls back to the module default — provide both so the
// config's `sign: './scripts/batch-sign-binaries.mjs'` binds to customSign.
export { customSign as sign }
export default customSign

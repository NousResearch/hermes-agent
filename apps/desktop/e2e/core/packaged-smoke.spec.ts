/**
 * Packaged-build smoke: the artifact users actually run, not the dev tree.
 *
 * Every other core spec launches the dev Electron binary against
 * apps/desktop, so a broken `electron-builder` output (asarUnpack not
 * honoured, the main bundle missing from the archive, an unpacked tree that
 * was never generated — #121097) ships green. This spec reads the Linux
 * `electron-builder --dir` output in release/ and asserts:
 *
 *  1. asarUnpack contract, derived from apps/desktop/package.json at test time
 *     (never a file-list snapshot): for every declared glob, matched with the
 *     packager's own matcher semantics (minimatch, dot: true), every archive
 *     entry it matches is flagged unpacked AND present on disk under
 *     resources/app.asar.unpacked with the recorded size, and it matches at
 *     least one file there. A glob that matches nothing is only tolerated when
 *     the staged native dependency it targets ships that content for OTHER
 *     platforms only (e.g. node-pty's `prebuilds/darwin-*`, `win32-*`), which
 *     is proven from the dependency's installed files, not assumed. The main
 *     entry the archive's own package.json names must be present.
 *  2. Boot to first chat: the PACKAGED binary (release/linux-unpacked/<exe>,
 *     `app.isPackaged` asserted) with a sandboxed HOME/HERMES_HOME and only
 *     the LLM faked (scripted loopback provider) completes a first turn: the
 *     reply is rendered, persisted (state.db + REST), and passes the core
 *     transcript oracle.
 *
 * No packaged build: skipped with a reason locally; FAILS when CI is set and
 * HERMES_E2E_REQUIRE_PACKAGED=1 (the CI step that builds the pack sets it).
 * HERMES_E2E_PACKAGED_DIR points the spec at another `*-unpacked` dir (used
 * by the sabotage proofs on a copied release).
 */

import * as fs from 'node:fs'
import { createRequire } from 'node:module'
import * as path from 'node:path'

import { _electron, type ElectronApplication, expect, type Page, test } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  DESKTOP_ROOT,
  recordWebSockets,
  REPO_ROOT,
  sandboxProcesses,
  send,
  storedSessionForMarker,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { assertTranscriptOracle, installDuplicateSampler } from './oracle'
import { startScriptedProvider } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`

// ─── Locating the packaged build ────────────────────────────────────────

interface BuildConfig {
  productName?: string
  executableName?: string
  asarUnpack?: string | string[]
  directories?: { output?: string }
  linux?: { executableName?: string; asarUnpack?: string | string[] }
}

interface DesktopPackageJson {
  name: string
  main: string
  build: BuildConfig
}

const PKG = JSON.parse(fs.readFileSync(path.join(DESKTOP_ROOT, 'package.json'), 'utf8')) as DesktopPackageJson

/** electron-builder's Linux `--dir` output: `linux-unpacked` for x64, `linux-<arch>-unpacked` otherwise. */
function unpackedDir(): string {
  if (process.env.HERMES_E2E_PACKAGED_DIR) {
    return path.resolve(process.env.HERMES_E2E_PACKAGED_DIR)
  }

  const out = path.resolve(DESKTOP_ROOT, PKG.build.directories?.output ?? 'dist')
  const arch = process.arch === 'x64' ? '' : `-${process.arch}`

  return path.join(out, `linux${arch}-unpacked`)
}

/** electron-builder's Linux executable name: linux.executableName > executableName > lowercased package name. */
function executableName(): string {
  return PKG.build.linux?.executableName ?? PKG.build.executableName ?? PKG.name.toLowerCase()
}

function asArray(value: string | string[] | undefined): string[] {
  return value === undefined ? [] : Array.isArray(value) ? value : [value]
}

interface PackagedBuild {
  dir: string
  resources: string
  asarPath: string
  unpackedRoot: string
  executable: string
}

/** The packaged build, or skip (local) / throw (CI with HERMES_E2E_REQUIRE_PACKAGED=1). */
function packagedBuildOrSkip(): PackagedBuild {
  test.skip(process.platform !== 'linux', 'packaged smoke covers the Linux electron-builder --dir output only')
  const dir = unpackedDir()
  const resources = path.join(dir, 'resources')

  const build: PackagedBuild = {
    dir,
    resources,
    asarPath: path.join(resources, 'app.asar'),
    unpackedRoot: path.join(resources, 'app.asar.unpacked'),
    executable: path.join(dir, executableName())
  }

  const missing = [build.executable, build.asarPath].filter(file => !fs.existsSync(file))

  if (missing.length > 0) {
    const reason =
      `no packaged build (missing ${missing.join(', ')}): run ` +
      "'npm run build && npm run builder -- --dir --publish never --linux' in apps/desktop"

    if (process.env.CI && process.env.HERMES_E2E_REQUIRE_PACKAGED === '1') {
      throw new Error(`HERMES_E2E_REQUIRE_PACKAGED=1 but ${reason}`)
    }

    test.skip(true, reason)
  }

  return build
}

// ─── asar archive reading (format: pickle-framed JSON header, then data) ─

interface AsarEntry {
  /** Posix path relative to the archive root. */
  rel: string
  unpacked: boolean
  size: number
  link: boolean
  offset: number
}

interface AsarArchive {
  files: Map<string, AsarEntry>
  dataStart: number
}

function readAsar(asarPath: string): AsarArchive {
  const fd = fs.openSync(asarPath, 'r')

  try {
    const prefix = Buffer.alloc(16)
    fs.readSync(fd, prefix, 0, 16, 0)
    // [u32 4][u32 headerPickleSize][u32 payloadSize][u32 jsonLength][json…]
    const headerPickleSize = prefix.readUInt32LE(4)
    const jsonLength = prefix.readUInt32LE(12)
    const json = Buffer.alloc(jsonLength)
    fs.readSync(fd, json, 0, jsonLength, 16)
    const header = JSON.parse(json.toString('utf8'))
    const files = new Map<string, AsarEntry>()

    const walk = (node: any, prefixPath: string) => {
      for (const [name, child] of Object.entries<any>(node.files ?? {})) {
        const rel = prefixPath ? `${prefixPath}/${name}` : name

        if (child.files) {
          walk(child, rel)
        } else {
          files.set(rel, {
            rel,
            unpacked: Boolean(child.unpacked),
            size: Number(child.size ?? 0),
            link: typeof child.link === 'string',
            offset: Number(child.offset ?? 0)
          })
        }
      }
    }

    walk(header, '')

    return { files, dataStart: 8 + headerPickleSize }
  } finally {
    fs.closeSync(fd)
  }
}

/** Contents of an archive file, following the unpacked flag the way Electron's asar fs does. */
function readArchived(build: PackagedBuild, archive: AsarArchive, rel: string): Buffer | null {
  const entry = archive.files.get(rel)

  if (!entry) {
    return null
  }

  if (entry.unpacked) {
    const onDisk = path.join(build.unpackedRoot, rel)

    return fs.existsSync(onDisk) ? fs.readFileSync(onDisk) : null
  }

  const out = Buffer.alloc(entry.size)
  const fd = fs.openSync(build.asarPath, 'r')

  try {
    fs.readSync(fd, out, 0, entry.size, archive.dataStart + entry.offset)
  } finally {
    fs.closeSync(fd)
  }

  return out
}

/** Every file under `root`, as posix paths relative to `root`. */
function walkFiles(root: string, prefix = ''): string[] {
  if (!fs.existsSync(root)) {
    return []
  }

  const out: string[] = []

  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const rel = prefix ? `${prefix}/${entry.name}` : entry.name

    if (entry.isDirectory()) {
      out.push(...walkFiles(path.join(root, entry.name), rel))
    } else {
      out.push(rel)
    }
  }

  return out
}

// ─── The packager's glob semantics ──────────────────────────────────────

type Matcher = (rel: string) => boolean

/**
 * The same matcher electron-builder applies to asarUnpack: minimatch from
 * app-builder-lib's own dependency tree with { dot: true }, plus its rule
 * that a literal, extension-less pattern also covers `<pattern>/**\/*`.
 */
function packagerMatcher(pattern: string): Matcher {
  const desktopRequire = createRequire(path.join(DESKTOP_ROOT, 'package.json'))
  const builderRequire = createRequire(desktopRequire.resolve('app-builder-lib/package.json'))
  const { Minimatch } = builderRequire('minimatch') as { Minimatch: new (p: string, o: object) => any }
  const normalized = path.posix.normalize(pattern.replace(/\\/g, '/').replace(/^\.\//, ''))
  const matchers = [new Minimatch(normalized, { dot: true })]

  const hasMagic = matchers[0].set.length > 1 || matchers[0].set[0].some((part: unknown) => typeof part !== 'string')

  if (!normalized.includes('.') && !hasMagic) {
    matchers.push(new Minimatch(`${normalized}/**/*`, { dot: true }))
  }

  return rel => matchers.some(m => m.match(rel))
}

const PLATFORM_DIR_RE = /(?:^|\/)(darwin|win32|linux|freebsd|openbsd|sunos|aix|android)-[a-z0-9]+(?:\/|$)/

/**
 * Is `glob` legitimately empty on this platform? True only when the staged
 * native dependencies (dist/node_modules/<name> in the archive) have
 * installed files matching it and every one sits under a platform-arch
 * directory for a different OS (node-pty ships prebuilds/ for darwin and
 * win32; Linux compiles build/Release instead). A typo'd or stale glob
 * matches nothing in the sources either and is not excused.
 */
function otherPlatformsOnly(match: Matcher, archive: AsarArchive): null | string {
  const staged = new Set<string>()

  for (const rel of archive.files.keys()) {
    const m = /^dist\/node_modules\/((?:@[^/]+\/)?[^/]+)\//.exec(rel)

    if (m) {
      staged.add(m[1])
    }
  }

  const desktopRequire = createRequire(path.join(DESKTOP_ROOT, 'package.json'))
  const hits: string[] = []

  for (const name of staged) {
    let root: string

    try {
      root = path.dirname(desktopRequire.resolve(`${name}/package.json`))
    } catch {
      continue
    }

    for (const rel of walkFiles(root)) {
      const asStaged = `dist/node_modules/${name}/${rel}`

      if (match(asStaged)) {
        hits.push(asStaged)
      }
    }
  }

  if (hits.length === 0) {
    return null
  }

  const platforms = new Set(hits.map(hit => PLATFORM_DIR_RE.exec(hit)?.[1] ?? '(none)'))

  if (platforms.has(process.platform) || platforms.has('(none)')) {
    return null
  }

  return `sources ship it only for ${[...platforms].sort().join(', ')} (${hits.length} files, e.g. ${hits[0]})`
}

interface GlobReport {
  glob: string
  archived: number
  onDisk: number
  note: string
}

function asarUnpackViolations(build: PackagedBuild): { problems: string[]; reports: GlobReport[] } {
  const globs = [...asArray(PKG.build.asarUnpack), ...asArray(PKG.build.linux?.asarUnpack)]
  const archive = readAsar(build.asarPath)
  const disk = walkFiles(build.unpackedRoot)
  const problems: string[] = []
  const reports: GlobReport[] = []

  if (globs.length === 0) {
    problems.push('package.json build.asarUnpack declares no patterns (nothing to verify; did the config move?)')
  }

  for (const glob of globs) {
    const match = packagerMatcher(glob)
    const archived = [...archive.files.values()].filter(entry => match(entry.rel))
    const onDisk = disk.filter(rel => match(rel))
    let note = ''

    const packed = archived.filter(entry => !entry.unpacked)

    if (packed.length > 0) {
      problems.push(
        `asarUnpack '${glob}': ${packed.length} matching file(s) are packed INSIDE app.asar, not unpacked ` +
          `(e.g. ${packed
            .slice(0, 3)
            .map(e => e.rel)
            .join(', ')})`
      )
    }

    const absent = archived.filter(entry => {
      if (!entry.unpacked) {
        return false
      }

      const file = path.join(build.unpackedRoot, entry.rel)

      try {
        return !entry.link && fs.statSync(file).size !== entry.size
      } catch {
        return true
      }
    })

    if (absent.length > 0) {
      problems.push(
        `asarUnpack '${glob}': ${absent.length} file(s) flagged unpacked in the archive header are missing ` +
          `(or truncated) under app.asar.unpacked (e.g. ${absent
            .slice(0, 3)
            .map(e => e.rel)
            .join(', ')})`
      )
    }

    if (onDisk.length === 0) {
      const excuse = otherPlatformsOnly(match, archive)

      if (excuse) {
        note = `empty on ${process.platform}: ${excuse}`
      } else {
        problems.push(
          `asarUnpack '${glob}' matched no file under ${build.unpackedRoot} ` +
            `(${archived.length} matching archive entries)`
        )
      }
    }

    reports.push({ glob, archived: archived.length, onDisk: onDisk.length, note })
  }

  return { problems, reports }
}

function mainEntryViolations(build: PackagedBuild): string[] {
  const archive = readAsar(build.asarPath)
  const raw = readArchived(build, archive, 'package.json')

  if (!raw) {
    return ['app.asar has no package.json: Electron cannot find the app entry']
  }

  const main = String((JSON.parse(raw.toString('utf8')) as { main?: string }).main ?? 'index.js')
  const rel = path.posix.normalize(main.replace(/^\.\//, ''))
  const entry = archive.files.get(rel)

  if (!entry) {
    return [`main entry '${main}' (from app.asar/package.json) is not in app.asar`]
  }

  const body = readArchived(build, archive, rel)

  if (!body || body.length === 0) {
    return [
      `main entry '${main}' is ${entry.unpacked ? 'flagged unpacked but missing/empty under app.asar.unpacked' : 'empty inside app.asar'}`
    ]
  }

  return []
}

// ─── Launching the packaged binary ──────────────────────────────────────

async function launchPackaged(
  build: PackagedBuild,
  env: Record<string, string>,
  cwd: string
): Promise<{ app: ElectronApplication; page: Page; logs: string[] }> {
  const logs: string[] = []

  const app = await _electron.launch({
    executablePath: build.executable,
    args: ['--disable-gpu', '--no-sandbox'],
    env,
    cwd
  })

  const collect = (chunk: Buffer) => {
    logs.push(...chunk.toString('utf8').split('\n').filter(Boolean))
    logs.splice(0, Math.max(0, logs.length - 200))
  }

  app.process().stdout?.on('data', collect)
  app.process().stderr?.on('data', collect)

  // A packaged main bundle that fails to load never opens a window; fail with
  // the main process's own story instead of a bare firstWindow timeout.
  const page = await app.firstWindow().catch(async error => {
    await app.close().catch(() => undefined)

    throw new Error(
      `packaged app never opened a window: ${(error as Error).message.split('\n')[0]}\n` +
        `packaged main-process log tail:\n${logs.slice(-60).join('\n')}`
    )
  })

  return { app, page, logs }
}

function assertPythonRuntime(): void {
  const candidates = ['.venv/bin/python', 'venv/bin/python'].map(rel => path.join(REPO_ROOT, rel))

  if (!candidates.some(file => fs.existsSync(file))) {
    throw new Error(
      `no Hermes Python venv at ${candidates.join(' or ')}: run 'uv sync' at the repo root ` +
        '(the packaged app runs the backend from HERMES_DESKTOP_HERMES_ROOT)'
    )
  }
}

// ─── Specs ──────────────────────────────────────────────────────────────

test('packaged build honours every declared asarUnpack pattern and ships its main entry', async () => {
  const build = packagedBuildOrSkip()
  const { problems, reports } = asarUnpackViolations(build)
  problems.push(...mainEntryViolations(build))

  for (const report of reports) {
    test.info().annotations.push({
      type: 'asarUnpack',
      description: `${report.glob}: ${report.archived} archived, ${report.onDisk} on disk${report.note ? ` (${report.note})` : ''}`
    })
  }

  expect(problems, `asarUnpack contract (${JSON.stringify(reports)})`).toEqual([])
})

test('packaged binary boots against a scripted provider and completes a first chat', async () => {
  const build = packagedBuildOrSkip()
  assertPythonRuntime()
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('packaged')
  writeProviderHome(sandbox.hermesHome, provider.url)
  let app: ElectronApplication | undefined

  try {
    const launched = await launchPackaged(build, coreAppEnv(sandbox), sandbox.root)
    const { app: electronApp, page, logs } = launched
    app = electronApp
    const ws = recordWebSockets(page)
    const logTail = () => logs.slice(-60).join('\n')

    await test.step('it is the packaged app, loaded from the archive', async () => {
      const facts = await electronApp.evaluate(({ app }) => ({ isPackaged: app.isPackaged, appPath: app.getAppPath() }))
      expect(facts, 'launched the packaged app, not a dev tree').toEqual({ isPackaged: true, appPath: build.asarPath })
    })

    await test.step('boots to an interactive composer', async () => {
      await waitForInteractive(electronApp, page).catch(error => {
        throw new Error(`${(error as Error).message}\npackaged main-process log tail:\n${logTail()}`)
      })
      await installDuplicateSampler(page)
    })

    await test.step('first turn: reply rendered and persisted', async () => {
      provider.script(U(1), [{ text: [`${A(1)} `, 'packaged ', 'reply'] }])
      await send(page, `${U(1)} hello from the packaged app`, 'Enter', ws)
      await expect
        .poll(() => provider.completions.some(c => c.marker === U(1) && c.finished), {
          timeout: 120_000,
          message: `provider finished ${U(1)}\n${logTail()}`
        })
        .toBe(true)
      await expect.poll(() => currentSessionId(page)).not.toBe('')
      const sessionId = await currentSessionId(page)
      await expect
        .poll(() => storedSessionForMarker(sandbox, 'default', U(1)), { message: 'first turn persisted to state.db' })
        .toBe(sessionId)
      await expect(page.getByText(`${A(1)} packaged reply`)).toBeVisible()
      await assertTranscriptOracle(page, ws, provider, { sessionId, expectUserMarkers: [U(1)] }, 'packaged first turn')
    })
  } finally {
    await app?.close().catch(() => undefined)

    // Never leave a test-made process behind (only this sandbox's).
    for (const proc of sandboxProcesses(sandbox)) {
      try {
        process.kill(proc.pid, 'SIGKILL')
      } catch {
        /* already gone */
      }
    }

    await provider.close()
    sandbox.cleanup()
  }
})

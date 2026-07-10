#!/usr/bin/env node
// ============================================================================
// build-runtime-bundle.mjs — ApexNodes prebuilt runtime bundle tool (hc-472 P1)
// ============================================================================
// Design: hermes-cloud docs/work-notes/DESIGN-hc472-runtime-bundle.md
// Root-cause: docs/work-notes/AUDIT-desktop-rootcause-2026-07-10.md F1/F5/F8 —
// installing = assembling a dev environment on the user's machine from 13
// network sources. This tool moves the assembly to CI: it prebuilds ONE
// self-contained, relocatable bundle per (os, arch) so a user install becomes
// "download 1 object + extract + atomic link switch".
//
// ApexNodes overlay file (upstream never creates it — zero merge surface).
// New install/update logic lives in Node, not PowerShell (design §7 / F8).
//
// Subcommands
//   build   — assemble a bundle for THIS machine's platform (native only:
//             venvs are platform-bound, design §3) and emit
//             runtime-bundle-<key>-<os>-<arch>.tar.gz + .sha256 + manifest.json
//   fixup   — stamp an extracted bundle for its current absolute location
//             (pyvenv.cfg home, editable-install finder paths, mac python
//             symlink). Idempotent; safe to re-run after every move.
//   verify  — re-hash an extracted bundle against its embedded files index
//             (skips the fixup-mutated files listed in the manifest).
//   smoke   — extract → fixup → verify → run interpreter/CLI/tool probes →
//             move the tree and re-run fixup + probes (proves relocatability).
//
// The bundle root IS the runtime source tree (matches today's
// HERMES_HOME/hermes-agent contract: pyproject.toml at root, venv/ inside),
// so this script ships inside every bundle at scripts/build-runtime-bundle.mjs
// and the manifest's fixup command invokes the bundled copy with the bundled
// node — no separate fixup binary to keep in sync.
//
// Layout produced (bundle root = versions/<key>/ once installed):
//   ./                      runtime source tree (git archive of the pinned ref)
//   ./venv/                 relocatable venv (uv venv --relocatable + uv sync)
//   ./.runtime/py/<name>/   python-build-standalone CPython (uv-managed)
//   ./.runtime/node/        portable Node 22 = npm -g prefix (agent-browser…)
//   ./.runtime/git/         MinGit (windows only)
//   ./.runtime/bin/         uv(.exe), uvx, rg(.exe)
//   ./.runtime/files.tsv    per-file sha256 index (verify)
//   ./.bundle-manifest.json embedded manifest (sibling manifest.json on COS
//                           adds the archive name/size/sha)
// ============================================================================

import { createHash } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const MANIFEST_SCHEMA = 1
const MANIFEST_NAME = '.bundle-manifest.json'
const FILES_INDEX_REL = '.runtime/files.tsv'
const FIXUP_STATE_REL = '.runtime/fixup-state.json'
const FRAMEWORK = 'hermes-agent'
const PYTHON_SERIES = '3.11' // lockstep: scripts/install.ps1 $PythonVersion / install.sh PYTHON_VERSION
const NODE_SERIES = 'v22.'   // lockstep: scripts/install.ps1 $NodeVersion = "22"
const MINGIT_VERSION = '2.54.0' // lockstep: install.ps1 $gitVer + publish-runtime-tarball.sh GIT_VERSION
const RIPGREP_VERSION = '14.1.1'
// Lockstep with scripts/install.{ps1,sh} agent-browser install (npm -g --prefix
// <node dir> --ignore-scripts). Keep the specs identical to the installers'.
const AGENT_BROWSER_SPECS = ['agent-browser@^0.26.0', '@askjo/camofox-browser@^1.5.2']

// ---------------------------------------------------------------------------
// small utils
// ---------------------------------------------------------------------------

function log(msg) { process.stdout.write(`→ ${msg}\n`) }
function warn(msg) { process.stdout.write(`⚠ ${msg}\n`) }
function die(msg) { process.stderr.write(`✗ ${msg}\n`); process.exit(1) }

function parseArgs(argv) {
  const args = { _: [] }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a.startsWith('--')) {
      const key = a.slice(2)
      const next = argv[i + 1]
      if (next === undefined || next.startsWith('--')) { args[key] = true } else { args[key] = next; i++ }
    } else {
      args._.push(a)
    }
  }
  return args
}

function run(cmd, argv, opts = {}) {
  const pretty = `${cmd} ${argv.join(' ')}`
  log(`$ ${pretty}`)
  const res = spawnSync(cmd, argv, {
    stdio: opts.capture ? ['ignore', 'pipe', 'pipe'] : 'inherit',
    cwd: opts.cwd,
    env: opts.env || process.env,
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
    shell: false,
  })
  if (res.error) die(`${pretty} failed to launch: ${res.error.message}`)
  if (res.status !== 0 && !opts.allowFail) {
    if (opts.capture) process.stderr.write(String(res.stderr || ''))
    die(`${pretty} exited with ${res.status}`)
  }
  return res
}

function sha256File(file) {
  const h = createHash('sha256')
  const fd = fs.openSync(file, 'r')
  const buf = Buffer.alloc(4 * 1024 * 1024)
  try {
    let n
    while ((n = fs.readSync(fd, buf, 0, buf.length)) > 0) h.update(buf.subarray(0, n))
  } finally { fs.closeSync(fd) }
  return h.digest('hex')
}

function sha256Text(text) { return createHash('sha256').update(text).digest('hex') }

// curl, not node fetch: curl exists on every GH runner (and Win10+ System32),
// streams to disk, retries, and honors https_proxy/no_proxy env — node's
// undici fetch ignores proxy env entirely, which hard-fails on proxied
// networks (observed on a mainland-CN build host).
function curlHeaderArgs(headers) {
  const args = []
  for (const [k, v] of Object.entries(headers)) if (v) args.push('-H', `${k}: ${v}`)
  return args
}

async function download(url, dest, { headers = {}, attempts = 3, timeoutSec = 900 } = {}) {
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  // Idempotent: a pre-seeded / previously fetched artifact is reused. CI
  // runners start empty so this never masks staleness there; locally it lets
  // a slow-network seat seed .tools/ from the COS re-hosts.
  if (fs.existsSync(dest) && fs.statSync(dest).size > 0) {
    log(`cached ${human(fs.statSync(dest).size)} ${path.basename(dest)} (skipping fetch)`)
    return dest
  }
  const tmp = `${dest}.part`
  rmrf(tmp)
  log(`fetch ${url}`)
  const res = spawnSync('curl', [
    '-fL', '--silent', '--show-error',
    '--retry', String(attempts), '--retry-delay', '2', '--retry-all-errors',
    '--connect-timeout', '30', '--max-time', String(timeoutSec),
    ...curlHeaderArgs(headers),
    '-o', tmp, url,
  ], { stdio: ['ignore', 'inherit', 'pipe'], encoding: 'utf8' })
  if (res.status !== 0) {
    rmrf(tmp)
    throw new Error(`curl failed (${res.status}) for ${url}: ${(res.stderr || '').trim()}`)
  }
  fs.renameSync(tmp, dest)
  log(`fetched ${human(fs.statSync(dest).size)} -> ${path.basename(dest)}`)
  return dest
}

async function fetchJson(url, { headers = {} } = {}) {
  const res = spawnSync('curl', [
    '-fsSL', '--retry', '3', '--retry-delay', '2', '--connect-timeout', '20', '--max-time', '120',
    ...curlHeaderArgs(headers),
    url,
  ], { stdio: ['ignore', 'pipe', 'pipe'], encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 })
  if (res.status !== 0) throw new Error(`curl failed (${res.status}) for ${url}: ${(res.stderr || '').trim()}`)
  return JSON.parse(res.stdout)
}

function githubHeaders() {
  const h = { 'user-agent': 'apexnodes-runtime-bundle', accept: 'application/vnd.github+json' }
  if (process.env.GITHUB_TOKEN) h.authorization = `Bearer ${process.env.GITHUB_TOKEN}`
  return h
}

// Host bsdtar. Windows: ALWAYS System32\tar.exe (bsdtar, Win10 1803+) — the
// same binary user machines extract with (contract test) and immune to the
// GNU-tar "C: looks like a remote host" trap (see Install-RuntimeFromCos).
function tarBin() {
  if (process.platform === 'win32') {
    const sys = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'tar.exe')
    if (!fs.existsSync(sys)) die(`System32 tar.exe not found at ${sys}`)
    return sys
  }
  return 'tar'
}

function rmrf(p) { fs.rmSync(p, { recursive: true, force: true }) }

function* walk(dir, rel = '') {
  const entries = fs.readdirSync(dir, { withFileTypes: true })
  // Sort for a deterministic files.tsv ordering.
  entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
  for (const e of entries) {
    const abs = path.join(dir, e.name)
    const r = rel ? `${rel}/${e.name}` : e.name
    if (e.isSymbolicLink()) { yield { rel: r, abs, type: 'link' } } else if (e.isDirectory()) { yield* walk(abs, r) } else if (e.isFile()) { yield { rel: r, abs, type: 'file' } }
  }
}

// ---------------------------------------------------------------------------
// platform targets (native builds only — design §3)
// ---------------------------------------------------------------------------

function detectTarget() {
  if (process.platform === 'win32') {
    if (process.arch !== 'x64') die(`unsupported windows arch for P1: ${process.arch}`)
    return {
      os: 'win', arch: 'x64',
      uvTriple: 'x86_64-pc-windows-msvc', uvExt: 'zip',
      nodePlat: 'win-x64', nodeExt: 'zip',
      rgTriple: 'x86_64-pc-windows-msvc', rgExt: 'zip',
      exe: '.exe',
      venvBin: ['venv', 'Scripts'],
      sitePackages: ['venv', 'Lib', 'site-packages'],
    }
  }
  if (process.platform === 'darwin') {
    const arm = process.arch === 'arm64'
    return {
      os: 'mac', arch: arm ? 'arm64' : 'x64',
      uvTriple: arm ? 'aarch64-apple-darwin' : 'x86_64-apple-darwin', uvExt: 'tar.gz',
      nodePlat: arm ? 'darwin-arm64' : 'darwin-x64', nodeExt: 'tar.gz',
      rgTriple: arm ? 'aarch64-apple-darwin' : 'x86_64-apple-darwin', rgExt: 'tar.gz',
      exe: '',
      venvBin: ['venv', 'bin'],
      sitePackages: null, // resolved after venv exists (needs python3.X dir name)
    }
  }
  die(`unsupported build platform: ${process.platform} (P1 targets win-x64 / mac-arm64 / mac-x64)`)
}

function resolveSitePackages(root, target) {
  if (target.os === 'win') return path.join(root, 'venv', 'Lib', 'site-packages')
  const libDir = path.join(root, 'venv', 'lib')
  const py = fs.readdirSync(libDir).find((n) => n.startsWith('python'))
  if (!py) die(`no python* dir under ${libDir}`)
  return path.join(libDir, py, 'site-packages')
}

// ---------------------------------------------------------------------------
// build
// ---------------------------------------------------------------------------

async function cmdBuild(args) {
  const target = detectTarget()
  const repoRoot = path.resolve(__dirname, '..')
  const outDir = path.resolve(args.out || path.join(repoRoot, 'dist', 'runtime-bundle'))
  const ref = args.ref || 'HEAD'
  const minDesktopVersion = args['min-desktop-version'] || '0.17.0'

  const sha = run('git', ['-C', repoRoot, 'rev-parse', `${ref}^{commit}`], { capture: true }).stdout.trim()
  if (!/^[0-9a-f]{40}$/.test(sha)) die(`could not resolve ref '${ref}' to a commit sha`)
  const key = sha.slice(0, 12)
  const bundleBase = `runtime-bundle-${key}-${target.os}-${target.arch}`

  const stage = path.join(outDir, 'stage')
  const tools = path.join(outDir, '.tools')
  rmrf(stage)
  fs.mkdirSync(stage, { recursive: true })
  fs.mkdirSync(tools, { recursive: true })
  log(`building ${bundleBase} (ref ${ref} = ${sha})`)
  log(`stage: ${stage}`)

  // ── 1. runtime source (git archive: clean tree, no .git — same semantics as
  //       publish-runtime-tarball.sh, but extracted at bundle ROOT). ──────────
  const srcTar = path.join(tools, 'src.tar')
  run('git', ['-C', repoRoot, 'archive', '--format=tar', '-o', srcTar, sha])
  run(tarBin(), ['-xf', srcTar, '-C', stage])
  fs.rmSync(srcTar)
  // dist/cos-runtime holds an accidentally-committed 52MB source tarball (a
  // coscli-era publish artifact). Never ship it inside bundles; repo removal
  // is a separate cleanup.
  rmrf(path.join(stage, 'dist', 'cos-runtime'))

  // ── 2. uv (host tool + bundled binary; native build ⇒ same triple) ────────
  let uvVersion = args['uv-version'] || ''
  if (!uvVersion) {
    const rel = await fetchJson('https://api.github.com/repos/astral-sh/uv/releases/latest', { headers: githubHeaders() })
    uvVersion = rel.tag_name
  }
  const uvArchive = path.join(tools, `uv.${target.uvExt}`)
  await download(
    `https://github.com/astral-sh/uv/releases/download/${uvVersion}/uv-${target.uvTriple}.${target.uvExt}`,
    uvArchive, { headers: githubHeaders() }
  )
  const uvUnpack = path.join(tools, 'uv')
  rmrf(uvUnpack); fs.mkdirSync(uvUnpack, { recursive: true })
  run(tarBin(), ['-xf', uvArchive, '-C', uvUnpack])
  const findBin = (dir, name) => {
    for (const f of walk(dir)) if (f.type === 'file' && path.basename(f.rel) === name) return f.abs
    return null
  }
  const uvHost = findBin(uvUnpack, `uv${target.exe}`)
  if (!uvHost) die(`uv${target.exe} not found inside ${uvArchive}`)
  fs.chmodSync(uvHost, 0o755)
  // Truth from the artifact, not the tag lookup (a pre-seeded binary may
  // differ from the 'latest' the API reported).
  const uvVersionActual = run(uvHost, ['--version'], { capture: true }).stdout.trim().split(/\s+/)[1] || uvVersion
  uvVersion = uvVersionActual

  // ── 3. CPython (python-build-standalone via uv, into the bundle) ──────────
  const pyRoot = path.join(stage, '.runtime', 'py')
  fs.mkdirSync(pyRoot, { recursive: true })
  run(uvHost, ['python', 'install', PYTHON_SERIES], {
    env: { ...process.env, UV_PYTHON_INSTALL_DIR: pyRoot },
  })
  // Canonicalize the managed-python dir. uv leaves (a) an ABSOLUTE-symlink
  // minor-version alias (cpython-3.11 -> /abs/.../cpython-3.11.15) that would
  // dangle on every relocation, and (b) .lock/.temp/.gitignore bookkeeping.
  // Keep exactly one REAL fully-versioned dir; drop everything else so the
  // bundle carries zero absolute references here.
  let pyName = null
  for (const name of fs.readdirSync(pyRoot)) {
    const p = path.join(pyRoot, name)
    const isRealVersionedDir = /^cpython-\d+\.\d+\.\d+/.test(name) && !fs.lstatSync(p).isSymbolicLink() && fs.lstatSync(p).isDirectory()
    if (isRealVersionedDir) {
      if (pyName) die(`multiple managed pythons under ${pyRoot}: ${pyName}, ${name}`)
      pyName = name
    } else {
      rmrf(p)
      log(`pruned managed-python artifact: .runtime/py/${name}`)
    }
  }
  if (!pyName) die(`no cpython-* dir under ${pyRoot} after uv python install`)
  const pyDirRel = `.runtime/py/${pyName}`
  const pythonExe = target.os === 'win'
    ? path.join(pyRoot, pyName, 'python.exe')
    : path.join(pyRoot, pyName, 'bin', `python${PYTHON_SERIES}`)
  if (!fs.existsSync(pythonExe)) die(`bundled python not found at ${pythonExe}`)
  const pyVersion = run(pythonExe, ['-c', 'import platform;print(platform.python_version())'], { capture: true }).stdout.trim()

  // ── 4. relocatable venv on the bundled interpreter ─────────────────────────
  const venvDir = path.join(stage, 'venv')
  const buildEnv = {
    ...process.env,
    UV_PYTHON_DOWNLOADS: 'never',
    UV_PYTHON_INSTALL_DIR: pyRoot,
  }
  delete buildEnv.VIRTUAL_ENV
  delete buildEnv.PYTHONPATH
  delete buildEnv.PYTHONHOME
  run(uvHost, ['venv', venvDir, '--relocatable', '--python', pythonExe], { env: buildEnv })
  const pyvenvCfg = fs.readFileSync(path.join(venvDir, 'pyvenv.cfg'), 'utf8')
  if (!/^relocatable\s*=\s*true$/m.test(pyvenvCfg)) die('uv venv did not record relocatable = true in pyvenv.cfg')

  // mac: uv links venv/bin/python -> ABSOLUTE bundled-python path. Re-link
  // relative at pack time so tar carries a location-independent symlink.
  if (target.os === 'mac') relinkMacPython(stage, pyDirRel)

  // ── 5. python deps — EXACTLY the installer's hash-verified premium tier:
  //       `uv sync --extra all --locked` (NOT --all-extras; see install.sh).
  //       Index env is SANITIZED: a mirror index (UV_DEFAULT_INDEX=TUNA etc.)
  //       re-keys package sources and makes --locked fail "lockfile needs to
  //       be updated" (verified with uv 0.11.28) — a bundle must always be
  //       resolved against the lock's recorded registry, host env be damned.
  //       --dev-unlocked (local validation ONLY): keeps host index env, drops
  //       --locked, and stamps dev_unlocked into the manifest so such a
  //       bundle is tamper-evident and can never ship silently. ──────────────
  const devUnlocked = Boolean(args['dev-unlocked'])
  const venvPython = target.os === 'win'
    ? path.join(venvDir, 'Scripts', 'python.exe')
    : path.join(venvDir, 'bin', 'python')
  const syncEnv = { ...buildEnv, UV_PROJECT_ENVIRONMENT: venvDir, UV_PYTHON: venvPython }
  if (devUnlocked) {
    warn('--dev-unlocked: NOT hash-verified against uv.lock; manifest will carry dev_unlocked=true')
  } else {
    for (const k of ['UV_DEFAULT_INDEX', 'UV_INDEX_URL', 'UV_EXTRA_INDEX_URL', 'UV_INDEX', 'PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL']) delete syncEnv[k]
  }
  run(uvHost, ['sync', '--extra', 'all', ...(devUnlocked ? [] : ['--locked'])], { cwd: stage, env: syncEnv })

  // ── 6. portable Node 22 (dist layout == today's HERMES_HOME/node).
  //       HERMES_NODE_DIST_BASE: same override env install.ps1's Install-Node
  //       honors (CN mirror = npmmirror binary mirror). ──────────────────────
  const nodeDistBase = (process.env.HERMES_NODE_DIST_BASE || 'https://nodejs.org/dist').replace(/\/+$/, '')
  const nodeIndex = await fetchJson(`${nodeDistBase}/index.json`)
  const nodeEntry = nodeIndex.find((e) => e.version.startsWith(NODE_SERIES))
  if (!nodeEntry) die(`no ${NODE_SERIES}x release in ${nodeDistBase}/index.json`)
  const nodeVersion = nodeEntry.version
  const nodeBase = `node-${nodeVersion}-${target.nodePlat}`
  const nodeArchive = path.join(tools, `${nodeBase}.${target.nodeExt}`)
  await download(`${nodeDistBase}/${nodeVersion}/${nodeBase}.${target.nodeExt}`, nodeArchive)
  const nodeUnpack = path.join(tools, 'node')
  rmrf(nodeUnpack); fs.mkdirSync(nodeUnpack, { recursive: true })
  run(tarBin(), ['-xf', nodeArchive, '-C', nodeUnpack])
  const nodeStage = path.join(stage, '.runtime', 'node')
  fs.renameSync(path.join(nodeUnpack, nodeBase), nodeStage)
  const nodeExe = target.os === 'win' ? path.join(nodeStage, 'node.exe') : path.join(nodeStage, 'bin', 'node')
  const npmCli = path.join(nodeStage, target.os === 'win' ? 'node_modules' : 'lib/node_modules', 'npm', 'bin', 'npm-cli.js')
  if (!fs.existsSync(nodeExe) || !fs.existsSync(npmCli)) die(`staged node incomplete under ${nodeStage}`)

  // npm invocations use the STAGED node (bundle-faithful; ABI == bundle).
  const npmEnv = {
    ...buildEnv,
    npm_config_cache: path.join(tools, 'npm-cache'),
    npm_config_update_notifier: 'false',
    npm_config_fund: 'false',
    npm_config_audit: 'false',
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD: '1', // Chromium is a lazy COS attachment, never in-bundle (design §2)
    PATH: `${path.dirname(nodeExe)}${path.delimiter}${process.env.PATH}`,
  }
  const npm = (argv, opts = {}) => run(nodeExe, [npmCli, ...argv], { ...opts, env: { ...npmEnv, ...(opts.env || {}) } })

  // ── 7. agent-browser + camofox: npm -g --prefix <node dir> --ignore-scripts
  //       (byte-for-byte the installers' invocation, prefix = bundled node). ──
  npm(['install', '-g', '--prefix', nodeStage, '--ignore-scripts', ...AGENT_BROWSER_SPECS])

  // ── 8. repo npm trees. Deliberate A2 curation (NOT today's accidental
  //       full-workspace install): root deps (browser tool resolution) +
  //       ui-tui workspace (hermes --tui; tui_dist is NOT committed). Both
  //       resolve from the root package-lock.json. web/ and apps/* are
  //       excluded — the packaged shell ships its own Electron. ──────────────
  npm(['install', '--workspaces=false', '--silent'], { cwd: stage })
  npm(['install', '--workspace', 'ui-tui', '--silent'], { cwd: stage })

  // ── 9. MinGit (windows only; replaces PortableGit — bundle installs never
  //       git-clone, runtime shell-outs only. design §2). ────────────────────
  let gitComponent = null
  if (target.os === 'win') {
    const gitDir = path.join(stage, '.runtime', 'git')
    const relTag = `v${MINGIT_VERSION}.windows.1`
    const asset = `MinGit-${MINGIT_VERSION}-64-bit.zip`
    const gitArchive = path.join(tools, asset)
    await download(`https://github.com/git-for-windows/git/releases/download/${relTag}/${asset}`, gitArchive, { headers: githubHeaders() })
    fs.mkdirSync(gitDir, { recursive: true })
    run(tarBin(), ['-xf', gitArchive, '-C', gitDir])
    const gitExe = path.join(gitDir, 'cmd', 'git.exe')
    if (!fs.existsSync(gitExe)) die(`MinGit missing cmd/git.exe under ${gitDir}`)
    gitComponent = { path: '.runtime/git', version: MINGIT_VERSION, flavor: 'MinGit' }
  }

  // ── 10. small tools: uv + rg into .runtime/bin ─────────────────────────────
  const binDir = path.join(stage, '.runtime', 'bin')
  fs.mkdirSync(binDir, { recursive: true })
  fs.copyFileSync(uvHost, path.join(binDir, `uv${target.exe}`))
  fs.chmodSync(path.join(binDir, `uv${target.exe}`), 0o755)
  const uvxHost = findBin(uvUnpack, `uvx${target.exe}`)
  if (uvxHost) { fs.copyFileSync(uvxHost, path.join(binDir, `uvx${target.exe}`)); fs.chmodSync(path.join(binDir, `uvx${target.exe}`), 0o755) }
  const rgArchiveName = `ripgrep-${RIPGREP_VERSION}-${target.rgTriple}.${target.rgExt}`
  const rgArchive = path.join(tools, rgArchiveName)
  await download(`https://github.com/BurntSushi/ripgrep/releases/download/${RIPGREP_VERSION}/${rgArchiveName}`, rgArchive, { headers: githubHeaders() })
  const rgUnpack = path.join(tools, 'rg')
  rmrf(rgUnpack); fs.mkdirSync(rgUnpack, { recursive: true })
  run(tarBin(), ['-xf', rgArchive, '-C', rgUnpack])
  const rgBin = findBin(rgUnpack, `rg${target.exe}`)
  if (!rgBin) die(`rg${target.exe} not found inside ${rgArchiveName}`)
  fs.copyFileSync(rgBin, path.join(binDir, `rg${target.exe}`))
  fs.chmodSync(path.join(binDir, `rg${target.exe}`), 0o755)

  // ── 11. prune build detritus + neutralize location-bound links ────────────
  prunePycache(stage)
  normalizeStageLinks(stage)

  // ── 12. chromium lazy hook metadata (env reservation, design §2/A2) ───────
  const playwrightVersion = detectPlaywrightVersion(stage, target)

  // ── 13-14. files index + embedded manifest ─────────────────────────────────
  const mutates = fixupMutatedFiles(stage, target)
  const filesIndex = writeFilesIndex(stage)
  const manifest = {
    schema: MANIFEST_SCHEMA,
    kind: 'apexnodes-runtime-bundle',
    framework: FRAMEWORK,
    key,
    runtime_commit: sha,
    os: target.os,
    arch: target.arch,
    format: 'tar.gz',
    created_at: new Date().toISOString(),
    min_desktop_version: minDesktopVersion,
    ...(devUnlocked ? { dev_unlocked: true } : {}),
    build_root: stage,
    components: {
      src: { path: '.', note: 'runtime source tree at bundle root (git archive, no .git)' },
      python: { path: pyDirRel, version: pyVersion, source: 'python-build-standalone (uv-managed)' },
      venv: { path: 'venv', relocatable: true, sync: 'uv sync --extra all --locked', uv_version: uvVersion },
      node: { path: '.runtime/node', version: nodeVersion, npm_prefix: true, globals: AGENT_BROWSER_SPECS },
      npm_trees: { root: 'node_modules (--workspaces=false)', 'ui-tui': 'hoisted via npm install -w ui-tui' },
      ...(gitComponent ? { git: gitComponent } : {}),
      uv: { path: `.runtime/bin/uv${target.exe}`, version: uvVersion },
      ripgrep: { path: `.runtime/bin/rg${target.exe}`, version: RIPGREP_VERSION },
    },
    chromium: {
      included: false,
      mode: 'lazy-cos',
      playwright_version: playwrightVersion,
      env: { PLAYWRIGHT_BROWSERS_PATH: '{bundle_root}/.runtime/chromium' },
      cos_key_hint: `bundle/${FRAMEWORK}/${key}/${target.os}-${target.arch}/chromium-${playwrightVersion || '<pwver>'}.tar.gz`,
    },
    fixup: {
      script: 'scripts/build-runtime-bundle.mjs',
      command: `{bundle_root}/.runtime/node/${target.os === 'win' ? 'node.exe' : 'bin/node'} {bundle_root}/scripts/build-runtime-bundle.mjs fixup --root {bundle_root}`,
      mutates,
    },
    files_index: filesIndex,
  }
  fs.writeFileSync(path.join(stage, MANIFEST_NAME), JSON.stringify(manifest, null, 2) + '\n')

  // ── 15-16. archive + sha + sibling manifest ────────────────────────────────
  fs.mkdirSync(outDir, { recursive: true })
  const archivePath = path.join(outDir, `${bundleBase}.tar.gz`)
  rmrf(archivePath)
  log('creating archive (this can take a few minutes)...')
  run(tarBin(), ['-czf', archivePath, '-C', stage, '.'])
  const archiveSha = sha256File(archivePath)
  const archiveSize = fs.statSync(archivePath).size
  fs.writeFileSync(path.join(outDir, `${bundleBase}.tar.gz.sha256`), `${archiveSha}  ${bundleBase}.tar.gz\n`)
  const sibling = { ...manifest, archive: { name: `${bundleBase}.tar.gz`, sha256: archiveSha, size: archiveSize } }
  fs.writeFileSync(path.join(outDir, `${bundleBase}.manifest.json`), JSON.stringify(sibling, null, 2) + '\n')

  const stageBytes = duBytes(stage)
  const summary = {
    bundle: `${bundleBase}.tar.gz`,
    key,
    runtime_commit: sha,
    os: target.os,
    arch: target.arch,
    python: pyVersion,
    node: nodeVersion,
    uv: uvVersion,
    archive_bytes: archiveSize,
    archive_human: human(archiveSize),
    unpacked_bytes: stageBytes,
    unpacked_human: human(stageBytes),
    files: manifest.files_index.count,
    sha256: archiveSha,
    ...(devUnlocked ? { dev_unlocked: true } : {}),
  }
  log(`BUNDLE_SUMMARY ${JSON.stringify(summary)}`)
  // machine-readable for the workflow
  fs.writeFileSync(path.join(outDir, `${bundleBase}.summary.json`), JSON.stringify(summary, null, 2) + '\n')
  if (!args['keep-stage']) rmrf(stage)
  rmrf(tools)
  log(`done: ${archivePath} (${human(archiveSize)})`)
}

function relinkMacPython(root, pyDirRel) {
  const bin = path.join(root, 'venv', 'bin')
  const relTarget = path.join('..', '..', ...pyDirRel.split('/'), 'bin', `python${PYTHON_SERIES}`)
  for (const name of fs.readdirSync(bin)) {
    if (!/^python(\d(\.\d+)?)?$/.test(name)) continue
    const p = path.join(bin, name)
    const st = fs.lstatSync(p)
    if (!st.isSymbolicLink()) continue
    const cur = fs.readlinkSync(p)
    if (path.isAbsolute(cur)) {
      // `python` carries the absolute base-interpreter target; python3/python3.11
      // chain to `python` relatively and stay as-is.
      fs.rmSync(p)
      fs.symlinkSync(relTarget, p)
      log(`re-linked venv/bin/${name} -> ${relTarget}`)
    }
  }
}

// Absolute link targets are location-bound and dangle after relocation — the
// exact failure the win-x64 CI verify caught on npm's workspace links
// (node_modules/@hermes/ink, node_modules/hermes-tui): npm materializes
// workspaces as JUNCTIONS on Windows, and junctions cannot be relative.
//   windows: replace every in-stage-targeted link with a real copy of its
//            target (bundle is immutable, so the copy can never drift).
//   posix:   rewrite absolute in-stage symlinks as equivalent relative links.
// Links pointing OUTSIDE the stage would be a build bug — fail loudly.
// Fixpoint loop: a copied target may itself contain links.
// Remove a symlink/junction WITHOUT following it: unlink for file symlinks,
// rmdir for directory junctions / dir-symlinks (both delete only the reparse
// point, never the target's contents).
function removeLinkOnly(p) {
  try { fs.unlinkSync(p) } catch { fs.rmdirSync(p) }
}

function normalizeStageLinks(root) {
  for (let pass = 0; pass < 4; pass++) {
    let changed = 0
    for (const f of [...walk(root)]) {
      if (f.type !== 'link') continue
      const target = fs.readlinkSync(f.abs)
      if (!path.isAbsolute(target)) continue
      const resolved = path.resolve(path.dirname(f.abs), target)
      if (!(resolved === root || resolved.startsWith(root + path.sep))) {
        die(`link ${f.rel} points outside the stage (${target}) — refusing to ship a machine-bound bundle`)
      }
      if (process.platform === 'win32') {
        removeLinkOnly(f.abs) // NEVER recursive: rm -r through a junction deletes the TARGET's contents
        fs.cpSync(resolved, f.abs, { recursive: true })
        log(`dereferenced windows junction: ${f.rel} -> copy of ${path.relative(root, resolved)}`)
      } else {
        const rel = path.relative(path.dirname(f.abs), resolved)
        removeLinkOnly(f.abs)
        fs.symlinkSync(rel, f.abs)
        log(`re-linked ${f.rel} -> ${rel}`)
      }
      changed++
    }
    if (changed === 0) return
  }
  die('normalizeStageLinks did not reach a fixpoint in 4 passes')
}

function prunePycache(root) {
  let n = 0
  const stack = [root]
  while (stack.length) {
    const dir = stack.pop()
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      if (!e.isDirectory() || e.isSymbolicLink()) continue
      const p = path.join(dir, e.name)
      if (e.name === '__pycache__') { rmrf(p); n++ } else { stack.push(p) }
    }
  }
  log(`pruned ${n} __pycache__ dirs`)
}

function detectPlaywrightVersion(stage, target) {
  const gmods = path.join(stage, '.runtime', 'node', ...(target.os === 'win' ? ['node_modules'] : ['lib', 'node_modules']))
  const candidates = [
    // observed layout: playwright-core nests under the camofox global
    path.join(gmods, '@askjo', 'camofox-browser', 'node_modules', 'playwright-core', 'package.json'),
    path.join(gmods, 'agent-browser', 'node_modules', 'playwright-core', 'package.json'),
    path.join(gmods, 'playwright-core', 'package.json'),
    path.join(stage, 'node_modules', 'playwright-core', 'package.json'),
    path.join(stage, 'node_modules', 'playwright', 'package.json'),
  ]
  for (const c of candidates) {
    try { return JSON.parse(fs.readFileSync(c, 'utf8')).version } catch { /* next */ }
  }
  return null
}

// The exact files fixup rewrites — verify must skip them (their content is
// location-dependent by design).
function fixupMutatedFiles(root, target) {
  const sp = resolveSitePackages(root, target)
  const spRel = path.relative(root, sp).split(path.sep).join('/')
  const out = ['venv/pyvenv.cfg', FIXUP_STATE_REL]
  for (const name of fs.readdirSync(sp)) {
    if (name.startsWith('__editable__') && (name.endsWith('.py') || name.endsWith('.pth'))) out.push(`${spRel}/${name}`)
    if (name.endsWith('.dist-info')) {
      const du = path.join(sp, name, 'direct_url.json')
      if (fs.existsSync(du)) out.push(`${spRel}/${name}/direct_url.json`)
    }
  }
  return out.sort()
}

function writeFilesIndex(root) {
  log('hashing bundle contents for files.tsv ...')
  const lines = []
  let count = 0
  let total = 0
  for (const f of walk(root)) {
    if (f.rel === FILES_INDEX_REL || f.rel === MANIFEST_NAME) continue
    if (f.type === 'link') {
      lines.push(`${f.rel}\tlink\t0\t${fs.readlinkSync(f.abs)}`)
    } else {
      const st = fs.statSync(f.abs)
      lines.push(`${f.rel}\tfile\t${st.size}\t${sha256File(f.abs)}`)
      total += st.size
    }
    count++
  }
  const text = lines.join('\n') + '\n'
  const dest = path.join(root, FILES_INDEX_REL)
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  fs.writeFileSync(dest, text)
  return { path: FILES_INDEX_REL, sha256: sha256Text(text), count, total_size: total }
}

function duBytes(root) {
  let total = 0
  for (const f of walk(root)) if (f.type === 'file') total += fs.statSync(f.abs).size
  return total
}

function human(bytes) {
  if (bytes > 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`
  if (bytes > 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`
  return `${bytes} B`
}

// ---------------------------------------------------------------------------
// fixup — stamp an extracted bundle for its current location. Idempotent.
// ---------------------------------------------------------------------------

function loadManifest(root) {
  const p = path.join(root, MANIFEST_NAME)
  if (!fs.existsSync(p)) die(`${MANIFEST_NAME} not found under ${root} — not an extracted bundle?`)
  return JSON.parse(fs.readFileSync(p, 'utf8'))
}

function pathForms(p) {
  // A path can be embedded raw, backslash-escaped (py/json source) or
  // forward-slashed. Rewrite all three spellings.
  return [p, p.split('\\').join('\\\\'), p.split('\\').join('/')]
}

function cmdFixup(args) {
  const root = path.resolve(args.root || '.')
  const manifest = loadManifest(root)
  const target = manifest.os === 'win'
    ? { os: 'win' }
    : { os: 'mac' }

  // Previous location: fixup-state (already stamped once) else build_root.
  const statePath = path.join(root, FIXUP_STATE_REL)
  let previousRoot = manifest.build_root
  if (fs.existsSync(statePath)) {
    try { previousRoot = JSON.parse(fs.readFileSync(statePath, 'utf8')).current_root || previousRoot } catch { /* keep build_root */ }
  }

  // 1) pyvenv.cfg home -> the bundled interpreter's directory at THIS location.
  const pyRel = manifest.components.python.path
  const homeDir = manifest.os === 'win'
    ? path.join(root, ...pyRel.split('/'))
    : path.join(root, ...pyRel.split('/'), 'bin')
  const cfgPath = path.join(root, 'venv', 'pyvenv.cfg')
  const cfg = fs.readFileSync(cfgPath, 'utf8')
  const newCfg = cfg.replace(/^home\s*=.*$/m, `home = ${homeDir}`)
  if (newCfg !== cfg) fs.writeFileSync(cfgPath, newCfg)
  log(`pyvenv.cfg home = ${homeDir}`)

  // 2) editable-install artifacts: literal path replace previousRoot -> root.
  let rewrites = 0
  if (previousRoot && previousRoot !== root) {
    const oldForms = pathForms(previousRoot)
    const newForms = pathForms(root)
    for (const rel of manifest.fixup.mutates) {
      if (rel === 'venv/pyvenv.cfg' || rel === FIXUP_STATE_REL) continue
      const p = path.join(root, ...rel.split('/'))
      if (!fs.existsSync(p)) continue
      let text = fs.readFileSync(p, 'utf8')
      let changed = false
      for (let i = 0; i < oldForms.length; i++) {
        if (text.includes(oldForms[i])) { text = text.split(oldForms[i]).join(newForms[i]); changed = true }
      }
      if (changed) { fs.writeFileSync(p, text); rewrites++ }
    }
  }
  log(`rewrote ${rewrites} editable-install file(s) (${previousRoot === root ? 'already at this root' : `from ${previousRoot}`})`)

  // 3) mac: venv/bin/python must be a RELATIVE symlink into the bundle.
  // lstat (not existsSync): on a user machine an absolute build-host target is
  // a BROKEN symlink — existsSync follows it and reports false, which would
  // skip exactly the repair this exists for.
  if (target.os === 'mac') {
    const p = path.join(root, 'venv', 'bin', 'python')
    let st = null
    try { st = fs.lstatSync(p) } catch { st = null }
    if (st && st.isSymbolicLink() && path.isAbsolute(fs.readlinkSync(p))) {
      const relTarget = path.join('..', '..', ...pyRel.split('/'), 'bin', `python${PYTHON_SERIES}`)
      fs.rmSync(p)
      fs.symlinkSync(relTarget, p)
      log(`re-linked venv/bin/python -> ${relTarget}`)
    }
  }

  fs.writeFileSync(statePath, JSON.stringify({ current_root: root, stamped_at: new Date().toISOString() }, null, 2) + '\n')
  log(`fixup complete: ${root}`)
}

// ---------------------------------------------------------------------------
// verify — re-hash against files.tsv (skips fixup-mutated files).
// ---------------------------------------------------------------------------

function cmdVerify(args) {
  const root = path.resolve(args.root || '.')
  const manifest = loadManifest(root)
  const skip = new Set([...manifest.fixup.mutates, FILES_INDEX_REL, MANIFEST_NAME])
  const tsvPath = path.join(root, FILES_INDEX_REL)
  const text = fs.readFileSync(tsvPath, 'utf8')
  if (sha256Text(text) !== manifest.files_index.sha256) die('files.tsv does not match the sha recorded in the manifest')
  let checked = 0
  const bad = []
  for (const line of text.split('\n')) {
    if (!line) continue
    const [rel, type, sizeStr, digest] = line.split('\t')
    if (skip.has(rel)) continue
    const p = path.join(root, ...rel.split('/'))
    if (type === 'link') {
      let ok = false
      try { ok = fs.readlinkSync(p) === digest } catch { ok = false }
      // fixup may legitimately re-point venv python symlinks; treat venv/bin
      // python links as mutable.
      if (!ok && !/^venv\/bin\/python/.test(rel)) bad.push(`${rel} (symlink target changed)`)
    } else {
      let ok = false
      try { ok = fs.statSync(p).size === Number(sizeStr) && sha256File(p) === digest } catch { ok = false }
      if (!ok) bad.push(rel)
    }
    checked++
    if (checked % 20000 === 0) log(`verified ${checked} entries...`)
  }
  if (bad.length) {
    for (const b of bad.slice(0, 40)) warn(`MISMATCH: ${b}`)
    die(`verify FAILED: ${bad.length} of ${checked} entries mismatched`)
  }
  log(`verify OK: ${checked} entries match (skipped ${skip.size} fixup-mutable files)`)
}

// ---------------------------------------------------------------------------
// smoke — extract → fixup → verify → probes → move → re-fixup → probes.
// ---------------------------------------------------------------------------

function cmdSmoke(args) {
  if (!args.archive) die('smoke requires --archive <runtime-bundle-*.tar.gz>')
  const archive = path.resolve(args.archive)
  const work = path.resolve(args.workdir || path.join(os.tmpdir(), `hb-smoke-${process.pid}`))
  rmrf(work)
  const rootA = path.join(work, 'a')
  fs.mkdirSync(rootA, { recursive: true })

  // .sha256 sidecar gate first — the same check the installer will run.
  const shaFile = `${archive}.sha256`
  if (fs.existsSync(shaFile)) {
    const expected = fs.readFileSync(shaFile, 'utf8').trim().split(/\s+/)[0]
    const actual = sha256File(archive)
    if (expected !== actual) die(`archive sha mismatch: ${actual} != ${expected}`)
    log('archive sha256 OK')
  } else {
    warn('no .sha256 sidecar next to archive; skipping archive hash gate')
  }

  log(`extracting to ${rootA} ...`)
  run(tarBin(), ['-xzf', archive, '-C', rootA])

  const manifest = loadManifest(rootA)
  const nodeRel = manifest.os === 'win' ? ['.runtime', 'node', 'node.exe'] : ['.runtime', 'node', 'bin', 'node']
  const bundledNode = (root) => path.join(root, ...nodeRel)
  const bundledTool = (root) => path.join(root, 'scripts', 'build-runtime-bundle.mjs')

  // Hermetic probe environment: `hermes --version` walks the full CLI import
  // chain on mac/win (the ultrafast pre-import path is Termux-only) and may
  // touch $HOME/.hermes state — never let probes poke the build host's real
  // hermes home (CI determinism + local-seat safety).
  const probeHome = path.join(work, 'home')
  fs.mkdirSync(probeHome, { recursive: true })
  const probeEnv = { ...process.env, HERMES_HOME: path.join(probeHome, '.hermes') }
  if (process.platform === 'win32') { probeEnv.USERPROFILE = probeHome } else { probeEnv.HOME = probeHome }
  delete probeEnv.VIRTUAL_ENV
  delete probeEnv.PYTHONPATH
  delete probeEnv.PYTHONHOME

  const runProbes = (root, label) => {
    log(`── probes @ ${label} (${root})`)
    // fixup + verify with the BUNDLED node running the BUNDLED tool copy.
    run(bundledNode(root), [bundledTool(root), 'fixup', '--root', root], { env: probeEnv })
    run(bundledNode(root), [bundledTool(root), 'verify', '--root', root], { env: probeEnv })

    const venvBin = manifest.os === 'win' ? path.join(root, 'venv', 'Scripts') : path.join(root, 'venv', 'bin')
    const py = path.join(venvBin, manifest.os === 'win' ? 'python.exe' : 'python')

    // interpreter identity + venv/base wiring
    const idn = run(py, ['-c', [
      'import sys',
      `assert sys.version_info[:2] == (${PYTHON_SERIES.split('.').join(',')}), sys.version`,
      'assert sys.prefix != sys.base_prefix, "not a venv"',
      'assert ".runtime" in sys.base_prefix, sys.base_prefix',
      'print("python-ok", sys.version.split()[0], sys.base_prefix)',
    ].join('\n')], { capture: true, env: probeEnv })
    log(idn.stdout.trim())

    // editable finder covers packages AND py-modules
    run(py, ['-c', 'import hermes_cli, run_agent, toolsets; print("imports-ok")'], { capture: true, env: probeEnv, cwd: os.tmpdir() })
    log('imports-ok (hermes_cli, run_agent, toolsets)')

    // entry-point trampoline end to end (the real relocation assertion)
    const hermes = path.join(venvBin, manifest.os === 'win' ? 'hermes.exe' : 'hermes')
    const v = run(hermes, ['--version'], { capture: true, env: probeEnv, cwd: os.tmpdir() })
    log(`hermes --version: ${(v.stdout || v.stderr).trim().split('\n')[0]}`)

    // bundled tool binaries
    const nv = run(bundledNode(root), ['--version'], { capture: true, env: probeEnv }).stdout.trim()
    if (!nv.startsWith(NODE_SERIES)) die(`bundled node is ${nv}, expected ${NODE_SERIES}x`)
    log(`node-ok ${nv}`)
    const exe = manifest.os === 'win' ? '.exe' : ''
    run(path.join(root, '.runtime', 'bin', `rg${exe}`), ['--version'], { capture: true, env: probeEnv })
    run(path.join(root, '.runtime', 'bin', `uv${exe}`), ['--version'], { capture: true, env: probeEnv })
    log('rg-ok uv-ok')
    if (manifest.os === 'win') {
      const gv = run(path.join(root, '.runtime', 'git', 'cmd', 'git.exe'), ['--version'], { capture: true, env: probeEnv }).stdout.trim()
      log(`git-ok ${gv}`)
    }
    // agent-browser global shim landed in the npm prefix
    const shim = manifest.os === 'win'
      ? path.join(root, '.runtime', 'node', 'node_modules', 'agent-browser')
      : path.join(root, '.runtime', 'node', 'lib', 'node_modules', 'agent-browser')
    if (!fs.existsSync(shim)) die(`agent-browser not found under the bundled npm prefix (${shim})`)
    log('agent-browser-ok (npm -g prefix)')
  }

  runProbes(rootA, 'extract location')

  // The core hc-472 claim: the SAME tree keeps working after a move.
  const rootB = path.join(work, 'b')
  fs.renameSync(rootA, rootB)
  runProbes(rootB, 'moved location')

  if (!args.keep) rmrf(work)
  log('SMOKE OK')
}

// ---------------------------------------------------------------------------

const args = parseArgs(process.argv.slice(2))
const sub = args._[0]
switch (sub) {
  case 'build': await cmdBuild(args); break
  case 'fixup': cmdFixup(args); break
  case 'verify': cmdVerify(args); break
  case 'smoke': cmdSmoke(args); break
  default:
    die('usage: build-runtime-bundle.mjs <build|fixup|verify|smoke> [--out DIR] [--ref REF] [--root DIR] [--archive FILE] [--workdir DIR] [--min-desktop-version X] [--uv-version X] [--keep-stage] [--keep]')
}

/**
 * Write the desktop artifact stamp for source, bundled and Light builds.
 * bundle-electron-main.mjs bakes it into the running code. The packaged
 * sidecar exists only to detect replacement of that artifact by an update.
 * PM's bundle builder supplies relative launch paths for bundled artifacts.
 * Provenance comes from CI, local git, or an explicit unknown-source stamp.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "fs"
import { resolve, join, relative, posix } from "path"
import productIdentity from "../product-identity.cjs"
import { execFileSync } from "child_process"

import { isMain } from "./utils.mjs"

const STAMP_SCHEMA_VERSION = 1

/** All-zero placeholder used when no real commit can be resolved. */
export const FALLBACK_COMMIT = "0000000000000000000000000000000000000000"
export const FALLBACK_BRANCH = "main"

const DESKTOP_ROOT = resolve(import.meta.dirname, "..")
const REPO_ROOT = resolve(DESKTOP_ROOT, "..", "..")
const OUT_DIR = join(DESKTOP_ROOT, "build")
const OUT_FILE = join(OUT_DIR, "install-stamp.json")

function tryExec(argv, opts) {
  try {
    return execFileSync(argv[0], argv.slice(1), { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 5000, ...opts }).trim()
  } catch {
    return null
  }
}

export function fromCI(env = process.env) {
  const sha = env.GITHUB_SHA
  if (!sha) return null
  const branch = env.GITHUB_REF_NAME || env.GITHUB_HEAD_REF || null
  return {
    commit: sha,
    branch: branch,
    dirty: false, // CI builds from a checkout-of-ref by definition
    source: "ci"
  }
}

export function fromLocalGit(repoRoot = REPO_ROOT, execFn = tryExec) {
  const sha = execFn(["git", "rev-parse", "HEAD"], { cwd: repoRoot })
  if (!sha) return null
  const branch = execFn(["git", "rev-parse", "--abbrev-ref", "HEAD"], { cwd: repoRoot })
  // `git status --porcelain -uno` is empty iff tracked files match HEAD.
  // We exclude untracked files (-uno) intentionally: a developer who's
  // checked out an installer scratch dir alongside the repo shouldn't
  // poison every local build with a [DIRTY] stamp.  We DO care about
  // tracked-but-modified files because those mean the .exe content
  // differs from the commit being pinned.
  const status = execFn(["git", "status", "--porcelain", "-uno"], { cwd: repoRoot })
  const dirty = status !== null && status.length > 0
  return {
    commit: sha,
    branch: branch === "HEAD" ? null : branch, // detached HEAD -> null
    dirty: dirty,
    source: "local"
  }
}

export function fromFallback(branch = FALLBACK_BRANCH) {
  // Non-git builds (ZIP download, bootstrap installer without a resolvable
  // HEAD) cannot determine a real commit.  Use a placeholder so local /
  // personal builds can still complete.  The desktop bootstrap treats the
  // all-zero commit as "unknown" and falls back to an unpinned branch
  // bootstrap instead of trying to fetch a non-existent GitHub commit.
  return {
    commit: FALLBACK_COMMIT,
    branch: branch || FALLBACK_BRANCH,
    dirty: false,
    source: "fallback"
  }
}

/**
 * Resolve the install stamp without writing it.  Pure enough for unit tests:
 * inject env / execFn / repoRoot to simulate CI, local git, or no-git trees.
 */
export function resolveStamp({
  env = process.env,
  repoRoot = REPO_ROOT,
  execFn = tryExec,
  fallbackBranch = FALLBACK_BRANCH
} = {}) {
  if (env.HERMES_BUILD_COMMIT) {
    if (!/^[a-f0-9]{40}$/.test(env.HERMES_BUILD_COMMIT) || env.HERMES_PAYLOAD_TAG) {
      throw new Error('Commit builds require an exact full SHA without a release tag')
    }
    const local = fromLocalGit(repoRoot, execFn)
    if (!local || local.commit !== env.HERMES_BUILD_COMMIT) {
      throw new Error('Commit build identity does not match the checkout HEAD')
    }
    return { ...local, branch: null, source: 'commit-build' }
  }
  return fromCI(env) || fromLocalGit(repoRoot, execFn) || fromFallback(fallbackBranch)
}

export function isFallbackCommit(commit) {
  return typeof commit === "string" && /^0{7,40}$/.test(commit)
}

/** Qualify desktop CLI files before the immutable runtime paths are baked.
 * Canonical command keys remain stable for backend consumers; public aliases
 * come from the declared filenames, never those internal keys.
 */
export function stageDesktopLaunchers(root, identity = productIdentity) {
  const file = join(root, 'manifest.json')
  const manifest = JSON.parse(readFileSync(file, 'utf8'))
  const windows = manifest.target.startsWith('win32')
  const commands = {}
  const launchers = []
  for (const [name, source] of Object.entries(manifest.runtime.commands)) {
    const alias = name.replace(/^hermes(?=-|$)/, identity.cliName)
    const destination = posix.join(posix.dirname(source), `${alias}${windows ? '.exe' : ''}`)
    if (source !== destination && existsSync(join(root, source))) {
      renameSync(join(root, source), join(root, destination))
    }
    if (!existsSync(join(root, destination))) throw new Error(`Missing desktop launcher: ${destination}`)
    commands[name] = destination
    launchers.push(alias)
  }
  manifest.runtime.commands = commands
  manifest.launchers = launchers
  writeFileSync(file, JSON.stringify(manifest, null, 2) + '\n')
  return manifest
}

/** Electron and the embedded CLI must see the same immutable provenance. */
export function writeDesktopStamp(outDir, built) {
  const json = JSON.stringify(built, null, 2) + "\n"
  mkdirSync(outDir, { recursive: true })
  if (built.payload === 'bundled') {
    writeFileSync(join(outDir, 'agent-payload', built.runtime.repoDir, 'install-stamp.json'), json, 'utf8')
  }
  writeFileSync(join(outDir, 'install-stamp.json'), json, 'utf8')
}

function main() {
  const stamp = resolveStamp()
  if (!stamp || !stamp.commit) {
    // Should not happen — fromFallback() always provides a commit.
    console.error(
      "[write-build-stamp] ERROR: could not determine git commit.\n" +
        "  - $GITHUB_SHA not set\n" +
        "  - `git rev-parse HEAD` failed at " +
        REPO_ROOT +
        "\n" +
        "Packaged builds require a git ref to pin first-launch install.ps1\n" +
        "against. Run from a git checkout or set $GITHUB_SHA explicitly."
    )
    process.exit(1)
  }

  if (isFallbackCommit(stamp.commit)) {
    console.warn(
      "[write-build-stamp] WARNING: no git commit found (non-git checkout?).\n" +
        "  Using placeholder commit — the packaged app will fall back to the\n" +
        "  default branch for first-launch bootstrap.  For production builds,\n" +
        "  run from a git checkout or set $GITHUB_SHA."
    )
  }

  if (stamp.dirty) {
    console.warn(
      "[write-build-stamp] WARNING: working tree is dirty.\n" +
        "  Pinning to " +
        stamp.commit.slice(0, 12) +
        " but the packaged code may differ from that commit.\n" +
        "  Commit your changes before publishing this build."
    )
  }

  const bundled = ['bundled', 'store'].includes(process.env.HERMES_DESKTOP_VARIANT)
  const payload = bundled
    ? stageDesktopLaunchers(join(OUT_DIR, 'agent-payload'))
    : null
  const built = buildStampPayload(stamp, process.env, process.platform, payload)
  writeDesktopStamp(OUT_DIR, built)
  console.log(
    "[write-build-stamp] wrote " +
      relative(REPO_ROOT, OUT_FILE) +
      " -> " +
      stamp.commit.slice(0, 12) +
      (stamp.branch ? " (" + stamp.branch + ")" : "") +
      (stamp.dirty ? " [DIRTY]" : "") +
      (stamp.source === "fallback" ? " [FALLBACK]" : "")
  )
}

/** One artifact schema for source, bundled and Light builds.
 * The PM bundle builder supplies launch paths only for bundled artifacts.
 */
export function buildStampPayload(stamp, env = process.env, platform = process.platform, payload = null) {
  const variant = (env.HERMES_DESKTOP_VARIANT || "").trim()
  const commitBuild = env.HERMES_BUILD_COMMIT || null
  if (commitBuild && (!/^[a-f0-9]{40}$/.test(commitBuild) || commitBuild !== stamp.commit)) {
    throw new Error('Commit build identity does not match the stamp commit')
  }
  if (commitBuild && env.HERMES_PAYLOAD_TAG) {
    throw new Error('Commit builds cannot also set a release tag')
  }
  const version = env.HERMES_PAYLOAD_VERSION || (env.HERMES_PAYLOAD_TAG || '').replace(/^v/, '') || null
  const base = {
    schemaVersion: STAMP_SCHEMA_VERSION,
    commit: stamp.commit,
    branch: commitBuild ? null : stamp.branch,
    builtAt: new Date().toISOString(),
    dirty: stamp.dirty,
    source: commitBuild ? 'commit-build' : stamp.source,
    commitDate: stamp.commitDate ?? null,
    baseVersion: stamp.baseVersion ?? version?.split('-')[0] ?? null,
    displayVersion: stamp.displayVersion ?? version,
    distance: stamp.distance ?? null
  }

  const updateMechanism = {
    '': 'self',
    bootstrap: 'self',
    store: 'microsoft-store',
    bundled: { win32: 'app-installer', darwin: 'electron-updater' }[platform] || 'external',
    light: platform === 'darwin' ? 'electron-updater' : 'external'
  }[variant]
  if (!updateMechanism) throw new Error(`Unknown desktop variant: ${variant}`)
  const bundled = variant === 'bundled' || variant === 'store'
  if (bundled && !payload?.runtime?.commands?.hermes) {
    throw new Error('PM payload has no completed launch contract; stage the bundle before packaging')
  }
  return {
    ...base,
    payload: variant === "store" ? "bundled" : variant || "bootstrap",
    distribution: "desktop-app",
    updateMechanism: commitBuild ? 'external' : updateMechanism,
    tag: env.HERMES_PAYLOAD_TAG || null,
    ...(bundled ? { runtime: payload.runtime } : {})
  }
}

if (isMain(import.meta.url)) {
  main()
}


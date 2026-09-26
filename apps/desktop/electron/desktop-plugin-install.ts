/**
 * Probe and install desktop runtime plugins from Git repositories.
 * Pure helpers are exported for unit tests; IPC handlers in main.ts call the
 * async entry points with a resolved git binary.
 */

import { spawn } from 'node:child_process'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { publishDesktopTree, writeDesktopHalfMarker } from './desktop-plugins-root'
import { execGit, hiddenGitSpawnSpec } from './no-console-git'

const GITHUB_BROWSER_SEGMENTS = new Set(['tree', 'blob', 'commit'])

export interface ResolvedGitUrl {
  gitUrl: string
  subdir: string | null
}

export interface PluginComponentDetection {
  agent: boolean
  desktop: boolean
  agentName: string | null
  desktopName: string | null
  desktopSourceSubdir: string | null
}

export interface PluginProbeResult {
  ok: boolean
  agent: boolean
  desktop: boolean
  agentName?: string | null
  desktopName?: string | null
  warnings: string[]
  insecure: boolean
  error?: string
}

export interface DesktopPluginInstallResult {
  ok: boolean
  pluginName?: string
  path?: string
  error?: string
}

export function resolvePluginGitUrl(identifier: string): ResolvedGitUrl {
  const trimmed = identifier.trim()

  if (!trimmed) {
    throw new Error('Plugin identifier is required.')
  }

  if (/^(https?:\/\/|git@|ssh:\/\/|file:\/\/)/.test(trimmed)) {
    if (trimmed.startsWith('https://github.com/')) {
      const rest = trimmed.slice('https://github.com/'.length).split(/[?#]/)[0].replace(/\/+$/, '')
      const parts = rest.split('/').filter(Boolean)

      if (parts.length >= 3 && parts[2] && GITHUB_BROWSER_SEGMENTS.has(parts[2])) {
        const repo = parts[1].replace(/\.git$/, '')
        let subdir: string | null = null

        if (parts[2] === 'tree' && parts.length >= 5) {
          subdir = parts.slice(4).join('/').replace(/\/+$/, '') || null
        }

        return { gitUrl: `https://github.com/${parts[0]}/${repo}.git`, subdir }
      }
    }

    if (trimmed.includes('#')) {
      const hashIdx = trimmed.indexOf('#')
      const gitUrl = trimmed.slice(0, hashIdx)
      const subdir = trimmed.slice(hashIdx + 1).replace(/^\/+|\/+$/g, '') || null

      return { gitUrl, subdir }
    }

    const marker = '.git/'

    if (trimmed.includes(marker)) {
      const idx = trimmed.indexOf(marker)
      const gitUrl = trimmed.slice(0, idx + marker.length - 1)
      const subdir = trimmed.slice(idx + marker.length).replace(/^\/+|\/+$/g, '') || null

      return { gitUrl, subdir }
    }

    return { gitUrl: trimmed, subdir: null }
  }

  const parts = trimmed.split('/').filter(Boolean)

  if (parts.length >= 2) {
    const [owner, repo, ...rest] = parts
    const gitUrl = `https://github.com/${owner}/${repo}.git`
    const subdir = rest.join('/').replace(/\/+$/, '') || null

    return { gitUrl, subdir }
  }

  throw new Error("Invalid plugin identifier. Use a Git URL or 'owner/repo' (optionally with a subdirectory).")
}

export function repoNameFromUrl(url: string): string {
  let name = url.replace(/\/+$/, '')

  if (name.endsWith('.git')) {
    name = name.slice(0, -4)
  }

  name = name.split('/').pop() || name

  if (name.includes(':')) {
    name = name.split(':').pop() || name
    name = name.split('/').pop() || name
  }

  return name
}

/** Stable on-disk folder for a desktop plugin. Never the clone temp dir or a generic `desktop/` folder. */
export function desktopPluginFolderName(gitUrl: string, subdir: string | null): string {
  if (subdir) {
    const last = subdir
      .split(/[/\\]/)
      .filter(part => part && part !== '.' && part !== 'desktop')
      .pop()

    if (last) {
      return last
    }
  }

  return repoNameFromUrl(gitUrl)
}

export function resolveSubdirWithin(cloneRoot: string, subdir: string): string {
  const root = path.resolve(cloneRoot)
  const candidate = path.resolve(root, subdir)

  if (candidate !== root && !candidate.startsWith(root + path.sep)) {
    throw new Error(`Plugin subdirectory '${subdir}' escapes the repository.`)
  }

  return candidate
}

function pathExistsSync(filePath: string): boolean {
  try {
    fs.accessSync(filePath)

    return true
  } catch {
    return false
  }
}

async function pathIsDirectory(filePath: string): Promise<boolean> {
  try {
    const stat = await fsp.stat(filePath)

    return stat.isDirectory()
  } catch {
    return false
  }
}

async function pathIsFile(filePath: string): Promise<boolean> {
  try {
    const stat = await fsp.stat(filePath)

    return stat.isFile()
  } catch {
    return false
  }
}

export function findDesktopEntry(pluginRoot: string): { entryFile: string; sourceSubdir: string } | null {
  const rootPlugin = path.join(pluginRoot, 'plugin.js')

  if (pathExistsSync(rootPlugin)) {
    return { entryFile: rootPlugin, sourceSubdir: '.' }
  }

  const nestedPlugin = path.join(pluginRoot, 'desktop', 'plugin.js')

  if (pathExistsSync(nestedPlugin)) {
    return { entryFile: nestedPlugin, sourceSubdir: 'desktop' }
  }

  return null
}

export async function detectPluginComponents(pluginRoot: string): Promise<PluginComponentDetection> {
  const hasYaml =
    pathExistsSync(path.join(pluginRoot, 'plugin.yaml')) || pathExistsSync(path.join(pluginRoot, 'plugin.yml'))

  const hasInit = pathExistsSync(path.join(pluginRoot, '__init__.py'))
  const hasPortable = pathExistsSync(path.join(pluginRoot, 'plugin.json'))
  const agent = (hasYaml && hasInit) || hasPortable

  const desktopEntry = findDesktopEntry(pluginRoot)
  const desktop = desktopEntry !== null

  let agentName: string | null = null

  if (agent) {
    agentName = path.basename(pluginRoot)

    if (hasYaml) {
      try {
        const yamlPath = pathExistsSync(path.join(pluginRoot, 'plugin.yaml'))
          ? path.join(pluginRoot, 'plugin.yaml')
          : path.join(pluginRoot, 'plugin.yml')

        const text = await fsp.readFile(yamlPath, 'utf8')
        const match = text.match(/^name:\s*['"]?([^'"\n]+)['"]?\s*$/m)

        if (match?.[1]) {
          agentName = match[1].trim()
        }
      } catch {
        // Fall back to directory name.
      }
    } else if (hasPortable) {
      try {
        const raw = await fsp.readFile(path.join(pluginRoot, 'plugin.json'), 'utf8')
        const parsed = JSON.parse(raw) as { name?: string }

        if (parsed.name) {
          agentName = parsed.name
        }
      } catch {
        // Fall back to directory name.
      }
    }
  }

  const desktopName = desktop
    ? desktopEntry!.sourceSubdir === '.'
      ? path.basename(pluginRoot)
      : path.basename(path.dirname(desktopEntry!.entryFile))
    : null

  return {
    agent,
    desktop,
    agentName,
    desktopName,
    desktopSourceSubdir: desktopEntry?.sourceSubdir ?? null
  }
}

function noninteractiveGitEnv(): NodeJS.ProcessEnv {
  return {
    ...process.env,
    GIT_TERMINAL_PROMPT: '0',
    GIT_ASKPASS: 'echo',
    SSH_ASKPASS: 'echo'
  }
}

// Matches the backend's default `plugins.clone_timeout_seconds`.
const GIT_TIMEOUT_MS = 300_000

function runGit(gitBin: string, args: string[], cwd?: string): Promise<{ code: number; stderr: string }> {
  return new Promise((resolve, reject) => {
    const spec = hiddenGitSpawnSpec(gitBin, args, {
      cwd,
      env: noninteractiveGitEnv(),
      stdio: ['ignore', 'ignore', 'pipe']
    })

    const child = spawn(spec.command, spec.args, spec.options)

    let stderr = ''

    const timer = setTimeout(() => {
      child.kill('SIGKILL')
      reject(new Error(`Git ${args[0]} timed out after ${GIT_TIMEOUT_MS / 1000} seconds.`))
    }, GIT_TIMEOUT_MS)

    child.stderr?.on('data', chunk => {
      stderr += String(chunk)
    })

    child.on('error', err => {
      clearTimeout(timer)
      reject(err)
    })

    child.on('close', code => {
      clearTimeout(timer)
      resolve({ code: code ?? 1, stderr })
    })
  })
}

async function runGitOrThrow(gitBin: string, args: string[], cwd?: string): Promise<void> {
  const { code, stderr } = await runGit(gitBin, args, cwd)

  if (code !== 0) {
    throw new Error(`Git ${args[0]} failed:\n${stderr.trim()}`)
  }
}

/** Sparse-check-out only `subdir` via the classic pattern file, which older Git clients understand. */
function sparseCheckoutPattern(subdir: string): string {
  return `/${subdir.replace(/^\/+|\/+$/g, '').replace(/([\\*?[])/g, '\\$1')}/\n`
}

// A subdirectory install is a blobless clone with a sparse checkout of that folder: a plugin inside
// a monorepo (Hindsight: 170 MB at depth 1, 2 MB for its plugin folder) otherwise downloads every
// file in the repository and times out on slow connections.
async function cloneToTemp(gitBin: string, gitUrl: string, subdir: string | null): Promise<string> {
  const tmpRoot = await fsp.mkdtemp(path.join(os.tmpdir(), 'hermes-plugin-'))

  try {
    if (!subdir) {
      await runGitOrThrow(gitBin, ['clone', '--depth', '1', gitUrl, tmpRoot])

      return tmpRoot
    }

    await runGitOrThrow(gitBin, ['clone', '--depth', '1', '--filter=blob:none', '--no-checkout', gitUrl, tmpRoot])
    await runGitOrThrow(gitBin, ['config', 'core.sparseCheckout', 'true'], tmpRoot)
    await fsp.mkdir(path.join(tmpRoot, '.git', 'info'), { recursive: true })
    await fsp.writeFile(path.join(tmpRoot, '.git', 'info', 'sparse-checkout'), sparseCheckoutPattern(subdir), 'utf8')
    await runGitOrThrow(gitBin, ['checkout', 'HEAD'], tmpRoot)

    return tmpRoot
  } catch (err) {
    await fsp.rm(tmpRoot, { recursive: true, force: true }).catch(() => undefined)
    throw err
  }
}

async function resolvePluginRoot(cloneRoot: string, subdir: string | null): Promise<string> {
  if (!subdir) {
    return cloneRoot
  }

  const resolved = resolveSubdirWithin(cloneRoot, subdir)

  if (!(await pathIsDirectory(resolved))) {
    throw new Error(`Plugin subdirectory '${subdir}' does not exist in the repository.`)
  }

  return resolved
}

function insecureSchemeWarnings(gitUrl: string): { warnings: string[]; insecure: boolean } {
  if (gitUrl.startsWith('http://') || gitUrl.startsWith('file://')) {
    return {
      warnings: ['This URL uses an insecure or local scheme. Prefer https:// or git@ for production installs.'],
      insecure: true
    }
  }

  return { warnings: [], insecure: false }
}

export async function probePluginRepo(gitBin: string, identifier: string): Promise<PluginProbeResult> {
  try {
    const { gitUrl, subdir } = resolvePluginGitUrl(identifier)
    const { warnings, insecure } = insecureSchemeWarnings(gitUrl)
    const cloneRoot = await cloneToTemp(gitBin, gitUrl, subdir)

    try {
      const pluginRoot = await resolvePluginRoot(cloneRoot, subdir)
      const detected = await detectPluginComponents(pluginRoot)
      const repoFallback = repoNameFromUrl(gitUrl)

      if (!detected.agent && !detected.desktop) {
        return {
          ok: false,
          agent: false,
          desktop: false,
          warnings,
          insecure,
          error: 'No agent or desktop plugin artifacts found in this repository.'
        }
      }

      return {
        ok: true,
        agent: detected.agent,
        desktop: detected.desktop,
        agentName: detected.agentName ?? (detected.agent ? repoFallback : null),
        desktopName: detected.desktop ? desktopPluginFolderName(gitUrl, subdir) : null,
        warnings,
        insecure
      }
    } finally {
      await fsp.rm(cloneRoot, { recursive: true, force: true }).catch(() => undefined)
    }
  } catch (err) {
    return {
      ok: false,
      agent: false,
      desktop: false,
      warnings: [],
      insecure: false,
      error: err instanceof Error ? err.message : String(err)
    }
  }
}

export async function installDesktopPluginFromGit(
  gitBin: string,
  identifier: string,
  desktopPluginsRoot: string,
  force = false
): Promise<DesktopPluginInstallResult> {
  try {
    const { gitUrl, subdir } = resolvePluginGitUrl(identifier)
    const cloneRoot = await cloneToTemp(gitBin, gitUrl, subdir)

    try {
      const pluginRoot = await resolvePluginRoot(cloneRoot, subdir)
      const detected = await detectPluginComponents(pluginRoot)

      if (!detected.desktop || !detected.desktopSourceSubdir) {
        return { ok: false, error: 'No desktop plugin.js found in this repository.' }
      }

      const sourceDir =
        detected.desktopSourceSubdir === '.' ? pluginRoot : path.join(pluginRoot, detected.desktopSourceSubdir)

      // A repo carrying BOTH halves is one package: land its desktop half under
      // the AGENT package name, so the copy this app makes and the one
      // `reconcileUnifiedDesktopHalves` would make are the same folder (#100412)
      // and the Plugins page pairs them into one row. A desktop-only repo keeps
      // the git-derived folder name and stays a standalone plugin.
      const packageName = detected.agent ? (detected.agentName ?? desktopPluginFolderName(gitUrl, subdir)) : null
      const pluginName = packageName ?? desktopPluginFolderName(gitUrl, subdir)
      const targetDir = path.join(desktopPluginsRoot, pluginName)
      const targetPlugin = path.join(targetDir, 'plugin.js')

      if ((await pathIsDirectory(targetDir)) || (await pathIsFile(targetPlugin))) {
        if (!force) {
          return {
            ok: false,
            error: `Desktop plugin '${pluginName}' already exists. Enable force reinstall to replace it.`
          }
        }

        await fsp.rm(targetDir, { recursive: true, force: true })
      }

      // Staged copy + rename: a failed copy must not leave an empty `targetDir`
      // that turns every retry into "already exists. Enable force reinstall".
      // The half of a unified package is stamped with the package marker as
      // part of that publication — without it the Plugins page cannot tell this
      // copy belongs to the agent row (it sits on "copying…" forever) and the
      // half loads default-enabled instead of opt-in.
      await publishDesktopTree(sourceDir, targetDir, async staged => {
        if (!packageName) {
          return
        }

        await writeDesktopHalfMarker(staged, {
          package: packageName,
          repo: gitUrl,
          // The clone is deleted below, so this source never matches a local
          // package's `desktop/` dir: the first reconcile that finds the agent
          // half re-copies from there and the package folder takes over as the
          // single source of truth.
          source: sourceDir,
          sourceMtimeMs: (await fsp.stat(path.join(sourceDir, 'plugin.js'))).mtimeMs
        })
      })

      if (!(await pathIsFile(targetPlugin))) {
        return { ok: false, error: `Install completed but ${targetPlugin} is missing.` }
      }

      return { ok: true, pluginName, path: targetDir }
    } finally {
      await fsp.rm(cloneRoot, { recursive: true, force: true }).catch(() => undefined)
    }
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) }
  }
}

/** Resolve git binary via execFile which path on unix; caller passes Windows-resolved path. */
export function runGitVersion(gitBin: string): Promise<boolean> {
  return execGit(gitBin, ['--version'], { timeoutMs: 5_000 }).then(
    result => result.code === 0,
    () => false
  )
}

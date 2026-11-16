import { spawnSync } from 'node:child_process'
import { cpSync, existsSync, globSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { load } from 'js-yaml'
import { afterAll, expect, test } from 'vitest'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const { packages } = load(readFileSync(join(repo, 'pnpm-workspace.yaml'), 'utf8'))
const workspaces = packages.flatMap(pattern => globSync(pattern, { cwd: repo }))
  .filter(dir => existsSync(join(repo, dir, 'package.json'))).sort()
// The root too: its preinstall does not stop `npm install --ignore-scripts`.
const folders = ['.', ...workspaces]
const scratch = mkdtempSync(join(tmpdir(), 'npm in a workspace-'))
afterAll(() => rmSync(scratch, { recursive: true, force: true }))
const userconfig = join(scratch, 'empty-npmrc')
writeFileSync(userconfig, '')

// A scratch copy of the workspace's own npm inputs (package.json and .npmrc) with
// its dependencies dropped: npm then resolves nothing, so it runs offline and any
// refusal comes from the folder's npm config, not from a `workspace:` spec.
function folder(workspace) {
  const dir = join(scratch, workspace === '.' ? 'root' : workspace.replaceAll('/', '_'))
  mkdirSync(dir, { recursive: true })
  const manifest = JSON.parse(readFileSync(join(repo, workspace, 'package.json'), 'utf8'))
  for (const field of ['dependencies', 'devDependencies', 'optionalDependencies', 'peerDependencies']) delete manifest[field]
  writeFileSync(join(dir, 'package.json'), JSON.stringify(manifest))
  if (existsSync(join(repo, workspace, '.npmrc'))) cpSync(join(repo, workspace, '.npmrc'), join(dir, '.npmrc'))
  return dir
}

// Only the folder's config counts: an empty user npmrc and no inherited npm_config_*
// (npm_config_force or npm_config_engine_strict would change the answer).
function npm(cwd, args) {
  const env = Object.fromEntries(Object.entries(process.env).filter(([name]) => !/^npm_config_/i.test(name)))
  return spawnSync('npm', args, {
    cwd, encoding: 'utf8', shell: process.platform === 'win32', timeout: 120_000,
    env: { ...env, npm_config_userconfig: userconfig, npm_config_cache: join(scratch, 'npm-cache'),
      npm_config_logs_dir: join(scratch, 'npm-logs'), npm_config_offline: 'true', npm_config_update_notifier: 'false',
      npm_config_audit: 'false', npm_config_fund: 'false' },
  })
}

// The root preinstall never runs for `npm install` inside a workspace folder, and
// --ignore-scripts skips it anyway: npm treats the folder as a project of its own
// and resolves it from the registry, past pnpm-workspace.yaml's overrides and
// release-age gate, into a stray package-lock.json.
test.each(folders)('npm refuses to install or lock inside %s', workspace => {
  const cwd = folder(workspace)
  for (const args of [['install', '--package-lock-only', '--ignore-scripts', '--force'], ['install', '--ignore-scripts']]) {
    const run = npm(cwd, args)
    expect(run.error).toBeUndefined()
    expect(run.stderr).toMatch(/EBADENGINE[\s\S]*run pnpm install --frozen-lockfile at the repository root/)
    expect(run.status).not.toBe(0)
    expect(existsSync(join(cwd, 'package-lock.json'))).toBe(false)
    expect(existsSync(join(cwd, 'node_modules'))).toBe(false)
  }
}, 60_000)

// Tools and MCP servers started with a workspace folder as their cwd go through npx.
test.each(folders)('npm exec (npx) and npm run still work inside %s', workspace => {
  const cwd = folder(workspace)
  for (const args of [['exec', '--', 'node', '--version'], ['run']]) {
    const run = npm(cwd, args)
    expect(run.error).toBeUndefined()
    expect(run.stderr).not.toMatch(/EBADENGINE|EBADDEVENGINES/)
    expect(run.status).toBe(0)
  }
}, 60_000)

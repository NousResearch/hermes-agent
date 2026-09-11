#!/usr/bin/env node
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync, realpathSync } from 'node:fs'
import { createRequire } from 'node:module'
import { delimiter, dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'

// npm.cmd needs a shell. Use npm's JS entrypoint so paths remain argv on Windows.
export function npmCommand({ env = process.env } = {}) {
  const names = process.platform === 'win32' ? ['npm.cmd', 'npm'] : ['npm']
  const candidates = [env.npm_execpath]
  for (const dir of (env.PATH || env.Path || '').split(delimiter)) {
    for (const name of names) {
      const bin = join(dir, name)
      if (!existsSync(bin)) continue
      const prefix = dirname(realpathSync(bin))
      candidates.push(
        join(prefix, 'node_modules/npm/bin/npm-cli.js'),
        join(prefix, '../lib/node_modules/npm/bin/npm-cli.js'),
        join(prefix, '../lib/npm/bin/npm-cli.js'),
        realpathSync(bin),
      )
    }
  }
  const cli = candidates.find(path => path && path.endsWith('.js') && existsSync(path))
  if (!cli) throw new Error('npm CLI is required on PATH')
  return [process.execPath, cli]
}

/** Install the full requested workspace union in one strict, locked operation. */
export function prepareNodeDependencies({ source, workspaces, env = process.env }) {
  source = resolve(source)
  if (!Array.isArray(workspaces) || workspaces.length === 0) {
    throw new Error('Select at least one workspace; implicit all-workspace installation is not allowed')
  }
  const manifest = JSON.parse(readFileSync(join(source, 'package.json'), 'utf8'))
  const lock = JSON.parse(readFileSync(join(source, 'package-lock.json'), 'utf8'))
  const workspacePaths = Object.values(lock.packages || {})
    .filter(entry => entry.link)
    .map(entry => entry.resolved)
  const selected = [...new Set(workspaces.map(workspace => {
    const path = workspacePaths.find(path => path === workspace || lock.packages[path]?.name === workspace)
    if (!path || !existsSync(join(source, path, 'package.json'))) {
      throw new Error(`Unknown or missing locked workspace: ${workspace}`)
    }
    return path
  }))]
  const [node, npm] = npmCommand({ env })
  const npmVersion = execFileSync(node, [npm, '--version'], { cwd: source, env, encoding: 'utf8' }).trim()
  const { satisfies } = createRequire(npm)('semver')
  for (const [name, version] of [['node', process.versions.node], ['npm', npmVersion]]) {
    const range = manifest.engines?.[name]
    if (range && !satisfies(version, range)) throw new Error(`${name} ${version} violates ${range}`)
  }
  execFileSync(node, [npm, 'ci', '--no-audit', '--no-fund', '--engine-strict', '--include=dev',
    '--include=optional', '--include-workspace-root=true',
    ...selected.flatMap(workspace => ['--workspace', workspace]),
  ], { cwd: source, env, stdio: 'inherit' })
  return { source, workspaces: selected }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const { values } = parseArgs({ options: {
    source: { type: 'string' }, workspace: { type: 'string', multiple: true },
  } })
  if (!values.source) throw new Error('--source is required')
  prepareNodeDependencies({ source: values.source, workspaces: values.workspace })
}

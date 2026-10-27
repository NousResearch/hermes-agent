// npm's source-development composition for the browser-hosted renderer.
//
// Dependencies are installed into a private workspace, never the checkout:
// npm ci would prune the native Desktop install (node-pty, Electron), and
// headless hosts have no toolchain for its lifecycle scripts. The workspace
// mirrors exactly the inputs the Webapp receipt covers and keeps its
// node_modules between builds, so npm ci reruns only when the lock changes.
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { parseArgs } from 'node:util'
import { isMain, repoRoot } from '../../../scripts/build/frontend-common.mjs'
import { sourceEntries } from '../../../scripts/build/freshness.mjs'
import { prepareNodeDependencies } from '../../../scripts/build/node-deps.mjs'
import { buildWebapp } from '../../../scripts/build/webapp.mjs'

/** Make the workspace hash exactly like the checkout for the Webapp product. */
export function mirrorWebappInputs(source, workspace) {
  const wanted = new Map(sourceEntries(source, 'webapp').map(({ name, kind }) => [name, kind]))
  // Children before parents. Generated trees (node_modules) are never listed, so never removed.
  for (const { name, kind } of sourceEntries(workspace, 'webapp').reverse()) {
    if (wanted.get(name) !== kind) rmSync(join(workspace, name), { recursive: true, force: true })
  }
  for (const [name, kind] of wanted) {
    const target = join(workspace, name)
    // Loose inputs such as pm/lock.json have no listed parent directory.
    mkdirSync(kind === 'directory' ? target : dirname(target), { recursive: true })
    if (kind === 'file') copyFileSync(join(source, name), target)
  }
  // npm ci validates the whole locked workspace graph, so every workspace
  // manifest must exist even though only Desktop and shared are compiled.
  const lock = JSON.parse(readFileSync(join(source, 'package-lock.json'), 'utf8'))
  for (const entry of Object.values(lock.packages || {})) {
    const manifest = entry.link && join(entry.resolved, 'package.json')
    if (!manifest || !existsSync(join(source, manifest))) continue
    mkdirSync(dirname(join(workspace, manifest)), { recursive: true })
    copyFileSync(join(source, manifest), join(workspace, manifest))
  }
}

export async function buildSourceWebapp({
  source = repoRoot, workspace, install = true, prepare = prepareNodeDependencies,
} = {}) {
  source = resolve(source)
  workspace = resolve(workspace ?? join(source, '.build/webapp-workspace'))
  mkdirSync(workspace, { recursive: true })
  mirrorWebappInputs(source, workspace)
  prepare({
    source: workspace, workspaces: ['apps/desktop'], reuse: true, install,
    env: { ...process.env, npm_config_ignore_scripts: 'true' },
  })
  return buildWebapp({ source, workspace, out: join(source, 'apps/desktop/dist-webapp') })
}

if (isMain(import.meta.url)) {
  try {
    const { values } = parseArgs({ options: { 'no-install': { type: 'boolean', default: false } } })
    const result = await buildSourceWebapp({ install: !values['no-install'] })
    console.log(`built ${result.index}`)
  } catch (error) {
    console.error(error)
    process.exitCode = 1
  }
}

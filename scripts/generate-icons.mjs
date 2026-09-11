#!/usr/bin/env node
/**
 * Generate the icon assets on demand for the pipeline that needs them.
 *
 * Generated icons are NOT committed (see scripts/generate_icons.py) — every
 * consuming pipeline regenerates them before building:
 *   - website:  website/scripts/prebuild.mjs (docusaurus prebuild)
 *   - desktop:  apps/desktop/package.json prebuild + predev
 *   - installer: apps/bootstrap-installer/package.json prebuild
 *   - web:       web/package.json prebuild
 *
 * The locked icon-build group runs outside the application environment.
 * It is not a runtime extra, so --all-extras payloads do not include resvg.
 */
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import { parseArgs } from 'node:util'
import { fileURLToPath, pathToFileURL } from 'node:url'

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

export function generateIcons(args = [], { root = repoRoot, run = spawnSync, env = process.env } = {}) {
  const { values } = parseArgs({ args, options: {
    source: { type: 'string' }, out: { type: 'string' }, check: { type: 'boolean' }
  } })
  const source = path.resolve(values.source ?? root)
  const out = path.resolve(values.out ?? source)
  const childEnv = { ...env }
  // Parent payload paths must not shadow the isolated build dependencies.
  delete childEnv.PYTHONPATH
  delete childEnv.PYTHONHOME
  const result = run(env.HERMES_PYTHON || 'python', [
    path.join(root, 'scripts', 'build', 'icon_environment.py'), '--source', source, '--out', out,
    ...(values.check ? ['--check'] : [])
  ], { cwd: source, stdio: 'inherit', windowsHide: true, env: childEnv })
  if (result.error) {
    console.error('[generate-icons] failed to launch icon generator:', result.error.message)
    console.error('[generate-icons] a prepared Python (HERMES_PYTHON or PATH) is required to run the PM build driver')
    return 1
  }
  return result.status ?? 1
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  process.exitCode = generateIcons(process.argv.slice(2))
}

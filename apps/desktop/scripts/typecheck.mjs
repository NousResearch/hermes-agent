import { spawnSync } from 'node:child_process'
import { dirname, join, resolve } from 'node:path'
import { isMain, repoRoot, workspaceTool } from '../../../scripts/build/frontend-common.mjs'

// The desktop renderer program (`tsc -p . --noEmit`) checks ~8.6k files and
// peaks around 2.6 GiB of V8 heap, exceeding Node's ~2 GiB default ceiling on
// 8 GB machines (#144459). Set on the tsc children here rather than via a
// cross-env prefix (#103010, #110121) or `.npmrc` `node-options` (which
// overwrites inherited NODE_OPTIONS across every workspace, defeating
// scripts/build/web.mjs's bounded cgroup heap cap).
const MIN_HEAP_MB = 4096
const HEAP_FLAG = `--max-old-space-size=${MIN_HEAP_MB}`

export const TYPECHECK_STEPS = Object.freeze([
  Object.freeze(['-p', '.', '--noEmit']),
  Object.freeze(['-p', 'tsconfig.electron.json', '--noEmit']),
  Object.freeze(['-p', 'tsconfig.e2e.json', '--noEmit']),
  Object.freeze([
    '--ignoreConfig',
    '--allowJs',
    '--checkJs',
    '--noEmit',
    '--skipLibCheck',
    '--module',
    'nodenext',
    '--moduleResolution',
    'nodenext',
    '--target',
    'es2022',
    'electron-builder.config.cjs',
  ]),
])

/**
 * Preserve inherited NODE_OPTIONS byte-identically (quoted preload paths
 * survive); append the 4096 MB floor last unless the effective inherited
 * `--max-old-space-size` already meets or exceeds it.
 * @param {string} [inherited]
 * @returns {string}
 */
export function typecheckNodeOptions(inherited = process.env.NODE_OPTIONS ?? '') {
  const matches = [...inherited.matchAll(/(?:^|\s)--max-old-space-size=(\d+)(?=\s|$)/g)]
  const lastMb = matches.length ? Number(matches.at(-1)[1]) : 0
  if (lastMb >= MIN_HEAP_MB) return inherited
  return `${inherited} ${HEAP_FLAG}`.trim()
}

/**
 * @param {{ source?: string, spawn?: typeof spawnSync, env?: NodeJS.ProcessEnv }} [options]
 * @returns {number}
 */
export function runDesktopTypecheck({ source = repoRoot, spawn = spawnSync, env = process.env } = {}) {
  source = resolve(source)
  const app = join(source, 'apps/desktop')
  const tsc = join(dirname(workspaceTool(source, 'apps/desktop', 'typescript')), 'tsc.js')
  const childEnv = { ...env, NODE_OPTIONS: typecheckNodeOptions(env.NODE_OPTIONS ?? '') }
  for (const stepArgs of TYPECHECK_STEPS) {
    const result = spawn(process.execPath, [tsc, ...stepArgs], {
      cwd: app,
      stdio: 'inherit',
      env: childEnv,
    })
    if (result.error) throw result.error
    if (result.status !== 0) return result.status ?? 1
  }
  return 0
}

if (isMain(import.meta.url)) {
  process.exitCode = runDesktopTypecheck()
}

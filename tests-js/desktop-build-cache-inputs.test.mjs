// Editing any repo module that desktop preparation's Node programs load must change the desktop
// input-cache key (scripts/ci/desktop_build_cache.py), or a restore prefers an entry built from
// the old module. The module graph is observed at runtime with a resolve hook; the key comes from
// the Python cache owner's own CLI.
//
// Limits: a dynamic import() inside a function that does not run at import time, and a PowerShell
// dot-source (windows-deps.ps1 -> windows-build-deps.ps1), are invisible here and stay listed in
// _INPUT_FILES by hand. Importing a program must not run it (isMain guards).
import { execFileSync } from 'node:child_process'
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, expect, test } from 'vitest'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const python = process.env.HERMES_PYTHON || 'python3'
const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

// The Node programs scripts/bundles/desktop_prepare.py runs.
const PREPARATION_PROGRAMS = [
  'scripts/build/node-deps.mjs', 'apps/desktop/scripts/stage-native-deps.mjs',
  'apps/desktop/scripts/prepare-packaging-tools.mjs', 'apps/desktop/scripts/probe-prepared-native.mjs',
]
const GRAPH = `
import { registerHooks } from 'node:module'
import { fileURLToPath, pathToFileURL } from 'node:url'
import path from 'node:path'
const loaded = new Set()
registerHooks({ resolve(specifier, context, next) {
  const result = next(specifier, context)
  if (result.url.startsWith('file:')) loaded.add(fileURLToPath(result.url))
  return result
} })
// Programs arrive by env: argv[1] would make the first one run as main.
for (const program of JSON.parse(process.env.PROGRAMS)) await import(pathToFileURL(path.resolve(program)).href)
console.log(JSON.stringify([...loaded]))
`

function loadedModules() {
  const output = execFileSync(process.execPath, ['--input-type=module', '-e', GRAPH], {
    cwd: repo, encoding: 'utf8', env: { ...process.env, PROGRAMS: JSON.stringify(PREPARATION_PROGRAMS) },
  })
  const modules = JSON.parse(output).map(file => relative(repo, file).split(sep).join('/'))
  return [...new Set(modules)].filter(module => !module.startsWith('..') && !module.split('/').includes('node_modules')).sort()
}

// A minimal source tree describe_cache accepts: one pnpm workspace plus the lock inputs.
function sourceTree(modules) {
  const base = mkdtempSync(join(tmpdir(), 'desktop cache inputs-'))
  roots.push(base)
  const source = join(base, 'source')
  const write = (name, text) => {
    mkdirSync(dirname(join(source, name)), { recursive: true })
    writeFileSync(join(source, name), text)
  }
  write('package.json', JSON.stringify({ name: 'root' }))
  write('pnpm-workspace.yaml', 'packages: []\n')
  write('pnpm-lock.yaml', "lockfileVersion: '9.0'\n\nimporters:\n\n  .: {}\n")
  for (const module of modules) {
    mkdirSync(dirname(join(source, module)), { recursive: true })
    copyFileSync(join(repo, module), join(source, module))
  }
  return { base, source }
}

function inputKey({ base, source }) {
  const output = execFileSync(python, [join(repo, 'scripts/ci/desktop_build_cache.py'), 'describe',
    '--source', source, '--cache', join(base, 'cache'), '--producer', 'desktop-production'], {
    cwd: base, encoding: 'utf8', env: { ...process.env, GITHUB_OUTPUT: '', HERMES_HOME: join(base, 'home') },
  })
  return JSON.parse(output).restore_keys[0]
}

test('editing a module the preparation Node programs load changes the desktop input key', () => {
  const modules = loadedModules()
  expect(modules).toEqual(expect.arrayContaining(PREPARATION_PROGRAMS))
  const tree = sourceTree(modules)
  const first = inputKey(tree)
  const stale = modules.filter(module => {
    const path = join(tree.source, module)
    const old = readFileSync(path)
    writeFileSync(path, Buffer.concat([old, Buffer.from('\n')]))
    const changed = inputKey(tree) !== first
    writeFileSync(path, old)
    return !changed
  })
  expect(stale, 'desktop cache key ignores imported modules').toEqual([])
}, 120_000)

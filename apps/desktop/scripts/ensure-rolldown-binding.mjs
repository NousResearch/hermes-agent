import { existsSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')

// Rolldown's loader already resolves platform, arch and libc; when the native
// package is absent, its load-error chain names the exact `@rolldown/binding-*`
// it wanted, followed by the wasm fallback it also tried. Reuse that verdict
// instead of re-deriving it from `process.platform` (Windows bindings carry a
// `-msvc` suffix, Linux ones `-gnu`/`-musl`).
export function missingRolldownBinding(stderr) {
  const names = [...String(stderr ?? '').matchAll(/@rolldown\/binding-[a-z0-9-]+/g)].map(m => m[0])
  return names.find(name => !name.includes('wasm32')) ?? null
}

function probeRolldown(root) {
  return spawnSync(process.execPath, ['--input-type=module', '--eval', "await import('rolldown')"], {
    cwd: root,
    encoding: 'utf8'
  })
}

function installBinding(root, spec) {
  // npm.cmd needs a shell on Windows (Node refuses to spawn .cmd directly).
  return spawnSync('npm', ['install', '--no-save', '--package-lock=false', '--include=optional', spec], {
    cwd: root,
    stdio: 'inherit',
    shell: process.platform === 'win32'
  })
}

function installedRolldownVersion(root) {
  const packagePath = [
    join(root, 'node_modules', 'rolldown', 'package.json'),
    join(root, 'apps', 'desktop', 'node_modules', 'rolldown', 'package.json')
  ].find(existsSync)
  return packagePath ? JSON.parse(readFileSync(packagePath, 'utf8')).version : null
}

export function ensureRolldownBinding({
  root = rootDir,
  probe = probeRolldown,
  install = installBinding,
  rolldownVersion = installedRolldownVersion
} = {}) {
  const initial = probe(root)
  if (initial.status === 0) return true

  const name = missingRolldownBinding(initial.stderr)
  const version = rolldownVersion(root)
  if (!name || !version) {
    console.error(initial.stderr?.trim() || 'Rolldown could not load.')
    return false
  }

  console.warn(`Rolldown could not load; installing ${name}@${version}...`)
  if (install(root, `${name}@${version}`).status !== 0) return false

  const repaired = probe(root)
  if (repaired.status !== 0) {
    console.error(repaired.stderr?.trim() || `Rolldown still cannot load after installing ${name}.`)
    return false
  }
  return true
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = ensureRolldownBinding() ? 0 : 1
}

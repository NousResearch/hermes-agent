import { existsSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const appDir = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const rootDir = resolve(appDir, '..', '..')

export function selectRolldownBinding(optionalDependencies, platform, arch) {
  const suffix = `binding-${platform}-${arch}`
  const matches = Object.entries(optionalDependencies ?? {}).filter(([name]) => name.endsWith(suffix))

  return matches.length === 1 ? matches[0] : null
}

function probeRolldown(root) {
  return spawnSync(process.execPath, ['--input-type=module', '--eval', "await import('rolldown')"], {
    cwd: root,
    encoding: 'utf8'
  })
}

function installBinding(root, spec) {
  const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm'
  return spawnSync(npm, ['install', '--no-save', '--package-lock=false', '--include=optional', spec], {
    cwd: root,
    stdio: 'inherit'
  })
}

export function ensureRolldownBinding({
  root = rootDir,
  platform = process.platform,
  arch = process.arch,
  probe = probeRolldown,
  install = installBinding,
  findPackage = rootPath =>
    [
      join(rootPath, 'node_modules', 'rolldown', 'package.json'),
      join(rootPath, 'apps', 'desktop', 'node_modules', 'rolldown', 'package.json')
    ].find(existsSync),
  readPackage = path => JSON.parse(readFileSync(path, 'utf8'))
} = {}) {
  const initial = probe(root)
  if (initial.status === 0) return true

  let rolldownPackage
  try {
    const packagePath = findPackage(root)
    if (!packagePath) throw new Error('package.json was not found')
    rolldownPackage = readPackage(packagePath)
  } catch (error) {
    console.error(`Could not inspect the installed Rolldown package: ${error.message}`)
    return false
  }

  const binding = selectRolldownBinding(rolldownPackage.optionalDependencies, platform, arch)
  if (!binding) {
    console.error(`Rolldown has no unambiguous native binding for ${platform}/${arch}.`)
    return false
  }

  const [name, version] = binding
  console.warn(`Rolldown could not load; repairing ${name}@${version}...`)
  const installed = install(root, `${name}@${version}`)
  if (installed.status !== 0) return false

  const repaired = probe(root)
  if (repaired.status !== 0) {
    const detail = repaired.stderr?.trim() || repaired.stdout?.trim()
    console.error(detail || `Rolldown still cannot load after installing ${name}.`)
    return false
  }

  return true
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = ensureRolldownBinding() ? 0 : 1
}

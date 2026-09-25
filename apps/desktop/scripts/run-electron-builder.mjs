// Source convenience prepares first; --prepared only admits and consumes files.
import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { createRequire } from 'node:module'
import { isMain } from './utils.mjs'
import { readPackagingInputs, preparationRequired } from './prepared-packaging.mjs'
import { readNativeInputs } from './prepared-native-deps.mjs'
import { pinnedPackageRoot } from './prepare-packaging-tools.mjs'

const source = path.resolve(import.meta.dirname, '../../..')
const app = path.join(source, 'apps/desktop')
const platformFlags = new Map([['--win', 'win32'], ['-w', 'win32'], ['--windows', 'win32'],
  ['--mac', 'darwin'], ['--macos', 'darwin'], ['-m', 'darwin'], ['-o', 'darwin'], ['--linux', 'linux'], ['-l', 'linux']])
const architectures = ['--x64', '--arm64', '--ia32', '--armv7l', '--universal']

/** @param {string[]} args @param {string} name @returns {string | undefined} */
function takeOption(args, name) {
  const index = args.findIndex(arg => arg === name || arg.startsWith(`${name}=`))
  if (index < 0) return undefined
  const [option] = args.splice(index, 1)
  const value = option.includes('=') ? option.slice(name.length + 1) : args.splice(index, 1)[0]
  if (!value || value.startsWith('-')) throw new Error(`${name} requires a path`)
  return path.resolve(value)
}

/** @param {string[]} args @param {import('./prepared-packaging.mjs').PreparedPackaging} inputs @returns {void} */
export function validatePreparedBuilderArgs(args, inputs) {
  const requested = sourceFormats(args)
  if (requested.some(format => format !== 'dir' && !inputs.formats.includes(format))) {
    throw preparationRequired(`Package formats were not prepared: ${requested.join(', ')}`)
  }
  for (let index = 0; index < args.length; index++) {
    const arg = args[index]
    if (platformFlags.has(arg)) {
      if (!inputs.target.startsWith(`${platformFlags.get(arg)}-`)) throw preparationRequired(`Conflicting package platform: ${arg}`)
      continue
    }
    if (architectures.includes(arg)) {
      if (!inputs.target.endsWith(`-${arg.slice(2)}`)) throw preparationRequired(`Conflicting package architecture: ${arg}`)
      continue
    }
    if (arg === '--publish' || arg === '-p') {
      if (args[++index] !== 'never') throw new Error('Publishing belongs to the release adapter')
      continue
    }
    if (arg === '--dir' || arg === '--publish=never' || arg === '--') continue
    if (/^(?:-c|--config)\.(?:extraMetadata\.(?:version|shortVersion|shortVersionWindows)|directories\.output|mac\.identity)=/.test(arg)) continue
    if (!arg.startsWith('-') && inputs.formats.includes(arg)) continue
    throw preparationRequired(`Argument is not admitted by prepared packaging: ${arg}`)
  }
}

/**
 * Copy mutable toolsets into the build resources directory: upstream's custom
 * Windows tool contract requires containment there. Never chmod/sign the cache.
 * @param {import('./prepared-packaging.mjs').PreparedPackaging} inputs
 * @returns {string[]}
 */
function toolsetArguments(inputs) {
  const require = createRequire(path.join(source, 'apps/desktop/package.json'))
  const config = require(path.join(app, 'electron-builder.config.cjs'))
  const tools = path.join(app, config.directories?.buildResources || 'build', 'prepared-packaging-tools')
  fs.mkdirSync(tools, { recursive: true })
  return Object.entries(inputs.toolsets).map(([name, directory]) => {
    const destination = path.join(tools, name)
    fs.rmSync(destination, { recursive: true, force: true })
    fs.cpSync(directory, destination, { recursive: true, verbatimSymlinks: true })
    // Upstream parses this as a literal path after slicing file:// (not URL decoding).
    return `-c.toolsets.${name}.url=file://${destination}`
  })
}

/**
 * Prepare one isolated input set per source target; the strict child cannot acquire.
 * @param {string[]} args
 * @param {string | undefined} nativeDeps
 * @param {typeof spawnSync} spawn
 * @returns {number}
 */
function runSourceBuilds(args, nativeDeps, spawn) {
  const platform = selectedPlatform(args)
  const requested = [...new Set(args.filter(arg => architectures.includes(arg)))]
  if (requested.includes('--universal')) throw new Error('No prepared universal native payload; use --x64 --arm64 for separate packages')
  if (nativeDeps && requested.length > 1) throw new Error('--native-deps selects one architecture, not multiple source targets')
  for (const flag of requested.length ? requested : [`--${process.arch}`]) {
    const arch = flag.slice(2)
    const target = `${platform}-${arch}`
    const out = path.join(app, 'build/packager', target)
    const native = nativeDeps || path.join(app, requested.length ? `build/native-deps-${target}` : 'build/native-deps')
    const commands = []
    if (!nativeDeps && (requested.length || platform !== process.platform || !fs.existsSync(`${native}.prepared.json`))) {
      commands.push([path.join(import.meta.dirname, 'stage-native-deps.mjs'), '--source', source,
        '--out', native, '--platform', platform, '--arch', arch])
    }
    commands.push([path.join(import.meta.dirname, 'prepare-packaging-tools.mjs'),
      '--source', source, '--out', out, '--target', target, '--cache', path.join(source, '.cache/desktop-inputs/packager'),
      ...sourceFormats(args).flatMap(format => ['--format', format])])
    commands.push([path.join(import.meta.dirname, 'run-electron-builder.mjs'),
      '--prepared', path.join(out, 'prepared.json'), '--native-deps', native,
      ...args.filter(arg => !architectures.includes(arg)), flag])
    for (const command of commands) {
      const result = spawn(process.execPath, command, { cwd: app, stdio: 'inherit' })
      if (result.error) throw result.error
      if (result.status !== 0) return result.status ?? 1
    }
  }
  return 0
}

/** @param {string[]} args @returns {string} */
function selectedPlatform(args) {
  const platforms = [...new Set(args.filter(arg => platformFlags.has(arg)).map(arg => platformFlags.get(arg)))]
  if (platforms.length > 1) throw preparationRequired('Select one packaging platform per invocation')
  return platforms[0] || process.platform
}

/** @param {string[]} args @param {{ spawn?: typeof spawnSync }} [options] @returns {number} */
export function runElectronBuilder(args, { spawn = spawnSync } = {}) {
  const validateOnly = args.includes('--validate-only')
  args = args.filter(arg => arg !== '--validate-only')
  const manifest = takeOption(args, '--prepared')
  const nativeDeps = takeOption(args, '--native-deps')
  if (validateOnly && !manifest) throw preparationRequired('--validate-only requires --prepared')
  if (!manifest) return runSourceBuilds(args, nativeDeps, spawn)
  const platform = selectedPlatform(args)
  const arch = args.find(arg => architectures.includes(arg))?.slice(2) || process.arch
  const inputs = readPackagingInputs(manifest, source, `${platform}-${arch}`)
  validatePreparedBuilderArgs(args, inputs)
  if (!nativeDeps) throw preparationRequired('--native-deps is required with --prepared')
  readNativeInputs({ source, nativeDeps, platform, arch })
  if (validateOnly) return 0
  const builder = pinnedPackageRoot(source, 'electron-builder')
  pinnedPackageRoot(source, 'app-builder-lib')
  const require = createRequire(path.join(builder, 'package.json'))
  const bin = require(path.join(builder, 'package.json')).bin['electron-builder']
  const preloads = []
  if (process.platform === 'darwin') {
    preloads.push('--import', path.join(import.meta.dirname, 'patch-electron-builder-mac-binary.mjs'))
    preloads.push('--require', path.join(import.meta.dirname, 'dmgbuild-diagnostics.cjs'))
  }
  /** @type {NodeJS.ProcessEnv} */
  const env = { ...process.env, HERMES_PREPARED_PACKAGING: manifest,
    HERMES_PREPARED_NATIVE_DEPS: nativeDeps, HERMES_PREPARED_TARGET: inputs.target }
  if (inputs.dmgbuild) env.CUSTOM_DMGBUILD_PATH = inputs.dmgbuild
  if (inputs.windows?.dotnetRoot) env.DOTNET_ROOT = inputs.windows.dotnetRoot
  const result = spawn(process.execPath, [...preloads, path.join(builder, bin), ...args,
    '--config', 'electron-builder.config.cjs', '--publish', 'never', `-c.electronDist=${inputs.electron}`,
    ...toolsetArguments(inputs)], { cwd: app, stdio: 'inherit', env })
  if (result.error) throw result.error
  return result.status ?? 1
}

/** @param {string[]} args @returns {string[]} */
function sourceFormats(args) {
  if (args.includes('--dir')) return ['dir']
  const formats = args.filter(arg => ['dmg', 'zip', 'msix', 'AppImage', 'deb', 'rpm'].includes(arg))
  const platform = selectedPlatform(args)
  return formats.length ? formats : platform === 'darwin' ? ['dmg', 'zip'] : platform === 'win32' ? ['msix'] : ['AppImage']
}

if (isMain(import.meta.url)) process.exitCode = runElectronBuilder(process.argv.slice(2))

#!/usr/bin/env node
import fs from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'
import { isMain } from './utils.mjs'
import { publishPackagingInputs } from './prepared-packaging.mjs'
import { ensureWindowsBundleTools } from './windows-bundle-tools.mjs'
import { prepareDmgbuild } from './prepare-dmgbuild.mjs'

/** @param {string} source @param {string} name @returns {string} */
export function pinnedPackageRoot(source, name) {
  const require = createRequire(path.join(source, 'apps/desktop/package.json'))
  const entry = require.resolve(name)
  let directory = path.dirname(entry)
  while (!fs.existsSync(path.join(directory, 'package.json'))) {
    const parent = path.dirname(directory)
    if (parent === directory) throw new Error(`Cannot locate installed ${name}`)
    directory = parent
  }
  const installed = JSON.parse(fs.readFileSync(path.join(directory, 'package.json'), 'utf8'))
  const lock = JSON.parse(fs.readFileSync(path.join(source, 'package-lock.json'), 'utf8'))
  const key = path.relative(source, directory).split(path.sep).join('/')
  if (!lock.packages?.[key] || lock.packages[key].version !== installed.version || installed.name !== name) {
    throw new Error(`Installed ${name} is not the source's lock-pinned package; prepare Node dependencies first`)
  }
  return directory
}

/** @param {string} target @returns {'x64' | 'arm64'} */
export function packagingTargetArch(target) {
  if (target === `${process.platform}-x64`) return 'x64'
  if (target === `${process.platform}-arm64`) return 'arm64'
  throw new Error(`Packaging preparation requires a same-OS x64/arm64 target, got ${target}`)
}

/** @param {string} from @param {string} to @returns {string} */
function copyTool(from, to) {
  fs.rmSync(to, { recursive: true, force: true })
  fs.cpSync(from, to, { recursive: true, verbatimSymlinks: true })
  return to
}

/**
 * Acquire bytes without signing credentials. Builder modules are loaded only
 * after the explicit cache root has been selected, before their lazy state runs.
 * @param {{ source: string, out: string, cache: string, target?: string, formats?: string[], dmgbuild?: string }} options
 * @returns {Promise<string>}
 */
async function preparePackagingTools({ source, out, cache, target = `${process.platform}-${process.arch}`, formats, dmgbuild }) {
  source = fs.realpathSync(source)
  out = path.resolve(out)
  cache = path.resolve(cache)
  fs.mkdirSync(out, { recursive: true })
  fs.rmSync(path.join(out, 'prepared.json'), { force: true })
  packagingTargetArch(target)
  const builderRoot = pinnedPackageRoot(source, 'app-builder-lib')
  pinnedPackageRoot(source, 'electron-builder')
  const require = createRequire(path.join(source, 'apps/desktop/package.json'))
  const config = require(path.join(source, 'apps/desktop/electron-builder.config.cjs'))
  formats ??= process.platform === 'win32' ? ['msix'] : process.platform === 'darwin' ? ['dmg', 'zip'] : ['AppImage']
  if (process.env.CUSTOM_DMGBUILD_PATH) throw new Error('Preparation must select the pinned dmgbuild supplier, not CUSTOM_DMGBUILD_PATH')
  const supported = process.platform === 'win32' ? ['dir', 'msix', 'zip'] : process.platform === 'darwin' ? ['dir', 'dmg', 'zip'] : ['dir', 'AppImage', 'deb', 'rpm', 'zip']
  if (formats.some(format => !supported.includes(format))) throw new Error(`Unsupported prepared package formats: ${formats.join(', ')}`)
  const dmg = formats.includes('dmg') ? prepareDmgbuild({ source, out, cache, binary: dmgbuild }) : null
  const previousCache = process.env.ELECTRON_BUILDER_CACHE
  process.env.ELECTRON_BUILDER_CACHE = path.join(cache, 'builder')
  try {
    return await acquirePackagingTools({ source, out, cache, target, formats, builderRoot, config, dmgbuild: dmg })
  } finally {
    if (previousCache === undefined) delete process.env.ELECTRON_BUILDER_CACHE
    else process.env.ELECTRON_BUILDER_CACHE = previousCache
  }
}

/**
 * @param {{ source: string, out: string, cache: string, target: string, formats: string[], builderRoot: string, config: import('app-builder-lib').Configuration, dmgbuild: string | null }} options
 * @returns {Promise<string>}
 */
async function acquirePackagingTools({ source, out, cache, target, formats, builderRoot, config, dmgbuild }) {
  /** @param {string} relative */
  const load = (relative) => import(pathToFileURL(path.join(builderRoot, 'dist', relative)).href)
  const [electronGet, sevenZip, icons] = await Promise.all([
    load('util/electronGet.js'), load('toolsets/7zip.js'), load('toolsets/icons.js'),
  ])
  const resourcesDir = path.join(source, 'apps/desktop', config.directories?.buildResources || 'build')
  const [archive, archiveTool, iconTools] = await Promise.all([
    electronGet.downloadElectronArtifactZip({ version: config.electronVersion, platformName: process.platform, arch: packagingTargetArch(target),
      artifactName: 'electron', cacheDir: path.join(cache, 'electron') }),
    sevenZip.getPath7za(), icons.getIconsToolsetPath(config.toolsets?.icons, resourcesDir),
  ])
  const electron = copyTool(archive, path.join(out, 'electron.zip'))
  /** @type {import('./prepared-packaging.mjs').PackagingToolsets} */
  const toolsets = {
    sevenZip: copyTool(path.dirname(path.dirname(archiveTool)), path.join(out, 'sevenZip')),
    icons: copyTool(iconTools, path.join(out, 'icons')),
  }
  let windows = null
  if (process.platform === 'win32') {
    const builder = await load('toolsets/winCodeSign.js')
    const tools = await ensureWindowsBundleTools({ config, resourcesDir, signing: true, load: async () => builder, prepared: null })
    const kitRoot = copyTool(path.dirname(path.dirname(tools.makeappx)), path.join(out, 'winCodeSign'))
    if (!tools.dlib || !tools.dotnetRoot) throw new Error('Windows preparation requires the ATS dlib and paired .NET runtime')
    fs.cpSync(path.dirname(tools.dlib), path.join(kitRoot, path.basename(path.dirname(tools.signtool))), { recursive: true })
    const rcedit = await builder.getRceditBundle(config.toolsets?.winCodeSign, resourcesDir)
    fs.copyFileSync(rcedit.x64, path.join(kitRoot, 'rcedit-x64.exe'))
    fs.copyFileSync(rcedit.x86, path.join(kitRoot, 'rcedit-x86.exe'))
    const kit = path.join(kitRoot, path.basename(path.dirname(tools.makeappx)))
    windows = { makeappx: path.join(kit, 'makeappx.exe'), signtool: path.join(kit, 'signtool.exe'),
      dlib: path.join(kit, 'Azure.CodeSigning.Dlib.dll'), dotnetRoot: copyTool(tools.dotnetRoot, path.join(out, 'dotnet')) }
    toolsets.winCodeSign = kitRoot
  }
  if (formats.includes('AppImage')) {
    const appimage = await load('toolsets/appimage.js')
    const { Arch } = await import(pathToFileURL(path.join(builderRoot, 'dist/index.js')).href)
    const tools = await appimage.getAppImageTools(config.toolsets?.appimage, Arch[packagingTargetArch(target)], resourcesDir)
    toolsets.appimage = copyTool(path.dirname(tools.mksquashfs), path.join(out, 'appimage'))
  }
  if (formats.some(format => format === 'deb' || format === 'rpm')) {
    const fpm = await load('toolsets/fpm.js')
    toolsets.fpm = copyTool(path.dirname(await fpm.getFpmPath(config.toolsets?.fpm, resourcesDir)), path.join(out, 'fpm'))
  }
  return publishPackagingInputs({ source, out, target, formats, electron, toolsets, windows, dmgbuild })
}

if (isMain(import.meta.url)) {
  const { values } = parseArgs({ options: {
    source: { type: 'string' }, out: { type: 'string' }, cache: { type: 'string' }, target: { type: 'string' },
    format: { type: 'string', multiple: true }, dmgbuild: { type: 'string' },
  } })
  if (!values.source || !values.out || !values.cache) throw new Error('Usage: prepare-packaging-tools.mjs --source REPO --out WORK/packager --cache CACHE/packager [--target same-OS-target] [--format FORMAT] [--dmgbuild PM_BINARY]')
  console.log(await preparePackagingTools({ source: values.source, out: values.out, cache: values.cache, target: values.target, formats: values.format, dmgbuild: values.dmgbuild }))
}

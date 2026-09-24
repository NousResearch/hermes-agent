import { access, mkdtemp, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { Platform } from 'app-builder-lib'
import { PlatformPackager } from 'app-builder-lib/out/platformPackager.js'
import { expect, it, vi } from 'vitest'

import pkg from '../package.json' with { type: 'json' }

async function configuredHook(context) {
  if (pkg.build.afterPack) {
    const hook = await import(new URL(`../${pkg.build.afterPack}`, import.meta.url).href)
    await hook.default(context)
  }
}

async function writeDesktopDist(projectDir) {
  await mkdir(path.join(projectDir, 'dist'), { recursive: true })
  await Promise.all([
    writeFile(path.join(projectDir, 'dist', 'electron-main.mjs'), 'main bundle'),
    writeFile(path.join(projectDir, 'dist', 'index.html'), '<main />')
  ])
}

function context(appOutDir, productFilename = 'Hermes Preview') {
  // Use electron-builder's real bundle path resolution, including branding.
  const packager = Object.assign(Object.create(PlatformPackager.prototype), {
    platform: Platform.MAC,
    appInfo: { productFilename },
    info: { projectDir: appOutDir, framework: { distMacOsAppName: 'Electron.app' } }
  })
  return { appOutDir, electronPlatformName: 'darwin', packager }
}

it('restores app localizations from the filtered framework without copying locale data', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-locale-pack-'))
  try {
    await writeDesktopDist(root)
    const ctx = context(root)
    const framework = ctx.packager.getMacOsElectronFrameworkResourcesDir(root)
    const resources = ctx.packager.getResourcesDir(root)
    await mkdir(resources, { recursive: true })
    for (const name of ['nb.lproj', 'en_GB.lproj', 'nb_FEMININE.lproj']) {
      await mkdir(path.join(framework, name), { recursive: true })
      await writeFile(path.join(framework, name, 'locale.pak'), 'untouched locale data')
    }
    await writeFile(path.join(framework, 'not-a-directory.lproj'), 'not a locale')
    await mkdir(path.join(framework, 'other'), { recursive: true })
    await configuredHook(ctx)
    await configuredHook(ctx)
    expect((await readdir(resources)).sort()).toEqual(['app.asar.unpacked', 'en_GB.lproj', 'nb.lproj'])
    expect(await readdir(path.join(resources, 'nb.lproj'))).toEqual([])
    expect(await readFile(path.join(framework, 'nb.lproj', 'locale.pak'), 'utf8')).toBe('untouched locale data')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

it('copies the desktop bundle before reporting a missing macOS framework', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-locale-pack-'))
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
  try {
    await writeDesktopDist(root)
    await configuredHook(context(root))
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('macOS locale markers were not restored'))
    await access(
      path.join(root, 'Hermes Preview.app', 'Contents', 'Resources', 'app.asar.unpacked', 'dist', 'electron-main.mjs')
    )
  } finally {
    warn.mockRestore()
    await rm(root, { recursive: true, force: true })
  }
})

it('copies the runnable desktop bundle into app.asar.unpacked on every platform', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-unpacked-dist-'))
  try {
    const projectDir = path.join(root, 'project')
    const appOutDir = path.join(root, 'win-unpacked')
    await mkdir(path.join(projectDir, 'dist', 'assets'), { recursive: true })
    await writeFile(path.join(projectDir, 'dist', 'electron-main.mjs'), 'main bundle')
    await writeFile(path.join(projectDir, 'dist', 'index.html'), '<main />')
    await writeFile(path.join(projectDir, 'dist', 'assets', 'renderer.js'), 'renderer bundle')

    await configuredHook({
      appOutDir,
      electronPlatformName: 'win32',
      packager: {
        projectDir,
        getResourcesDir: output => path.join(output, 'resources')
      }
    })

    const unpackedDist = path.join(appOutDir, 'resources', 'app.asar.unpacked', 'dist')
    await access(path.join(unpackedDist, 'electron-main.mjs'))
    await access(path.join(unpackedDist, 'index.html'))
    await access(path.join(unpackedDist, 'assets', 'renderer.js'))
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

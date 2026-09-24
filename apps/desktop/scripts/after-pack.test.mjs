import { mkdtemp, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises'
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

function context(appOutDir, productFilename = 'Hermes Preview') {
  // Use electron-builder's real bundle path resolution, including branding.
  const packager = Object.assign(Object.create(PlatformPackager.prototype), {
    platform: Platform.MAC,
    appInfo: { productFilename },
    info: { framework: { distMacOsAppName: 'Electron.app' } }
  })
  return { appOutDir, electronPlatformName: 'darwin', packager }
}

it('restores app localizations from the filtered framework without copying locale data', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-locale-pack-'))
  try {
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
    expect((await readdir(resources)).sort()).toEqual(['en_GB.lproj', 'nb.lproj'])
    expect(await readdir(path.join(resources, 'nb.lproj'))).toEqual([])
    expect(await readFile(path.join(framework, 'nb.lproj', 'locale.pak'), 'utf8')).toBe('untouched locale data')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

it('leaves other platforms alone and reports a missing framework without failing packaging', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-locale-pack-'))
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
  try {
    for (const electronPlatformName of ['linux', 'win32']) {
      await configuredHook({ appOutDir: root, electronPlatformName })
    }
    expect(await readdir(root)).toEqual([])
    expect(warn).not.toHaveBeenCalled()
    await configuredHook(context(root))
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('macOS locale markers were not restored'))
    expect(await readdir(root)).toEqual([])
  } finally {
    warn.mockRestore()
    await rm(root, { recursive: true, force: true })
  }
})

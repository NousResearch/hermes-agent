import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import {
  migrateProfileScopedDesktopPlugins,
  PACKAGE_MARKER,
  reconcileUnifiedDesktopHalves
} from './desktop-plugins-root'

const homes: string[] = []

function makeHome(): string {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-dp-root-'))
  homes.push(home)

  return home
}

function write(file: string, text: string) {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, text)
}

afterEach(() => {
  for (const home of homes.splice(0)) {
    fs.rmSync(home, { force: true, recursive: true })
  }
})

describe('migrateProfileScopedDesktopPlugins', () => {
  it('lifts per-profile desktop plugins into the app root so they survive a profile switch', async () => {
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    write(path.join(home, 'profiles', 'workbot', 'desktop-plugins', 'hello', 'plugin.js'), 'export default {}')
    fs.mkdirSync(appRoot, { recursive: true })

    const moved = await migrateProfileScopedDesktopPlugins(home, appRoot)

    expect(moved).toEqual([path.join(appRoot, 'hello')])
    expect(fs.existsSync(path.join(appRoot, 'hello', 'plugin.js'))).toBe(true)
    expect(fs.existsSync(path.join(home, 'profiles', 'workbot', 'desktop-plugins'))).toBe(false)
  })

  it('never overwrites a plugin already installed at the app root', async () => {
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    write(path.join(appRoot, 'hello', 'plugin.js'), 'root copy')
    const scoped = path.join(home, 'profiles', 'workbot', 'desktop-plugins', 'hello', 'plugin.js')
    write(scoped, 'profile copy')

    expect(await migrateProfileScopedDesktopPlugins(home, appRoot)).toEqual([])
    expect(fs.readFileSync(path.join(appRoot, 'hello', 'plugin.js'), 'utf8')).toBe('root copy')
    expect(fs.existsSync(scoped)).toBe(true)
  })
})

describe('reconcileUnifiedDesktopHalves', () => {
  it('copies each unified package desktop half into the app root ONCE across all profiles, stamping the package marker', async () => {
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    write(path.join(home, 'plugins', 'media', 'plugin.yaml'), 'name: media')
    write(path.join(home, 'plugins', 'media', 'desktop', 'plugin.js'), 'default home copy')
    // Same package installed in a second profile — must not produce a second row.
    write(path.join(home, 'profiles', 'workbot', 'plugins', 'media', 'desktop', 'plugin.js'), 'profile copy')
    // Agent-only package: nothing to copy.
    write(path.join(home, 'profiles', 'workbot', 'plugins', 'snap', 'plugin.yaml'), 'name: snap')

    const touched = await reconcileUnifiedDesktopHalves(home, appRoot)

    expect(touched).toEqual([path.join(appRoot, 'media')])
    expect(fs.readFileSync(path.join(appRoot, 'media', 'plugin.js'), 'utf8')).toBe('default home copy')
    const marker = JSON.parse(fs.readFileSync(path.join(appRoot, 'media', PACKAGE_MARKER), 'utf8'))
    expect(marker.package).toBe('media')
    expect(marker.repo).toBeUndefined()
    expect(fs.existsSync(path.join(appRoot, 'snap'))).toBe(false)

    // Idempotent: a second pass with nothing changed touches nothing.
    expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([])
  })

  it('re-copies when the source half changes, and removes the copy when the package is uninstalled', async () => {
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    const source = path.join(home, 'plugins', 'media', 'desktop', 'plugin.js')
    write(source, 'v1')
    await reconcileUnifiedDesktopHalves(home, appRoot)

    // Update: bump mtime forward and change content.
    write(source, 'v2')
    const future = new Date(Date.now() + 60_000)
    fs.utimesSync(source, future, future)
    expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([path.join(appRoot, 'media')])
    expect(fs.readFileSync(path.join(appRoot, 'media', 'plugin.js'), 'utf8')).toBe('v2')

    // Uninstall: the package folder goes away → so does the app-root copy.
    fs.rmSync(path.join(home, 'plugins', 'media'), { force: true, recursive: true })
    expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([path.join(appRoot, 'media')])
    expect(fs.existsSync(path.join(appRoot, 'media'))).toBe(false)
  })

  it('stamps the package origin (catalog sidecar, else git remote) so "Install here" can reinstall the agent half', async () => {
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    write(path.join(home, 'plugins', 'cat', 'desktop', 'plugin.js'), 'x')
    write(
      path.join(home, 'plugins', 'cat', '.hermes-catalog.json'),
      JSON.stringify({ catalog_name: 'cat', repo: 'https://github.com/o/cat.git', sha: 'deadbeef' })
    )
    write(path.join(home, 'plugins', 'raw', 'desktop', 'plugin.js'), 'y')
    write(
      path.join(home, 'plugins', 'raw', '.git', 'config'),
      '[core]\n\tbare = false\n[remote "origin"]\n\turl = file:///srv/raw.git\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n'
    )

    await reconcileUnifiedDesktopHalves(home, appRoot)

    const cat = JSON.parse(fs.readFileSync(path.join(appRoot, 'cat', PACKAGE_MARKER), 'utf8'))
    expect(cat).toMatchObject({ catalogName: 'cat', repo: 'https://github.com/o/cat.git', sha: 'deadbeef' })
    const raw = JSON.parse(fs.readFileSync(path.join(appRoot, 'raw', PACKAGE_MARKER), 'utf8'))
    expect(raw.repo).toBe('file:///srv/raw.git')
  })

  it('never overwrites a standalone plugin the user installed under the same name', async () => {
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    write(path.join(appRoot, 'media', 'plugin.js'), 'user standalone')
    write(path.join(home, 'plugins', 'media', 'desktop', 'plugin.js'), 'package half')

    expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([])
    expect(fs.readFileSync(path.join(appRoot, 'media', 'plugin.js'), 'utf8')).toBe('user standalone')
  })
})

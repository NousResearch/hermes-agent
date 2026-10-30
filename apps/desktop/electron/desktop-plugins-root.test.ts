import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  materializeDesktopHalf,
  migrateProfileScopedDesktopPlugins,
  PACKAGE_MARKER,
  reconcileDesktopHalvesReport,
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

  it.skipIf(process.platform === 'win32')(
    'materializes a SYMLINKED dev package and re-syncs on any byte change, even with an older mtime',
    async () => {
      // `ln -s ~/src/my-plugin ~/.hermes/plugins/` is how a plugin is developed in
      // place; the Python loader follows it, but the reconcile skipped the link
      // (Dirent.isDirectory() is false for a symlink) — the desktop half never loaded.
      const home = makeHome()
      const appRoot = path.join(home, 'desktop-plugins')
      const checkout = path.join(home, 'src', 'media')
      const source = path.join(checkout, 'desktop', 'plugin.js')
      write(source, 'v1')
      fs.mkdirSync(path.join(home, 'plugins'), { recursive: true })
      fs.symlinkSync(checkout, path.join(home, 'plugins', 'media'))

      expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([path.join(appRoot, 'media')])
      expect(fs.readFileSync(path.join(appRoot, 'media', 'plugin.js'), 'utf8')).toBe('v1')
      const marker = JSON.parse(fs.readFileSync(path.join(appRoot, 'media', PACKAGE_MARKER), 'utf8'))
      expect(marker).toMatchObject({
        linked: true,
        package: 'media',
        source: path.join(home, 'plugins', 'media', 'desktop')
      })
      expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([])

      // An edit that lands an OLDER mtime (git stash pop, cp -p) still re-syncs.
      write(source, 'v2')
      const past = new Date(Date.now() - 3_600_000)
      fs.utimesSync(source, past, past)
      expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([path.join(appRoot, 'media')])
      expect(fs.readFileSync(path.join(appRoot, 'media', 'plugin.js'), 'utf8')).toBe('v2')
    }
  )

  it('keeps the mtime gate for an installed (non-linked) package', async () => {
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    const source = path.join(home, 'plugins', 'media', 'desktop', 'plugin.js')
    write(source, 'v1')
    await reconcileUnifiedDesktopHalves(home, appRoot)
    const marker = JSON.parse(fs.readFileSync(path.join(appRoot, 'media', PACKAGE_MARKER), 'utf8'))
    expect(marker.linked).toBeUndefined()

    write(source, 'v2')
    const past = new Date(Date.now() - 3_600_000)
    fs.utimesSync(source, past, past)
    expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([])
  })

  it.skipIf(process.platform === 'win32' || process.getuid?.() === 0)(
    'skips a package the app cannot read and still materializes its siblings',
    async () => {
      // #111804: one unreadable plugin folder rejected the whole reconcile, so the
      // desktop-plugins root never resolved and no desktop plugin loaded.
      const home = makeHome()
      const appRoot = path.join(home, 'desktop-plugins')
      const denied = path.join(home, 'plugins', 'denied', 'desktop', 'plugin.js')
      write(denied, 'x')
      write(path.join(home, 'plugins', 'good', 'desktop', 'plugin.js'), 'y')
      fs.chmodSync(denied, 0)
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

      try {
        expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([path.join(appRoot, 'good')])
        expect(warn).toHaveBeenCalled()
      } finally {
        warn.mockRestore()
        fs.chmodSync(denied, 0o600)
      }
    }
  )

  it.skipIf(process.platform === 'win32' || process.getuid?.() === 0)(
    'keeps (and warns about) the desktop half of a package folder it can no longer read',
    async () => {
      // A source the app cannot stat (Windows ACL EPERM, mode-000 folder) is not
      // an uninstall: the ghost-prune loop must not rm the materialized half.
      const home = makeHome()
      const appRoot = path.join(home, 'desktop-plugins')
      const denied = path.join(home, 'plugins', 'denied')
      write(path.join(denied, 'desktop', 'plugin.js'), 'x')
      await reconcileUnifiedDesktopHalves(home, appRoot)
      expect(fs.existsSync(path.join(appRoot, 'denied'))).toBe(true)
      fs.chmodSync(denied, 0)
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

      try {
        expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([])
        expect(fs.existsSync(path.join(appRoot, 'denied'))).toBe(true)
        expect(warn).toHaveBeenCalled()
      } finally {
        warn.mockRestore()
        fs.chmodSync(denied, 0o700)
      }
    }
  )

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

  it('copies a root-layout package (plugin.js at the package root) as the installers publish it, minus .git and __pycache__', async () => {
    // plugin.yaml + __init__.py + a ROOT plugin.js: the probe reports both halves, so a local-backend
    // install puts the agent half down alone and counts on this pass for the desktop one.
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    const pkg = path.join(home, 'plugins', 'radar')
    write(path.join(pkg, 'plugin.yaml'), 'name: radar')
    write(path.join(pkg, '__init__.py'), 'def register(ctx):\n    pass\n')
    write(path.join(pkg, 'plugin.js'), 'root entry')
    write(path.join(pkg, 'panel.js'), 'asset')
    write(path.join(pkg, '__pycache__', 'x.pyc'), 'pyc')
    write(path.join(pkg, '.git', 'config'), '[remote "origin"]\n\turl = https://github.com/o/radar\n')
    // A root entry beside a link would publish whatever the link names: refused, never copied.
    const leaky = path.join(home, 'plugins', 'leaky')
    write(path.join(leaky, 'plugin.js'), 'x')
    write(path.join(home, 'secret'), 's')
    fs.symlinkSync(path.join(home, 'secret'), path.join(leaky, 'secret'))
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    try {
      expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([path.join(appRoot, 'radar')])
    } finally {
      warn.mockRestore()
    }

    expect(fs.readFileSync(path.join(appRoot, 'radar', 'plugin.js'), 'utf8')).toBe('root entry')
    expect(fs.readFileSync(path.join(appRoot, 'radar', 'panel.js'), 'utf8')).toBe('asset')
    expect(fs.existsSync(path.join(appRoot, 'radar', '.git'))).toBe(false)
    expect(fs.existsSync(path.join(appRoot, 'radar', '__pycache__'))).toBe(false)
    expect(fs.existsSync(path.join(appRoot, 'leaky'))).toBe(false)
    const marker = JSON.parse(fs.readFileSync(path.join(appRoot, 'radar', PACKAGE_MARKER), 'utf8'))
    expect(marker).toMatchObject({ package: 'radar', source: pkg, repo: 'https://github.com/o/radar' })
    expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([])

    fs.rmSync(path.join(pkg, 'plugin.js'))
    expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([path.join(appRoot, 'radar')])
    expect(fs.existsSync(path.join(appRoot, 'radar'))).toBe(false)
  })

  it.skipIf(process.platform === 'win32')(
    'publishes a real copy of a LINKED root-layout checkout, keeps its marker out of the checkout, and prunes it with the link',
    async () => {
      const home = makeHome()
      const appRoot = path.join(home, 'desktop-plugins')
      const checkout = path.join(home, 'src', 'radar')
      write(path.join(checkout, 'plugin.yaml'), 'name: radar')
      write(path.join(checkout, 'plugin.js'), 'root entry')
      fs.mkdirSync(path.join(home, 'plugins'), { recursive: true })
      const link = path.join(home, 'plugins', 'radar')
      fs.symlinkSync(checkout, link)
      const target = path.join(appRoot, 'radar')

      expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([target])
      expect(fs.lstatSync(target).isSymbolicLink()).toBe(false)
      expect(fs.readFileSync(path.join(target, 'plugin.js'), 'utf8')).toBe('root entry')
      expect(JSON.parse(fs.readFileSync(path.join(target, PACKAGE_MARKER), 'utf8'))).toMatchObject({
        linked: true,
        package: 'radar',
        source: link
      })
      expect(fs.existsSync(path.join(checkout, PACKAGE_MARKER))).toBe(false)
      expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([])

      fs.unlinkSync(link)
      expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([target])
      expect(fs.existsSync(target)).toBe(false)
      expect(fs.readdirSync(checkout).sort()).toEqual(['plugin.js', 'plugin.yaml'])
    }
  )

  it.skipIf(process.platform === 'win32')(
    'replaces an earlier copy that kept the checkout link, and drops it once the package link is gone',
    async () => {
      // What a pass that copied the link itself left: <appRoot>/radar -> the checkout, marker inside it.
      const home = makeHome()
      const appRoot = path.join(home, 'desktop-plugins')
      const checkout = path.join(home, 'src', 'radar')
      const link = path.join(home, 'plugins', 'radar')
      const target = path.join(appRoot, 'radar')
      write(path.join(checkout, 'plugin.js'), 'root entry')
      const ghost = { linked: true, package: 'radar', source: link, sourceMtimeMs: Date.now() + 60_000 }
      write(path.join(checkout, PACKAGE_MARKER), JSON.stringify(ghost))
      fs.mkdirSync(path.join(home, 'plugins'), { recursive: true })
      fs.mkdirSync(appRoot, { recursive: true })
      fs.symlinkSync(checkout, link)
      fs.symlinkSync(checkout, target)

      expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([target])
      expect(fs.lstatSync(target).isSymbolicLink()).toBe(false)
      expect(fs.readFileSync(path.join(target, 'plugin.js'), 'utf8')).toBe('root entry')

      // The same leftover once the package link is gone: the link in the app root goes, the checkout stays.
      fs.rmSync(target, { force: true, recursive: true })
      fs.symlinkSync(checkout, target)
      fs.unlinkSync(link)
      expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([target])
      expect(fs.existsSync(target)).toBe(false)
      expect(fs.readFileSync(path.join(checkout, 'plugin.js'), 'utf8')).toBe('root entry')
    }
  )

  it.skipIf(process.platform === 'win32')('never stamps a marker into a plugin the user linked into the app root', async () => {
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    const own = path.join(home, 'src', 'pkg')
    write(path.join(own, 'plugin.js'), 'same')
    write(path.join(home, 'plugins', 'pkg', 'desktop', 'plugin.js'), 'same')
    fs.mkdirSync(appRoot, { recursive: true })
    fs.symlinkSync(own, path.join(appRoot, 'pkg'))

    expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([])
    expect(fs.existsSync(path.join(own, PACKAGE_MARKER))).toBe(false)
    expect(fs.lstatSync(path.join(appRoot, 'pkg')).isSymbolicLink()).toBe(true)
  })

  it('never overwrites a standalone plugin the user installed under the same name', async () => {
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    write(path.join(appRoot, 'media', 'plugin.js'), 'user standalone')
    write(path.join(home, 'plugins', 'media', 'desktop', 'plugin.js'), 'package half')

    expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([])
    expect(fs.readFileSync(path.join(appRoot, 'media', 'plugin.js'), 'utf8')).toBe('user standalone')
  })

  it('adopts an unmarked copy of its own half that a pre-marker install left behind', async () => {
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    // What `installDesktopPluginFromGit` used to publish: the package's half,
    // byte for byte, with no marker. The page showed it as a second, enabled
    // row while the agent row waited on "copying…".
    write(path.join(home, 'plugins', 'media', 'desktop', 'plugin.js'), 'package half')
    write(path.join(appRoot, 'media', 'plugin.js'), 'package half')

    expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([path.join(appRoot, 'media')])

    const marker = JSON.parse(fs.readFileSync(path.join(appRoot, 'media', PACKAGE_MARKER), 'utf8'))
    expect(marker.package).toBe('media')
    // Adoption is a stamp, not a copy: the folder's own files are untouched and
    // the next pass has nothing left to do.
    expect(fs.readFileSync(path.join(appRoot, 'media', 'plugin.js'), 'utf8')).toBe('package half')
    expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([])
  })

  it('replaces an interrupted marker-less copy, stamps it, and converges on retry', async () => {
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    write(path.join(home, 'plugins', 'media', 'desktop', 'plugin.js'), 'package half')
    // A failed pre-marker copy has files but no entry point and is not a
    // standalone plugin the user can run.
    write(path.join(appRoot, 'media', 'partial.js'), 'interrupted copy')

    expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([path.join(appRoot, 'media')])
    expect(fs.readFileSync(path.join(appRoot, 'media', 'plugin.js'), 'utf8')).toBe('package half')
    expect(fs.existsSync(path.join(appRoot, 'media', 'partial.js'))).toBe(false)
    expect(fs.existsSync(path.join(appRoot, 'media', PACKAGE_MARKER))).toBe(true)
    expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([])
  })

  it('overlapping passes publish a half once, from the first home that has it', async () => {
    // Every window's disk poll and the Plugins page each run a pass through IPC,
    // so passes overlap at boot. Unserialized, both copied, one rename hit the
    // other's fresh folder (ENOTEMPTY) and that pass then took the profile's
    // same-named package instead of the default home's.
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    write(path.join(home, 'plugins', 'media', 'desktop', 'plugin.js'), 'default home copy')
    write(path.join(home, 'profiles', 'workbot', 'plugins', 'media', 'desktop', 'plugin.js'), 'profile copy')
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    let passes: string[][]

    try {
      passes = await Promise.all([1, 2, 3].map(() => reconcileUnifiedDesktopHalves(home, appRoot)))
      expect(warn).not.toHaveBeenCalled()
    } finally {
      warn.mockRestore()
    }

    expect(passes.flat()).toEqual([path.join(appRoot, 'media')])
    expect(fs.readFileSync(path.join(appRoot, 'media', 'plugin.js'), 'utf8')).toBe('default home copy')
    expect(fs.readdirSync(appRoot)).toEqual(['media'])
    expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([])
  })

  it('cleans up failed staging copies without exposing a marker-less target', async () => {
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    const packageDir = path.join(home, 'plugins', 'media')
    write(path.join(packageDir, 'desktop', 'plugin.js'), 'package half')

    // Simulate a copy that dies partway: the destination already holds a marker-less
    // partial tree when the failure surfaces. A direct copy into the final target would
    // leave that half-tree behind; the staged copy must never let it reach `<appRoot>/media`.
    const copy = vi.spyOn(fs.promises, 'cp').mockImplementationOnce(async (_src, dest) => {
      write(path.join(String(dest), 'plugin.js'), 'partial')
      throw new Error('disk full')
    })

    await expect(materializeDesktopHalf(packageDir, appRoot)).rejects.toThrow('disk full')

    copy.mockRestore()
    expect(fs.existsSync(path.join(appRoot, 'media'))).toBe(false)
    expect(fs.readdirSync(appRoot)).toEqual([])
  })
})

describe('desktop tree publication', () => {
  it.skipIf(process.platform === 'win32')(
    'refuses an installed combined package whose desktop half links elsewhere in the package',
    async () => {
      // The Desktop install modal lands a combined package through the agent install, then publishes its
      // desktop half from plugins/<name>/desktop via this reconcile, not through the git installer.
      const home = makeHome()
      const appRoot = path.join(home, 'desktop-plugins')
      const pkg = path.join(home, 'plugins', 'media')
      write(path.join(pkg, 'plugin.yaml'), 'name: media\n')
      write(path.join(pkg, 'assets', 'asset.txt'), 'package file')
      write(path.join(pkg, 'desktop', 'plugin.js'), 'export default {}')
      fs.symlinkSync('../assets/asset.txt', path.join(pkg, 'desktop', 'asset.txt'))
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

      try {
        expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([])
        expect(warn.mock.calls.flat().join(' ')).toContain('symbolic link (asset.txt)')
      } finally {
        warn.mockRestore()
      }

      // Nothing published and nothing staged.
      expect(fs.existsSync(appRoot) ? fs.readdirSync(appRoot) : []).toEqual([])
    }
  )

  it.skipIf(process.platform === 'win32')('refuses an installed package whose desktop folder is a link', async () => {
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    write(path.join(home, 'elsewhere', 'plugin.js'), 'export default {}')
    fs.mkdirSync(path.join(home, 'plugins', 'media'), { recursive: true })
    fs.symlinkSync(path.join(home, 'elsewhere'), path.join(home, 'plugins', 'media', 'desktop'))

    await expect(materializeDesktopHalf(path.join(home, 'plugins', 'media'), appRoot)).rejects.toThrow(
      'folder is a symbolic link'
    )
    expect(fs.existsSync(path.join(appRoot, 'media'))).toBe(false)
  })

  it.skipIf(process.platform === 'win32')(
    'still publishes a development checkout the user linked in, links inside it included',
    async () => {
      const home = makeHome()
      const appRoot = path.join(home, 'desktop-plugins')
      const checkout = path.join(home, 'src', 'media')
      write(path.join(checkout, 'desktop', 'plugin.js'), 'export default {}')
      write(path.join(checkout, 'assets', 'asset.txt'), 'checkout file')
      fs.symlinkSync('../assets/asset.txt', path.join(checkout, 'desktop', 'asset.txt'))
      fs.mkdirSync(path.join(home, 'plugins'), { recursive: true })
      fs.symlinkSync(checkout, path.join(home, 'plugins', 'media'))

      expect(await reconcileUnifiedDesktopHalves(home, appRoot)).toEqual([path.join(appRoot, 'media')])
      expect(fs.readFileSync(path.join(appRoot, 'media', 'asset.txt'), 'utf8')).toBe('checkout file')
    }
  )

  it.skipIf(process.platform === 'win32')(
    'names each refused package and its reason beside the halves it did publish',
    async () => {
      // The install dialog reads this to report a refused desktop half as failed, not installed.
      const home = makeHome()
      const appRoot = path.join(home, 'desktop-plugins')
      const pkg = path.join(home, 'plugins', 'media')
      write(path.join(pkg, 'assets', 'asset.txt'), 'package file')
      write(path.join(pkg, 'desktop', 'plugin.js'), 'export default {}')
      fs.symlinkSync('../assets/asset.txt', path.join(pkg, 'desktop', 'asset.txt'))
      write(path.join(home, 'plugins', 'good', 'desktop', 'plugin.js'), 'export default {}')
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

      try {
        expect(await reconcileDesktopHalvesReport(home, appRoot)).toEqual({
          published: [path.join(appRoot, 'good')],
          refused: [
            { name: 'media', reason: 'The desktop plugin contains a symbolic link (asset.txt); it is not installed.' }
          ]
        })
      } finally {
        warn.mockRestore()
      }
    }
  )

  it.skipIf(process.platform === 'win32')(
    'does not report a refusal once another profile published the same package',
    async () => {
      const home = makeHome()
      const appRoot = path.join(home, 'desktop-plugins')
      const pkg = path.join(home, 'plugins', 'media')
      write(path.join(pkg, 'assets', 'asset.txt'), 'package file')
      write(path.join(pkg, 'desktop', 'plugin.js'), 'export default {}')
      fs.symlinkSync('../assets/asset.txt', path.join(pkg, 'desktop', 'asset.txt'))
      write(path.join(home, 'profiles', 'workbot', 'plugins', 'media', 'desktop', 'plugin.js'), 'profile copy')
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

      try {
        expect(await reconcileDesktopHalvesReport(home, appRoot)).toEqual({
          published: [path.join(appRoot, 'media')],
          refused: []
        })
      } finally {
        warn.mockRestore()
      }
    }
  )
})

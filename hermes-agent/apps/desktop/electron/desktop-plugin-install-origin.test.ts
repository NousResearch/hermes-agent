import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { afterEach, describe, expect, it } from 'vitest'

import { installDesktopPluginFromGit } from './desktop-plugin-install'
import { INSTALL_ORIGIN, PACKAGE_MARKER } from './desktop-plugins-root'

// A desktop-only catalog entry (Accent Picker) installs as a standalone plugin with no package
// marker. Without its catalog entry on disk the card never showed it installed, and the next
// Install stopped at "already exists" (reported on X, 2026-11-07).
const roots: string[] = []

afterEach(() => {
  for (const root of roots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

function mkdtemp(prefix: string) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix))
  roots.push(dir)

  return dir
}

function git(repo: string, ...args: string[]) {
  return execFileSync('git', args, { cwd: repo, stdio: 'pipe' }).toString().trim()
}

/** A desktop-only repo: a manifest beside `desktop/plugin.js`, no agent payload. */
function desktopOnlyRepo(extra: Record<string, string> = {}): string {
  const repo = mkdtemp('hermes-origin-repo-')
  fs.mkdirSync(path.join(repo, 'desktop'))
  fs.writeFileSync(path.join(repo, 'plugin.yaml'), 'name: accent-picker\n')
  fs.writeFileSync(path.join(repo, 'desktop', 'plugin.js'), 'export default { id: "accent" }\n')

  for (const [name, text] of Object.entries(extra)) {
    fs.writeFileSync(path.join(repo, 'desktop', name), text)
  }

  git(repo, 'init', '-q')
  git(repo, 'add', '.')
  git(repo, '-c', 'user.email=f@example.com', '-c', 'user.name=F', 'commit', '-qm', 'init')

  return repo
}

const originOf = (dir: string) => JSON.parse(fs.readFileSync(path.join(dir, INSTALL_ORIGIN), 'utf8'))

describe('a standalone desktop plugin installed from the catalog', () => {
  it('records its catalog entry, and stays standalone (no package marker)', async () => {
    const repo = desktopOnlyRepo()
    const appRoot = mkdtemp('hermes-origin-root-')
    const sha = git(repo, 'rev-parse', 'HEAD')

    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, {
      sha,
      catalogName: 'accent-picker'
    })

    expect(result).toMatchObject({ ok: true })
    const dir = path.join(appRoot, result.pluginName!)
    expect(originOf(dir)).toEqual({ catalogName: 'accent-picker', repo: pathToFileURL(repo).href, sha })
    expect(fs.existsSync(path.join(dir, PACKAGE_MARKER))).toBe(false)
  })

  it('records nothing for a Git URL install, and drops an origin file the repo ships', async () => {
    const forged = JSON.stringify({ catalogName: 'trusted-entry', repo: 'https://github.com/x/y', sha: 'a'.repeat(40) })
    const repo = desktopOnlyRepo({ [INSTALL_ORIGIN]: forged })
    const appRoot = mkdtemp('hermes-origin-root-')

    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot)

    expect(result).toMatchObject({ ok: true })
    expect(fs.existsSync(path.join(appRoot, result.pluginName!, INSTALL_ORIGIN))).toBe(false)
  })

  it('adopts an earlier install of the same tree instead of answering "already exists"', async () => {
    const repo = desktopOnlyRepo({ 'panel.js': 'mine\n' })
    const appRoot = mkdtemp('hermes-origin-root-')
    const sha = git(repo, 'rev-parse', 'HEAD')
    // The install every build before this one made: no origin recorded.
    const first = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, { sha })
    const dir = path.join(appRoot, first.pluginName!)
    fs.writeFileSync(path.join(dir, 'state.json'), 'kept\n')

    const again = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, {
      sha,
      catalogName: 'accent-picker'
    })

    expect(again).toMatchObject({ ok: true, pluginName: first.pluginName })
    expect(originOf(dir).catalogName).toBe('accent-picker')
    // Adopted in place: nothing in the folder was replaced.
    expect(fs.readFileSync(path.join(dir, 'state.json'), 'utf8')).toBe('kept\n')
  })

  it('still refuses to replace a different plugin of that name without Force', async () => {
    const repo = desktopOnlyRepo()
    const appRoot = mkdtemp('hermes-origin-root-')
    const sha = git(repo, 'rev-parse', 'HEAD')
    const first = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, { sha })
    const dir = path.join(appRoot, first.pluginName!)
    fs.writeFileSync(path.join(dir, 'plugin.js'), 'export default { id: "my-own-edit" }\n')

    const again = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, {
      sha,
      catalogName: 'accent-picker'
    })

    expect(again).toMatchObject({ ok: false, exists: true })
    expect(again.error).toContain('already exists')
    expect(fs.existsSync(path.join(dir, INSTALL_ORIGIN))).toBe(false)
    expect(fs.readFileSync(path.join(dir, 'plugin.js'), 'utf8')).toContain('my-own-edit')
  })

  it('records the commit it cloned when the install names no pin', async () => {
    const repo = desktopOnlyRepo()
    const appRoot = mkdtemp('hermes-origin-root-')

    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, {
      catalogName: 'accent-picker'
    })

    expect(result.ok).toBe(true)
    expect(originOf(path.join(appRoot, result.pluginName!))).toEqual({
      catalogName: 'accent-picker',
      repo: pathToFileURL(repo).href,
      sha: git(repo, 'rev-parse', 'HEAD')
    })
  })

  it('records the commit it checked out when the pin is an annotated tag', async () => {
    const repo = desktopOnlyRepo()
    const appRoot = mkdtemp('hermes-origin-root-')
    const commit = git(repo, 'rev-parse', 'HEAD')
    git(repo, '-c', 'user.email=t@example.invalid', '-c', 'user.name=t', 'tag', '-a', 'v1', '-m', 'v1')
    const tagObject = git(repo, 'rev-parse', 'v1')
    expect(tagObject).not.toBe(commit)

    const result = await installDesktopPluginFromGit('git', pathToFileURL(repo).href, appRoot, {
      sha: tagObject,
      catalogName: 'accent-picker'
    })

    expect(result).toMatchObject({ ok: true })
    expect(originOf(path.join(appRoot, result.pluginName!))).toMatchObject({ sha: commit, pin: tagObject })
  })
})

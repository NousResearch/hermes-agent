import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { migrateProfileScopedDesktopPlugins } from './desktop-plugins-root'

const homes: string[] = []

function makeHome(): string {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-dp-root-'))
  homes.push(home)

  return home
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
    const scoped = path.join(home, 'profiles', 'workbot', 'desktop-plugins', 'hello')
    fs.mkdirSync(scoped, { recursive: true })
    fs.writeFileSync(path.join(scoped, 'plugin.js'), 'export default { id: "hello" }')
    fs.mkdirSync(appRoot, { recursive: true })

    const moved = await migrateProfileScopedDesktopPlugins(home, appRoot)

    expect(moved).toEqual([path.join(appRoot, 'hello')])
    expect(fs.existsSync(path.join(appRoot, 'hello', 'plugin.js'))).toBe(true)
    expect(fs.existsSync(path.join(home, 'profiles', 'workbot', 'desktop-plugins'))).toBe(false)
  })

  it('never overwrites a plugin already installed at the app root', async () => {
    const home = makeHome()
    const appRoot = path.join(home, 'desktop-plugins')
    fs.mkdirSync(path.join(appRoot, 'hello'), { recursive: true })
    fs.writeFileSync(path.join(appRoot, 'hello', 'plugin.js'), 'root copy')
    const scoped = path.join(home, 'profiles', 'workbot', 'desktop-plugins', 'hello')
    fs.mkdirSync(scoped, { recursive: true })
    fs.writeFileSync(path.join(scoped, 'plugin.js'), 'profile copy')

    const moved = await migrateProfileScopedDesktopPlugins(home, appRoot)

    expect(moved).toEqual([])
    expect(fs.readFileSync(path.join(appRoot, 'hello', 'plugin.js'), 'utf8')).toBe('root copy')
    expect(fs.existsSync(path.join(scoped, 'plugin.js'))).toBe(true)
  })
})

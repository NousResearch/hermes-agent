import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { removeDesktopPlugin } from './desktop-plugin-remove'
import { PACKAGE_MARKER } from './desktop-plugins-root'

const homes: string[] = []

function makeRoot(): { home: string; root: string } {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-dp-remove-'))
  homes.push(home)
  const root = path.join(home, 'desktop-plugins')
  fs.mkdirSync(root, { recursive: true })

  return { home, root }
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

describe('removeDesktopPlugin', () => {
  it('deletes exactly the named standalone folder under the root and leaves its neighbours', async () => {
    const { root } = makeRoot()
    write(path.join(root, 'hello', 'plugin.js'), 'export default {}')
    write(path.join(root, 'hello', 'assets', 'icon.svg'), '<svg/>')
    write(path.join(root, 'other', 'plugin.js'), 'export default {}')

    const result = await removeDesktopPlugin(root, 'hello')

    expect(result).toEqual({ ok: true, path: path.join(root, 'hello') })
    expect(fs.existsSync(path.join(root, 'hello'))).toBe(false)
    expect(fs.existsSync(path.join(root, 'other', 'plugin.js'))).toBe(true)
  })

  it('refuses anything that is not a single folder name inside the root, deleting nothing', async () => {
    const { home, root } = makeRoot()
    write(path.join(home, 'config.yaml'), 'model: x')
    write(path.join(root, 'hello', 'plugin.js'), 'export default {}')

    for (const name of ['../config.yaml', '..', '.', '', 'hello/plugin.js', `..${path.sep}config.yaml`, path.join(root, 'hello')]) {
      const result = await removeDesktopPlugin(root, name)

      expect(result.ok, name).toBe(false)
    }

    expect(fs.existsSync(path.join(home, 'config.yaml'))).toBe(true)
    expect(fs.existsSync(path.join(root, 'hello', 'plugin.js'))).toBe(true)
  })

  it('refuses a unified package half (marker present) and a missing folder', async () => {
    const { root } = makeRoot()
    write(path.join(root, 'uni', 'plugin.js'), 'export default {}')
    write(path.join(root, 'uni', PACKAGE_MARKER), JSON.stringify({ package: 'uni', source: '/x' }))

    expect((await removeDesktopPlugin(root, 'uni')).ok).toBe(false)
    expect(fs.existsSync(path.join(root, 'uni', 'plugin.js'))).toBe(true)
    expect((await removeDesktopPlugin(root, 'ghost')).ok).toBe(false)
  })

  it('removes a symlinked plugin folder as the link, never following it', async () => {
    const { home, root } = makeRoot()
    const real = path.join(home, 'elsewhere')
    write(path.join(real, 'plugin.js'), 'export default {}')
    fs.symlinkSync(real, path.join(root, 'linked'), 'dir')

    expect((await removeDesktopPlugin(root, 'linked')).ok).toBe(true)
    expect(fs.existsSync(path.join(root, 'linked'))).toBe(false)
    expect(fs.existsSync(path.join(real, 'plugin.js'))).toBe(true)
  })
})

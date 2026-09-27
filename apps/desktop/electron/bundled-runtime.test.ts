import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, expect, test } from 'vitest'
import { bundledRuntimeBackend } from './bundled-runtime'

const roots: string[] = []
afterEach(() => roots.splice(0).forEach(root => fs.rmSync(root, { recursive: true, force: true })))

test.skipIf(process.platform !== 'win32')(
  'bundled backend uses its own Python and source, retaining the user profile',
  () => {
    const resourcesPath = fs.mkdtempSync(path.join(os.tmpdir(), 'czesiek-bundle-'))
    roots.push(resourcesPath)
    for (const file of [
      'runtime/python/python.exe',
      'runtime/agent/hermes_cli/main.py',
      'runtime/git/bin/bash.exe',
      'runtime/node/node.exe'
    ]) {
      const target = path.join(resourcesPath, file)
      fs.mkdirSync(path.dirname(target), { recursive: true })
      fs.writeFileSync(target, '')
    }
    fs.writeFileSync(
      path.join(resourcesPath, 'runtime/manifest.json'),
      JSON.stringify({
        schemaVersion: 1,
        platform: process.platform,
        arch: process.arch
      })
    )
    const backend = bundledRuntimeBackend({
      isPackaged: true,
      resourcesPath,
      hermesHome: path.join(resourcesPath, 'profile'),
      args: ['serve'],
      currentEnv: { PYTHONPATH: 'unrelated-install', VIRTUAL_ENV: 'unrelated-venv' }
    })!
    expect(backend.bootstrap).toBe(false)
    expect(backend.command).toBe(path.join(resourcesPath, 'runtime/python/python.exe'))
    expect(backend.env.PYTHONPATH).toBe(backend.root)
    expect(backend.env.VIRTUAL_ENV).toBe('')
    expect(backend.args).toEqual(['-m', 'hermes_cli.main', 'serve'])
    fs.unlinkSync(backend.command)
    expect(() => bundledRuntimeBackend({ isPackaged: true, resourcesPath, hermesHome: '', args: [] })).toThrow(
      /Zainstaluj ponownie/
    )
  }
)

test('development keeps its existing source runtime', () => {
  expect(bundledRuntimeBackend({ isPackaged: false, resourcesPath: '', hermesHome: '', args: [] })).toBeNull()
})

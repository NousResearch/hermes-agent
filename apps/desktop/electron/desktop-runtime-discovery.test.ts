import path from 'node:path'

import { expect, test } from 'vitest'

import { createDesktopRuntimeDiscovery } from './desktop-runtime-discovery'

test('runtime discovery keeps interpreter and venv ownership paired', async () => {
  const root = path.join(process.cwd(), 'runtime-discovery-fixture')
  const venv = path.join(root, '.venv')
  const python = path.join(venv, 'Scripts', 'python.exe')
  const existing = new Set([python, path.join(root, 'hermes_cli', 'main.py')])

  const runtime = createDesktopRuntimeDiscovery({
    hermesHome: root,
    isWindows: true,
    isWsl: false,
    fileExists: candidate => existing.has(candidate),
    directoryExists: candidate => candidate === root,
    rememberLog: () => undefined
  })

  expect(runtime.isHermesSourceRoot(root)).toBe(true)
  expect(runtime.findOnPath(python)).toBe(python)
  expect(runtime.getVenvPython(venv)).toBe(python)
  expect(await runtime.findPythonForRoot(root)).toBe(python)
  expect(runtime.venvRootForPython(python, root)).toBe(venv)
  expect(runtime.venvRootForPython(path.join(root, 'python.exe'), root)).toBeNull()
})

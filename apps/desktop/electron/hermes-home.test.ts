import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import { resolveHermesHome, windowsDefaultHermesHome } from './hermes-home'

function resolverOptions(options: Record<string, unknown> = {}) {
  const existing = new Set((options.existing as string[] | undefined) || [])

  return {
    homeDir: 'C:\\Users\\ada',
    isWindows: true,
    pathModule: path.win32,
    readUserEnvVar: () => null,
    directoryExists: (target: string) => existing.has(target),
    ...options
  }
}

test('Windows default Hermes home uses LOCALAPPDATA when present', () => {
  assert.equal(
    windowsDefaultHermesHome({
      env: { LOCALAPPDATA: 'C:\\Users\\ada\\AppData\\Local' },
      homeDir: 'C:\\Users\\ada',
      pathModule: path.win32
    }),
    'C:\\Users\\ada\\AppData\\Local\\hermes'
  )
})

test('Windows default Hermes home falls back to AppData\\Local when LOCALAPPDATA is absent', () => {
  assert.equal(
    windowsDefaultHermesHome({ env: {}, homeDir: 'C:\\Users\\ada', pathModule: path.win32 }),
    'C:\\Users\\ada\\AppData\\Local\\hermes'
  )
})

test('resolveHermesHome honors explicit HERMES_HOME before other sources', () => {
  assert.equal(
    resolveHermesHome(
      resolverOptions({
        env: { HERMES_HOME: 'D:\\Hermes\\profiles\\work' },
        userDataOverride: 'E:\\DesktopData',
        readUserEnvVar: () => 'F:\\RegistryHome'
      })
    ),
    'D:\\Hermes'
  )
})

test('resolveHermesHome honors the desktop user-data override before Windows defaults', () => {
  assert.equal(
    resolveHermesHome(
      resolverOptions({
        env: { LOCALAPPDATA: 'C:\\Users\\ada\\AppData\\Local' },
        userDataOverride: 'D:\\DesktopData',
        readUserEnvVar: () => 'E:\\RegistryHome'
      })
    ),
    'D:\\DesktopData\\hermes-home'
  )
})

test('resolveHermesHome reads the live Windows user environment before defaults', () => {
  assert.equal(
    resolveHermesHome(
      resolverOptions({
        env: { LOCALAPPDATA: 'C:\\Users\\ada\\AppData\\Local' },
        readUserEnvVar: (name: string) => (name === 'HERMES_HOME' ? 'E:\\Hermes' : null)
      })
    ),
    'E:\\Hermes'
  )
})

test('resolveHermesHome keeps an existing legacy Windows home when the new home does not exist', () => {
  assert.equal(
    resolveHermesHome(
      resolverOptions({
        env: { LOCALAPPDATA: 'C:\\Users\\ada\\AppData\\Local' },
        existing: ['C:\\Users\\ada\\.hermes']
      })
    ),
    'C:\\Users\\ada\\.hermes'
  )
})

test('resolveHermesHome prefers an existing LOCALAPPDATA home over the legacy home', () => {
  assert.equal(
    resolveHermesHome(
      resolverOptions({
        env: { LOCALAPPDATA: 'C:\\Users\\ada\\AppData\\Local' },
        existing: ['C:\\Users\\ada\\AppData\\Local\\hermes', 'C:\\Users\\ada\\.hermes']
      })
    ),
    'C:\\Users\\ada\\AppData\\Local\\hermes'
  )
})

test('resolveHermesHome aligns Windows fallback with Python when LOCALAPPDATA is missing', () => {
  assert.equal(
    resolveHermesHome(resolverOptions({ env: {} })),
    'C:\\Users\\ada\\AppData\\Local\\hermes'
  )
})

test('resolveHermesHome uses ~/.hermes on non-Windows platforms', () => {
  assert.equal(
    resolveHermesHome({ env: {}, homeDir: '/Users/ada', isWindows: false, pathModule: path.posix }),
    '/Users/ada/.hermes'
  )
})

import assert from 'node:assert/strict'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

import { createExecutableDiscoveryRuntime } from './executable-discovery-runtime'

const environmentKeys = ['PATH', 'LOCALAPPDATA', 'ProgramFiles', 'ProgramFiles(x86)'] as const
const originalEnvironment = Object.fromEntries(environmentKeys.map(key => [key, process.env[key]]))

afterEach(() => {
  for (const key of environmentKeys) {
    const original = originalEnvironment[key]

    if (original === undefined) {
      delete process.env[key]
    } else {
      process.env[key] = original
    }
  }
})

test('Windows Git keeps PortableGit precedence and GitHub CLI keeps a separate cache', () => {
  process.env.LOCALAPPDATA = 'C:/test-local-app-data'
  process.env.ProgramFiles = 'C:/test-program-files'
  process.env['ProgramFiles(x86)'] = 'C:/test-program-files-x86'

  const portableGit = path.join(process.env.LOCALAPPDATA, 'hermes', 'git', 'cmd', 'git.exe')
  const installedGit = path.join(process.env.ProgramFiles, 'Git', 'cmd', 'git.exe')
  const installedGh = path.join(process.env.ProgramFiles, 'GitHub CLI', 'gh.exe')
  const files = new Set([portableGit, installedGit, installedGh])
  const fileExists = vi.fn((file: string) => files.has(file))
  const findOnPath = vi.fn(() => null)
  const execFileSync = vi.fn()
  const getHomePath = vi.fn(() => 'C:/unused-home')

  const runtime = createExecutableDiscoveryRuntime({
    isWindows: true,
    fileExists,
    findOnPath,
    getHomePath,
    execFileSync
  })

  assert.equal(runtime.resolveGitBinary(), portableGit)
  files.delete(portableGit)
  assert.equal(runtime.resolveGitBinary(), portableGit, 'one controller keeps its selected Git binary')
  assert.equal(runtime.resolveGhBinary(), installedGh)
  assert.equal(execFileSync.mock.calls.length, 1, 'GitHub CLI is probed; Windows Git retains existence selection')
  assert.equal(getHomePath.mock.calls.length, 0, 'Windows Git and GitHub CLI never need the home directory')
  assert.equal(findOnPath.mock.calls.length, 0)
})

test('POSIX Git skips an unlaunchable first PATH hit and caches the working candidate', () => {
  const first = path.join('C:/test-bin-one', 'git')
  const second = path.join('C:/test-bin-two', 'git')
  process.env.PATH = ['C:/test-bin-one', 'C:/test-bin-two'].join(path.delimiter)
  const probes: string[] = []

  const runtime = createExecutableDiscoveryRuntime({
    isWindows: false,
    fileExists: (file: string) => file === first || file === second,
    findOnPath: () => null,
    getHomePath: () => 'C:/unused-home',
    execFileSync: (candidate: string, args, options) => {
      probes.push(candidate)
      assert.deepEqual(args, ['--version'])
      assert.deepEqual(options, { stdio: 'ignore', timeout: 5000, windowsHide: true })

      if (candidate === first) {
        throw new Error('wrong architecture')
      }

      return ''
    }
  })

  assert.equal(runtime.resolveGitBinary(), second)
  assert.deepEqual(probes, [first, second])
  process.env.PATH = 'C:/test-bin-three'
  assert.equal(runtime.resolveGitBinary(), second)
  assert.deepEqual(probes, [first, second], 'a cached binary is not probed again')
})

test('GitHub CLI checks fixed paths before PATH, asks for home lazily, and falls back to a bare name', () => {
  const pathGh = path.join('C:/test-bin', 'gh')
  process.env.PATH = 'C:/test-bin'
  const getHomePath = vi.fn(() => 'C:/test-home')

  const runtime = createExecutableDiscoveryRuntime({
    isWindows: false,
    fileExists: (file: string) => file === '/opt/homebrew/bin/gh' || file === pathGh,
    findOnPath: () => null,
    getHomePath,
    execFileSync: (candidate: string) => {
      if (candidate === '/opt/homebrew/bin/gh') {
        throw new Error('unlaunchable')
      }

      return ''
    }
  })

  assert.equal(getHomePath.mock.calls.length, 0)
  assert.equal(runtime.resolveGhBinary(), pathGh)
  assert.equal(getHomePath.mock.calls.length, 1)

  const fallback = createExecutableDiscoveryRuntime({
    isWindows: false,
    fileExists: () => false,
    findOnPath: () => null,
    getHomePath: () => 'C:/test-home',
    execFileSync: () => ''
  })

  assert.equal(fallback.resolveGitBinary(), 'git')
  assert.equal(fallback.resolveGhBinary(), 'gh')
})

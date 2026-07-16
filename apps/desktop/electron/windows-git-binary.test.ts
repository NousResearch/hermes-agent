// Real-behavior tests for the Windows git binary resolver.
//
// These exercise the actual resolveGitBinary() logic by injecting fake
// filesystem and PATH probes, so they run on any OS.

import assert from 'node:assert/strict'
import path from 'node:path'

import { beforeEach, test } from 'vitest'

import { resetGitBinaryCache, resolveGitBinary } from './windows-git-binary'

beforeEach(() => {
  resetGitBinaryCache()
})

function makeDeps(overrides: Partial<Parameters<typeof resolveGitBinary>[0]> = {}) {
  return {
    isWindows: false,
    fileExists: () => false,
    findOnPath: () => null,
    toShortPath: (p: string) => p,
    env: {},
    ...overrides
  }
}

test('non-Windows: returns the PATH result', () => {
  const deps = makeDeps({
    isWindows: false,
    findOnPath: (cmd: string) => (cmd === 'git' ? '/usr/local/bin/git' : null)
  })

  assert.equal(resolveGitBinary(deps), '/usr/local/bin/git')
})

test('non-Windows: falls back to the bare git command', () => {
  const deps = makeDeps({ isWindows: false })

  assert.equal(resolveGitBinary(deps), 'git')
})

test('Windows: returns the short path when a candidate exists and short-name conversion succeeds', () => {
  const env = { ProgramFiles: 'C:\\Program Files' }
  const longPath = path.win32.join(env.ProgramFiles, 'Git', 'cmd', 'git.exe')
  const shortPath = path.win32.join('C:\\PROGRA~1', 'Git', 'cmd', 'git.exe')

  const deps = makeDeps({
    isWindows: true,
    env,
    fileExists: (p: string) => p === longPath,
    toShortPath: (p: string) => (p === longPath ? shortPath : p)
  })

  assert.equal(resolveGitBinary(deps), shortPath)
})

test('Windows: keeps the long path when 8.3 short-name conversion returns the same path', () => {
  const env = { ProgramFiles: 'C:\\Program Files' }
  const longPath = path.win32.join(env.ProgramFiles, 'Git', 'cmd', 'git.exe')

  const deps = makeDeps({
    isWindows: true,
    env,
    fileExists: (p: string) => p === longPath,
    toShortPath: (p: string) => p
  })

  assert.equal(resolveGitBinary(deps), longPath)
})

test('Windows: falls back to PATH when no candidate exists', () => {
  const pathGit = path.win32.join('C:\\Users', 'Dev', 'bin', 'git.exe')
  const shortGit = path.win32.join('C:\\Users', 'DEV', 'bin', 'git.exe')

  const deps = makeDeps({
    isWindows: true,
    fileExists: () => false,
    findOnPath: (cmd: string) => (cmd === 'git' ? pathGit : null),
    toShortPath: (p: string) => (p === pathGit ? shortGit : p)
  })

  assert.equal(resolveGitBinary(deps), shortGit)
})

test('Windows: falls back to bare git command when nothing is found', () => {
  const deps = makeDeps({
    isWindows: true,
    fileExists: () => false,
    findOnPath: () => null
  })

  assert.equal(resolveGitBinary(deps), 'git')
})

test('caches the resolved binary so subsequent calls do not re-probe', () => {
  let probes = 0

  const deps = makeDeps({
    isWindows: false,
    findOnPath: (cmd: string) => {
      probes++

      return cmd === 'git' ? '/usr/bin/git' : null
    }
  })

  assert.equal(resolveGitBinary(deps), '/usr/bin/git')
  assert.equal(resolveGitBinary(deps), '/usr/bin/git')
  assert.equal(probes, 1)
})

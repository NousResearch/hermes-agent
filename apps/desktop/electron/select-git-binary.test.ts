import assert from 'node:assert/strict'

import { test } from 'vitest'

import { selectGitBinary } from './select-git-binary'

const WIN_ENV = {
  LOCALAPPDATA: 'C:\\Users\\test\\AppData\\Local',
  ProgramFiles: 'C:\\Program Files',
  'ProgramFiles(x86)': 'C:\\Program Files (x86)'
}

const HERMES_GIT = 'C:\\Users\\test\\AppData\\Local\\hermes\\git\\cmd\\git.exe'
const PROGRAM_FILES_GIT = 'C:\\Program Files\\Git\\cmd\\git.exe'

const yes = () => true
const no = () => false

test('a broken first candidate is skipped for one that runs', () => {
  // The real incident: install.ps1 left hermes\git\cmd\git.exe on disk
  // without its mingw64 payload, so it exists but cannot execute.
  const result = selectGitBinary({
    isWindows: true,
    env: WIN_ENV,
    fileExists: (p: string) => p === HERMES_GIT || p === PROGRAM_FILES_GIT,
    binaryRuns: (p: string) => p !== HERMES_GIT,
    findOnPath: () => null
  })

  assert.equal(result, PROGRAM_FILES_GIT)
})

test('the first candidate wins when it both exists and runs', () => {
  const result = selectGitBinary({
    isWindows: true,
    env: WIN_ENV,
    fileExists: (p: string) => p === HERMES_GIT || p === PROGRAM_FILES_GIT,
    binaryRuns: yes,
    findOnPath: () => null
  })

  assert.equal(result, HERMES_GIT)
})

test('when no candidate runs, fall back to the first that exists', () => {
  // Probing may be impossible (execution policy, AV interposing on spawn).
  // Falling back preserves the pre-probe behaviour rather than skipping a
  // git that would have worked.
  const result = selectGitBinary({
    isWindows: true,
    env: WIN_ENV,
    fileExists: (p: string) => p === HERMES_GIT || p === PROGRAM_FILES_GIT,
    binaryRuns: no,
    findOnPath: () => null
  })

  assert.equal(result, HERMES_GIT)
})

test('no candidate on disk falls through to PATH', () => {
  const result = selectGitBinary({
    isWindows: true,
    env: WIN_ENV,
    fileExists: no,
    binaryRuns: no,
    findOnPath: () => 'D:\\tools\\git.exe'
  })

  assert.equal(result, 'D:\\tools\\git.exe')
})

test('nothing found anywhere falls back to bare git', () => {
  const result = selectGitBinary({
    isWindows: true,
    env: WIN_ENV,
    fileExists: no,
    binaryRuns: no,
    findOnPath: () => null
  })

  assert.equal(result, 'git')
})

test('a runnable later candidate beats an existing-but-broken earlier one across all slots', () => {
  const userGit = 'C:\\Users\\test\\AppData\\Local\\Programs\\Git\\cmd\\git.exe'

  const result = selectGitBinary({
    isWindows: true,
    env: WIN_ENV,
    fileExists: yes,          // every candidate exists
    binaryRuns: (p: string) => p === userGit,   // only the last one runs
    findOnPath: () => null
  })

  assert.equal(result, userGit)
})

test('non-Windows uses PATH', () => {
  const result = selectGitBinary({
    isWindows: false,
    env: {},
    fileExists: no,
    binaryRuns: no,
    findOnPath: () => '/usr/bin/git'
  })

  assert.equal(result, '/usr/bin/git')
})

test('non-Windows with no git on PATH falls back to bare git', () => {
  const result = selectGitBinary({
    isWindows: false,
    env: {},
    fileExists: no,
    binaryRuns: no,
    findOnPath: () => null
  })

  assert.equal(result, 'git')
})

test('empty LOCALAPPDATA drops the hermes candidates without throwing', () => {
  const result = selectGitBinary({
    isWindows: true,
    env: { LOCALAPPDATA: '', ProgramFiles: 'C:\\Program Files' },
    fileExists: (p: string) => p === PROGRAM_FILES_GIT,
    binaryRuns: yes,
    findOnPath: () => null
  })

  assert.equal(result, PROGRAM_FILES_GIT)
})

test('the probe is not consulted for candidates that do not exist', () => {
  // Probing a non-existent path costs a failed spawn per candidate.
  const probed: string[] = []
  selectGitBinary({
    isWindows: true,
    env: WIN_ENV,
    fileExists: (p: string) => p === PROGRAM_FILES_GIT,
    binaryRuns: (p: string) => {
      probed.push(p)

      return true
    },
    findOnPath: () => null
  })

  assert.deepEqual(probed, [PROGRAM_FILES_GIT])
})

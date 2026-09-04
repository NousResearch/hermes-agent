import assert from 'node:assert/strict'

import { test } from 'vitest'

import { findGitBash } from './find-git-bash'

const yes = () => true
const no = () => false

test('HERMES_GIT_BASH_PATH override takes precedence', () => {
  const result = findGitBash({
    isWindows: true,
    env: { HERMES_GIT_BASH_PATH: 'D:\\CustomGit\\bin\\bash.exe' },
    fileExists: yes,
    findOnPath: () => null
  })

  assert.equal(result, 'D:\\CustomGit\\bin\\bash.exe')
})

test('HERMES_GIT_BASH_PATH invalid path falls through to candidates', () => {
  const env = {
    HERMES_GIT_BASH_PATH: 'X:\\Missing\\bash.exe',
    LOCALAPPDATA: 'C:\\Users\\test\\AppData\\Local',
    ProgramFiles: 'C:\\Program Files',
    'ProgramFiles(x86)': 'C:\\Program Files (x86)'
  }

  // robust to backslash count in constructed paths
  const fileExists = (p: string) => !p.includes('Missing') && p.includes('Git') && p.toLowerCase().endsWith('bash.exe')
  const result = findGitBash({ isWindows: true, env, fileExists, findOnPath: () => null })
  assert.ok(result && result.toLowerCase().includes('git') && result.toLowerCase().endsWith('bash.exe'))
})

test('HERMES_GIT_BASH_PATH empty string is ignored', () => {
  const result = findGitBash({
    isWindows: true,
    env: { HERMES_GIT_BASH_PATH: '', LOCALAPPDATA: '' },
    fileExists: no,
    findOnPath: () => 'C:\\msys64\\usr\\bin\\bash.exe'
  })

  assert.equal(result, 'C:\\msys64\\usr\\bin\\bash.exe')
})

test('non-Windows uses findOnPath', () => {
  const result = findGitBash({
    isWindows: false,
    env: {},
    fileExists: no,
    findOnPath: () => '/usr/bin/bash'
  })

  assert.equal(result, '/usr/bin/bash')
})

test('HERMES_GIT_EXE_PATH override for git.exe takes precedence', () => {
  const result = findGitBash({
    isWindows: true,
    env: { HERMES_GIT_EXE_PATH: 'D:\\CustomGit\\cmd\\git.exe' },
    fileExists: yes,
    findOnPath: () => null,
    executable: 'git.exe'
  })

  assert.equal(result, 'D:\\CustomGit\\cmd\\git.exe')
})

test('executable=git.exe uses cmd/ and bin/ under hermes for portable', () => {
  const env = {
    LOCALAPPDATA: 'C:\\Users\\test\\AppData\\Local',
    ProgramFiles: 'C:\\Program Files',
    'ProgramFiles(x86)': 'C:\\Program Files (x86)'
  }
  const fileExists = (p: string) => p.includes('hermes') && p.includes('git') && p.endsWith('git.exe')
  const result = findGitBash({ isWindows: true, env, fileExists, findOnPath: () => null, executable: 'git.exe' })
  assert.ok(result && result.includes('hermes') && result.endsWith('git.exe'))
})

test('executable=git.exe falls back to PATH git', () => {
  const result = findGitBash({
    isWindows: true,
    env: { LOCALAPPDATA: '' },
    fileExists: no,
    findOnPath: () => 'C:\\ProgramData\\Git\\cmd\\git.exe',
    executable: 'git.exe'
  })

  assert.equal(result, 'C:\\ProgramData\\Git\\cmd\\git.exe')
})

test('HERMES_GIT_BASH_PATH still only affects bash (not git)', () => {
  const env = {
    HERMES_GIT_BASH_PATH: 'D:\\BashOverride\\bin\\bash.exe',
    HERMES_GIT_EXE_PATH: 'D:\\GitOverride\\cmd\\git.exe',
    LOCALAPPDATA: ''
  }
  const fileExists = yes
  const bash = findGitBash({ isWindows: true, env, fileExists, findOnPath: () => null })
  const git = findGitBash({ isWindows: true, env, fileExists, findOnPath: () => null, executable: 'git.exe' })
  assert.equal(bash, 'D:\\BashOverride\\bin\\bash.exe')
  assert.equal(git, 'D:\\GitOverride\\cmd\\git.exe')
})

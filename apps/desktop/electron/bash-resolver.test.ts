import assert from 'node:assert/strict'
import { chmodSync, mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { resolveBashExecutable } from './bash-resolver'

test('resolveBashExecutable prefers the explicit override', () => {
  const override = '/custom/bash'

  const result = resolveBashExecutable({
    override,
    fileExists: candidate => candidate === override
  })

  assert.equal(result, override)
})

test('resolveBashExecutable ignores an override that does not exist', () => {
  const result = resolveBashExecutable({
    override: '/missing/bash',
    pathEnv: '',
    fileExists: () => false
  })

  assert.equal(result, null)
})

test('resolveBashExecutable walks PATH in order before known locations', () => {
  const dirA = mkdtempSync(path.join(os.tmpdir(), 'bash-resolver-a-'))
  const dirB = mkdtempSync(path.join(os.tmpdir(), 'bash-resolver-b-'))
  writeFileSync(path.join(dirB, 'bash'), '')

  const result = resolveBashExecutable({
    pathEnv: `${dirA}:${dirB}`,
    knownLocations: ['/bin/bash'],
    fileExists: candidate => candidate === path.join(dirB, 'bash')
  })

  assert.equal(result, path.join(dirB, 'bash'))
})

test('resolveBashExecutable rejects an executable directory named bash', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'bash-resolver-directory-'))
  const directory = path.join(root, 'bash')

  mkdirSync(directory)
  chmodSync(directory, 0o755)

  assert.equal(resolveBashExecutable({ pathEnv: root, knownLocations: [] }), null)
})

test('resolveBashExecutable falls back to a well-known location', () => {
  const result = resolveBashExecutable({
    pathEnv: '/empty',
    knownLocations: ['/usr/bin/bash'],
    fileExists: candidate => candidate === '/usr/bin/bash'
  })

  assert.equal(result, '/usr/bin/bash')
})

test('resolveBashExecutable returns null when no candidate exists', () => {
  const result = resolveBashExecutable({
    pathEnv: '/empty',
    fileExists: () => false
  })

  assert.equal(result, null)
})

test('resolveBashExecutable skips empty PATH entries', () => {
  const checked: string[] = []

  const result = resolveBashExecutable({
    pathEnv: '::',
    fileExists: candidate => {
      checked.push(candidate)

      return false
    }
  })

  assert.equal(result, null)
  assert.deepEqual(checked, ['/usr/bin/bash', '/bin/bash', '/usr/local/bin/bash'])
})

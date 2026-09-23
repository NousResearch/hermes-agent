import assert from 'node:assert/strict'
import fs from 'node:fs'

import { test } from 'vitest'

import {
  findSupportedPythonOnPath,
  isSupportedPythonVersion,
  readPythonVersion,
  selectSupportedPythonCandidate
} from './python-runtime'

const mainSource = fs.readFileSync(new URL('./main.ts', import.meta.url), 'utf8')

function extractMainFunction(startMarker: string, endMarker: string): string {
  const start = mainSource.indexOf(startMarker)
  const end = mainSource.indexOf(endMarker, start)

  assert.notEqual(start, -1, `missing start marker: ${startMarker}`)
  assert.notEqual(end, -1, `missing end marker: ${endMarker}`)

  return mainSource.slice(start, end)
}

test('skips an unsupported Python 3.9 candidate and selects a later supported interpreter', () => {
  const versions = new Map([
    ['/usr/bin/python3', '3.9.6'],
    ['/opt/hermes/venv/bin/python', '3.11.15']
  ])

  const selected = selectSupportedPythonCandidate(
    ['/usr/bin/python3', '/opt/hermes/venv/bin/python'],
    candidate => versions.get(candidate) ?? null
  )

  assert.deepEqual(selected, {
    path: '/opt/hermes/venv/bin/python',
    version: '3.11.15'
  })
})

test('matches the project Python range of 3.11 through 3.13', () => {
  assert.equal(isSupportedPythonVersion('3.10.14'), false)
  assert.equal(isSupportedPythonVersion('3.11.0'), true)
  assert.equal(isSupportedPythonVersion('3.12.9'), true)
  assert.equal(isSupportedPythonVersion('3.13.2'), true)
  assert.equal(isSupportedPythonVersion('3.14.0'), false)
  assert.equal(isSupportedPythonVersion('not-a-version'), false)
})

test('reads the interpreter version with a bounded no-shell probe', () => {
  let invocation = null

  const version = readPythonVersion('/opt/python', (command, args, options) => {
    invocation = { command, args, options }

    return '3.12.7\n'
  })

  assert.deepEqual(invocation, {
    command: '/opt/python',
    args: ['-c', 'import sys; print(".".join(map(str, sys.version_info[:3])))'],
    options: {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      timeout: 5_000,
      windowsHide: true
    }
  })
  assert.equal(version, '3.12.7')
})

test('treats an interpreter probe failure as unavailable', () => {
  assert.equal(
    readPythonVersion('/broken/python', () => {
      throw new Error('cannot execute')
    }),
    null
  )
})

test('continues from an unsupported python3 command to a supported python command on PATH', () => {
  const paths = new Map([
    ['python3', '/usr/bin/python3'],
    ['python', '/opt/hermes/venv/bin/python']
  ])

  const versions = new Map([
    ['/usr/bin/python3', '3.9.6'],
    ['/opt/hermes/venv/bin/python', '3.11.15']
  ])

  const selected = findSupportedPythonOnPath(
    ['python3', 'python'],
    command => paths.get(command) ?? null,
    candidate => versions.get(candidate) ?? null
  )

  assert.deepEqual(selected, {
    path: '/opt/hermes/venv/bin/python',
    version: '3.11.15'
  })
})

test('returns null when no PATH candidate is runnable and supported', () => {
  assert.equal(
    findSupportedPythonOnPath(
      ['python3', 'python'],
      command => `/usr/bin/${command}`,
      candidate => (candidate.endsWith('python3') ? '3.9.6' : null)
    ),
    null
  )
})

test('applies the supported-version gate only to the source resolver fallback', () => {
  const sourceResolver = extractMainFunction('function findPythonForRoot(root) {', '\nfunction findSystemPython() {')
  const sharedSystemResolver = extractMainFunction('function findSystemPython() {', '\n// findGitBash')

  assert.match(sourceResolver, /if \(!IS_WINDOWS\)[\s\S]*findSupportedPythonOnPath/)
  assert.doesNotMatch(sharedSystemResolver, /findSupportedPythonOnPath/)
  assert.match(sharedSystemResolver, /for \(const command of \['python3', 'python'\]\)/)
})

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { applyWindowsMsysBashEnvDefaults } from './windows-msys-bash-env'

test('applyWindowsMsysBashEnvDefaults sets both MSYS opt-outs on Windows when unset', () => {
  const env: NodeJS.ProcessEnv = { FOO: 'bar' }
  const result = applyWindowsMsysBashEnvDefaults(env, true)

  assert.equal(result, env)
  assert.equal(env.MSYS_NO_PATHCONV, '1')
  assert.equal(env.MSYS2_ARG_CONV_EXCL, '*')
  assert.equal(env.FOO, 'bar')
})

test('applyWindowsMsysBashEnvDefaults preserves an existing MSYS_NO_PATHCONV on Windows', () => {
  const env: NodeJS.ProcessEnv = { MSYS_NO_PATHCONV: '0' }
  applyWindowsMsysBashEnvDefaults(env, true)

  assert.equal(env.MSYS_NO_PATHCONV, '0')
  assert.equal(env.MSYS2_ARG_CONV_EXCL, '*')
})

test('applyWindowsMsysBashEnvDefaults preserves an existing MSYS2_ARG_CONV_EXCL on Windows', () => {
  const env: NodeJS.ProcessEnv = { MSYS2_ARG_CONV_EXCL: '/custom' }
  applyWindowsMsysBashEnvDefaults(env, true)

  assert.equal(env.MSYS_NO_PATHCONV, '1')
  assert.equal(env.MSYS2_ARG_CONV_EXCL, '/custom')
})

test('applyWindowsMsysBashEnvDefaults preserves both existing MSYS opt-outs on Windows', () => {
  const env: NodeJS.ProcessEnv = { MSYS_NO_PATHCONV: '', MSYS2_ARG_CONV_EXCL: '/FO' }
  applyWindowsMsysBashEnvDefaults(env, true)

  assert.equal(env.MSYS_NO_PATHCONV, '')
  assert.equal(env.MSYS2_ARG_CONV_EXCL, '/FO')
})

test('applyWindowsMsysBashEnvDefaults is a no-op off Windows', () => {
  const env: NodeJS.ProcessEnv = { FOO: 'bar' }
  const result = applyWindowsMsysBashEnvDefaults(env, false)

  assert.equal(result, env)
  assert.deepEqual(env, { FOO: 'bar' })
})

test('applyWindowsMsysBashEnvDefaults leaves existing MSYS opt-outs unchanged off Windows', () => {
  const env: NodeJS.ProcessEnv = { MSYS_NO_PATHCONV: '0', MSYS2_ARG_CONV_EXCL: '/custom' }
  applyWindowsMsysBashEnvDefaults(env, false)

  assert.deepEqual(env, { MSYS_NO_PATHCONV: '0', MSYS2_ARG_CONV_EXCL: '/custom' })
})

test('applyWindowsMsysBashEnvDefaults defaults isWindows from process.platform when omitted', () => {
  const env: NodeJS.ProcessEnv = {}
  applyWindowsMsysBashEnvDefaults(env)

  if (process.platform === 'win32') {
    assert.equal(env.MSYS_NO_PATHCONV, '1')
    assert.equal(env.MSYS2_ARG_CONV_EXCL, '*')
  } else {
    assert.equal(Object.hasOwn(env, 'MSYS_NO_PATHCONV'), false)
    assert.equal(Object.hasOwn(env, 'MSYS2_ARG_CONV_EXCL'), false)
  }
})

test('applyWindowsMsysBashEnvDefaults does not throw on a shallow clone of process.env', () => {
  assert.doesNotThrow(() => {
    applyWindowsMsysBashEnvDefaults({ ...process.env }, true)
  })
})

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { buildTerminalShellEnv } from './terminal-shell-env'

test('terminal shell environment preserves the inherited locale', () => {
  const env = buildTerminalShellEnv({
    appVersion: '0.17.0',
    currentEnv: { LANG: 'en_US.UTF-8' }
  })

  assert.equal(env.LANG, 'en_US.UTF-8')
  assert.equal(env.LC_CTYPE, undefined)
})

test('terminal shell environment keeps an explicitly inherited LC_CTYPE', () => {
  const env = buildTerminalShellEnv({
    appVersion: '0.17.0',
    currentEnv: { LC_CTYPE: 'C.UTF-8' }
  })

  assert.equal(env.LC_CTYPE, 'C.UTF-8')
})

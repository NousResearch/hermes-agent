import { describe, expect, it } from 'vitest'

import { applyWindowsMsysBashEnvDefaults } from './terminal-msys-env'

describe('applyWindowsMsysBashEnvDefaults', () => {
  it('sets both MSYS guards for native Windows shells', () => {
    const env: Record<string, string | undefined> = {}

    applyWindowsMsysBashEnvDefaults(env, true)

    expect(env).toEqual({ MSYS2_ARG_CONV_EXCL: '*', MSYS_NO_PATHCONV: '1' })
  })

  it('preserves explicit values, including an empty opt-out', () => {
    const env: Record<string, string | undefined> = {
      MSYS2_ARG_CONV_EXCL: '/custom',
      MSYS_NO_PATHCONV: ''
    }

    applyWindowsMsysBashEnvDefaults(env, true)

    expect(env).toEqual({ MSYS2_ARG_CONV_EXCL: '/custom', MSYS_NO_PATHCONV: '' })
  })

  it('does not change POSIX shell environments', () => {
    const env: Record<string, string | undefined> = {}

    applyWindowsMsysBashEnvDefaults(env, false)

    expect(env).toEqual({})
  })
})

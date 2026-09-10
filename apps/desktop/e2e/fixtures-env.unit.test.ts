import { describe, expect, it } from 'vitest'

import { buildAppEnvFromParent } from './fixtures-env'

const ISOLATED_DESKTOP_ENV_KEYS = [
  'HERMES_DESKTOP_DEV_SERVER',
  'HERMES_DESKTOP_REMOTE_URL',
  'HERMES_DESKTOP_REMOTE_TOKEN',
  'HERMES_DESKTOP_HERMES',
  'HERMES_DESKTOP_PYTHON',
  'HERMES_DESKTOP_BOOT_FAKE',
  'HERMES_DESKTOP_BOOT_FAKE_ERROR',
  'HERMES_DESKTOP_IS_PACKAGED',
  'HERMES_DESKTOP_FORCE_DEV',
  'HERMES_E2E_PYTHON',
] as const

describe('buildAppEnvFromParent', () => {
  it('removes every inherited desktop launch override without mutating the parent', () => {
    const parentEnv: Record<string, string> = { SYNTHETIC_PARENT_MARKER: 'preserved' }

    for (const key of ISOLATED_DESKTOP_ENV_KEYS) {
      parentEnv[key] = `synthetic-inherited-${key.toLowerCase()}`
    }

    const parentSnapshot = { ...parentEnv }

    const sandbox = {
      hermesHome: 'synthetic-hermes-home',
      userDataDir: 'synthetic-user-data',
    }

    const childEnv = buildAppEnvFromParent(parentEnv, sandbox, 'synthetic-repo-root')

    for (const key of ISOLATED_DESKTOP_ENV_KEYS) {
      if (key === 'HERMES_DESKTOP_FORCE_DEV') {
        expect(childEnv[key]).toBe('1')
      } else {
        expect(childEnv).not.toHaveProperty(key)
      }
    }

    expect(parentEnv).toEqual(parentSnapshot)
  })

  it('applies explicit desktop launch overrides after inherited isolation', () => {
    const childEnv = buildAppEnvFromParent(
      {
        HERMES_DESKTOP_BOOT_FAKE: 'synthetic-inherited-fake',
        HERMES_DESKTOP_REMOTE_URL: 'https://synthetic-inherited.invalid',
      },
      {
        hermesHome: 'synthetic-hermes-home',
        userDataDir: 'synthetic-user-data',
      },
      'synthetic-repo-root',
      {
        HERMES_DESKTOP_BOOT_FAKE: '1',
        HERMES_DESKTOP_REMOTE_URL: 'https://synthetic-explicit.invalid',
      },
    )

    expect(childEnv.HERMES_DESKTOP_BOOT_FAKE).toBe('1')
    expect(childEnv.HERMES_DESKTOP_REMOTE_URL).toBe('https://synthetic-explicit.invalid')
  })

  it('preserves credential stripping and sandbox path precedence', () => {
    const childEnv = buildAppEnvFromParent(
      {
        SYNTHETIC_PARENT_MARKER: 'preserved',
        SYNTHETIC_API_KEY: 'synthetic-credential',
        HERMES_HOME: 'synthetic-inherited-home',
        HERMES_DESKTOP_USER_DATA_DIR: 'synthetic-inherited-user-data',
        HERMES_DESKTOP_HERMES_ROOT: 'synthetic-inherited-repo-root',
      },
      {
        hermesHome: 'synthetic-sandbox-home',
        userDataDir: 'synthetic-sandbox-user-data',
      },
      'synthetic-repo-root',
    )

    expect(childEnv).toMatchObject({
      SYNTHETIC_PARENT_MARKER: 'preserved',
      HERMES_HOME: 'synthetic-sandbox-home',
      HERMES_DESKTOP_USER_DATA_DIR: 'synthetic-sandbox-user-data',
      HERMES_DESKTOP_HERMES_ROOT: 'synthetic-repo-root',
    })
    expect(childEnv).not.toHaveProperty('SYNTHETIC_API_KEY')
  })
})

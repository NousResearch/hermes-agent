import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  buildDesktopServeChildEnv,
  buildDesktopTerminalEnv,
  isHermesCredentialEnvVar,
  scrubDesktopChildEnv
} from './scrub-child-env'

test('matches Hermes-owned credentials case-insensitively', () => {
  assert.equal(isHermesCredentialEnvVar('OPENROUTER_API_KEY'), true)
  assert.equal(isHermesCredentialEnvVar('openrouter_api_key'), true)
  assert.equal(isHermesCredentialEnvVar('FAL_KEY'), true)
  assert.equal(isHermesCredentialEnvVar('CUSTOM_API_KEY'), true)
  assert.equal(isHermesCredentialEnvVar('GATEWAY_RELAY_DELIVERY_KEY'), true)
  assert.equal(isHermesCredentialEnvVar('AUXILIARY_VISION_BASE_URL'), true)
  assert.equal(isHermesCredentialEnvVar('HERMES_CUSTOM_LOCAL_API_KEY'), true)
  assert.equal(isHermesCredentialEnvVar('HERMES_DESKTOP_REMOTE_TOKEN'), true)
})

test('closes Apptainer and Singularity effective-name tunnelling', () => {
  assert.equal(isHermesCredentialEnvVar('APPTAINERENV_OPENAI_API_KEY'), true)
  assert.equal(isHermesCredentialEnvVar('singularityenv_FAL_KEY'), true)
  assert.equal(isHermesCredentialEnvVar('APPTAINERENV_SINGULARITYENV_OPENROUTER_API_KEY'), true)
})

test('preserves non-secret endpoints and operator-owned credentials', () => {
  for (const name of [
    'OPENROUTER_BASE_URL',
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'AWS_SESSION_TOKEN',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'NPM_TOKEN'
  ]) {
    assert.equal(isHermesCredentialEnvVar(name), false, name)
  }
})

test('scrubs every source map without deleting empty non-secret values', () => {
  const scrubbed = scrubDesktopChildEnv(
    {
      PATH: '/usr/bin',
      EMPTY: '',
      OPENAI_API_KEY: 'source-secret',
      NPM_TOKEN: 'operator-secret'
    },
    {
      OPENAI_API_KEY: 'override-secret',
      openrouter_api_key: 'mixed-case-secret',
      HERMES_HOME: '/home/u/.hermes'
    }
  )

  assert.equal(scrubbed.PATH, '/usr/bin')
  assert.equal(scrubbed.EMPTY, '')
  assert.equal(scrubbed.NPM_TOKEN, 'operator-secret')
  assert.equal(scrubbed.HERMES_HOME, '/home/u/.hermes')
  assert.equal(scrubbed.OPENAI_API_KEY, undefined)
  assert.equal(scrubbed.openrouter_api_key, undefined)
})

test('serve env admits only its freshly minted dashboard token', () => {
  const env = buildDesktopServeChildEnv({
    source: {
      PATH: '/usr/bin',
      HERMES_DASHBOARD_SESSION_TOKEN: 'inherited-token',
      FAL_KEY: 'inherited-fal'
    },
    backendEnv: {
      PYTHONPATH: '/app',
      OPENAI_API_KEY: 'backend-override-secret'
    },
    hermesHome: '/home/u/.hermes',
    terminalCwd: '/work',
    dashboardSessionToken: 'minted-token',
    parentIdentityEnv: {
      HERMES_DESKTOP_PARENT_PID: '42',
      GATEWAY_RELAY_SECRET: 'relay-secret'
    },
    webDist: '/app/web_dist',
    readyFile: '/tmp/ready.json'
  })

  assert.equal(env.PATH, '/usr/bin')
  assert.equal(env.PYTHONPATH, '/app')
  assert.equal(env.HERMES_HOME, '/home/u/.hermes')
  assert.equal(env.TERMINAL_CWD, '/work')
  assert.equal(env.HERMES_DASHBOARD_SESSION_TOKEN, 'minted-token')
  assert.equal(env.HERMES_DESKTOP, '1')
  assert.equal(env.HERMES_DESKTOP_PARENT_PID, '42')
  assert.equal(env.HERMES_WEB_DIST, '/app/web_dist')
  assert.equal(env.HERMES_DESKTOP_READY_FILE, '/tmp/ready.json')
  assert.equal(env.FAL_KEY, undefined)
  assert.equal(env.OPENAI_API_KEY, undefined)
  assert.equal(env.GATEWAY_RELAY_SECRET, undefined)
})

test('terminal env scrubs Hermes credentials and retains the operator shell contract', () => {
  const env = buildDesktopTerminalEnv(
    {
      PATH: '/usr/bin',
      OPENROUTER_API_KEY: 'provider-secret',
      NPM_TOKEN: 'operator-secret',
      AWS_ACCESS_KEY_ID: 'operator-aws',
      npm_config_prefix: '/npm',
      NO_COLOR: '1',
      LC_CTYPE: 'ja_JP.UTF-8'
    },
    '0.17.0'
  )

  assert.equal(env.OPENROUTER_API_KEY, undefined)
  assert.equal(env.NPM_TOKEN, 'operator-secret')
  assert.equal(env.AWS_ACCESS_KEY_ID, 'operator-aws')
  assert.equal(env.npm_config_prefix, undefined)
  assert.equal(env.NO_COLOR, undefined)
  assert.equal(env.LC_CTYPE, 'ja_JP.UTF-8')
  assert.equal(env.COLORTERM, 'truecolor')
  assert.equal(env.TERM, 'xterm-256color')
  assert.equal(env.TERM_PROGRAM, 'Hermes')
  assert.equal(env.TERM_PROGRAM_VERSION, '0.17.0')
  assert.equal(env.HERMES_DESKTOP_TERMINAL, '1')
})

import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import {
  parsePreUpdateBackupPolicy,
  resolvePreUpdateBackupPolicy,
  type RunPolicyProcess,
  type RunPolicyProcessOptions
} from './pre-update-backup-policy'

const VALID = JSON.stringify({
  backup_keep: 3,
  mode: 'full',
  quick_keep: 1,
  quick_max_file_size: 1024
})

describe('parsePreUpdateBackupPolicy', () => {
  test('accepts the strict producer schema', () => {
    assert.deepEqual(parsePreUpdateBackupPolicy(`${VALID}\n`), {
      backupKeep: 3,
      mode: 'full',
      quickKeep: 1,
      quickMaxFileSize: 1024
    })
  })

  test.each(['', 'not-json\n', `${VALID}\nnoise\n`, `noise\n${VALID}\n`, '{}\n'])(
    'rejects empty, noisy, or malformed output',
    stdout => {
      assert.throws(() => parsePreUpdateBackupPolicy(stdout), /policy/)
    }
  )

  test.each([
    { ...JSON.parse(VALID), mode: 'maybe' },
    { ...JSON.parse(VALID), backup_keep: 0 },
    { ...JSON.parse(VALID), quick_keep: 0 },
    { ...JSON.parse(VALID), quick_max_file_size: 0 },
    { ...JSON.parse(VALID), extra: true }
  ])('rejects values the strict Python producer must never emit', payload => {
    assert.throws(() => parsePreUpdateBackupPolicy(`${JSON.stringify(payload)}\n`), /policy/)
  })
})

describe('resolvePreUpdateBackupPolicy', () => {
  test('uses one bounded no-shell Python invocation with explicit cwd and HERMES_HOME', async () => {
    const calls: Array<{ args: string[]; command: string; options: RunPolicyProcessOptions }> = []

    const runProcess: RunPolicyProcess = async (command, args, options) => {
      calls.push({ args, command, options })

      return { stderr: '', stdout: `${VALID}\n` }
    }

    const policy = await resolvePreUpdateBackupPolicy(
      {
        hermesHome: 'C:\\Hermes Home',
        pythonPath: 'C:\\Python\\python.exe',
        updateRoot: 'C:\\Hermes Source'
      },
      runProcess
    )

    assert.equal(policy.mode, 'full')
    assert.equal(calls.length, 1)
    assert.equal(calls[0].command, 'C:\\Python\\python.exe')
    assert.deepEqual(calls[0].args, ['-m', 'hermes_cli.update_preflight_policy'])
    assert.equal(calls[0].options.cwd, 'C:\\Hermes Source')
    assert.equal(calls[0].options.shell, false)
    assert.equal(calls[0].options.timeout, 15_000)
    assert.equal((calls[0].options.env as NodeJS.ProcessEnv).HERMES_HOME, 'C:\\Hermes Home')
    assert.equal((calls[0].options.env as NodeJS.ProcessEnv).PYTHONUTF8, '1')
  })

  test.each([
    Object.assign(new Error('exit 2'), { code: 2 }),
    Object.assign(new Error('timed out'), { code: 'ETIMEDOUT' })
  ])('fails closed when the producer process rejects', async error => {
    const runProcess: RunPolicyProcess = async () => {
      throw error
    }

    await assert.rejects(
      resolvePreUpdateBackupPolicy({ hermesHome: 'home', pythonPath: 'python', updateRoot: 'root' }, runProcess),
      /could not resolve pre-update backup policy/
    )
  })

  test('fails closed when a successful process emits malformed JSON', async () => {
    const runProcess: RunPolicyProcess = async () => ({ stderr: '', stdout: '{bad}\n' })

    await assert.rejects(
      resolvePreUpdateBackupPolicy({ hermesHome: 'home', pythonPath: 'python', updateRoot: 'root' }, runProcess),
      /policy/
    )
  })

  test('fails closed on noisy stderr even when stdout is valid', async () => {
    const runProcess: RunPolicyProcess = async () => ({ stderr: 'unexpected warning\n', stdout: VALID })

    await assert.rejects(
      resolvePreUpdateBackupPolicy({ hermesHome: 'home', pythonPath: 'python', updateRoot: 'root' }, runProcess),
      /unexpected stderr/
    )
  })
})

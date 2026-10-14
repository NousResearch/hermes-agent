import { mkdtempSync, mkdirSync, existsSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { expect, test, vi } from 'vitest'
import notarize, { runCommand } from '../apps/desktop/scripts/notarize.mjs'

const submissionId = '00000000-0000-4000-8000-000000000001'
const missingTicket = () => Object.assign(new Error('CloudKit query for Hermes.app failed due to "Record not found".\nThe staple and validate action failed! Error 65.'), { code: 65 })

function fixture() {
  const root = mkdtempSync(path.join(os.tmpdir(), 'hermes-notary-test-'))
  mkdirSync(path.join(root, 'Hermes.app'))
  const key = path.join(root, 'api-key.p8')
  writeFileSync(key, 'test credential path only')
  return {
    root,
    context: { electronPlatformName: 'darwin', appOutDir: root, packager: { appInfo: { productFilename: 'Hermes' } } },
    environments: [
      { APPLE_NOTARY_PROFILE: 'test-profile' },
      { APPLE_API_KEY: key, APPLE_API_KEY_ID: 'test-key', APPLE_API_ISSUER: 'test-issuer' }
    ],
    cleanup: () => rmSync(root, { recursive: true, force: true })
  }
}

test('accepted submissions retry only ticket propagation, without resubmitting or rebuilding', async () => {
  const f = fixture()
  try {
    for (const env of f.environments) {
      const calls = []
      let attempts = 0
      const run = vi.fn(async (command, args) => {
        calls.push([command, ...args])
        if (command === 'ditto') writeFileSync(args.at(-1), 'archive fixture')
        if (args[0] === 'notarytool') return { stdout: JSON.stringify({ id: submissionId, status: 'Accepted' }) }
        if (args[0] === 'stapler' && ++attempts <= 2) {
          // Exercise exit-code and output capture with a real child process.
          return runCommand(process.execPath, ['-e',
            'process.stdout.write(process.argv[1]); process.exitCode = 65', missingTicket().message])
        }
        return { stdout: '', stderr: '' }
      })
      const sleep = vi.fn(async () => {})
      const log = vi.fn()
      await notarize(f.context, { run, sleep, log, env })
      expect(calls.filter(c => c[0] === 'ditto')).toHaveLength(1)
      const submissions = calls.filter(c => c[1] === 'notarytool' && c[2] === 'submit')
      expect(submissions).toHaveLength(1)
      expect(calls.filter(c => c[1] === 'notarytool' && c[2] === 'wait')).toHaveLength(0)
      expect(submissions[0]).toEqual(expect.arrayContaining(['--wait', '--output-format', 'json']))
      expect(submissions[0]).toEqual(expect.arrayContaining(env.APPLE_NOTARY_PROFILE
        ? ['--keychain-profile', env.APPLE_NOTARY_PROFILE]
        : ['--key', env.APPLE_API_KEY, '--key-id', env.APPLE_API_KEY_ID, '--issuer', env.APPLE_API_ISSUER]))
      expect(attempts).toBe(3)
      expect(sleep.mock.calls.map(([ms]) => ms)).toEqual([15000, 30000])
      expect(log).toHaveBeenCalledWith(expect.stringContaining(`${submissionId}: Accepted`))
      expect(existsSync(path.join(f.root, 'Hermes.zip'))).toBe(false)
    }
  } finally {
    f.cleanup()
  }
})

test('HTTP timeouts resume the same submission with notarytool wait and a shrinking budget', async () => {
  const f = fixture()
  // Replay the status-request failure from the macOS release log.
  const timeoutOutput = `Error: HTTPError(statusCode: nil, error: Error Domain=NSURLErrorDomain Code=-1001 "The request timed out." UserInfo={NSErrorFailingURLStringKey=https://appstoreconnect.apple.com/notary/v2/submissions/${submissionId}?, _kCFStreamErrorDomainKey=1})`
  try {
    for (const env of f.environments) {
      const calls = []
      let elapsed = 0
      let waits = 0
      const log = vi.fn()
      const run = async (command, args, options = {}) => {
        calls.push({ command, args, options })
        if (command === 'ditto') writeFileSync(args.at(-1), 'archive fixture')
        if (args[0] === 'notarytool' && args[1] === 'submit') {
          elapsed += 10 * 60 * 1000
          return runCommand(process.execPath, ['-e',
            'process.stderr.write(process.argv[1]); process.exitCode = 1', timeoutOutput])
        }
        if (args[0] === 'notarytool' && args[1] === 'wait') {
          if (++waits === 1) {
            elapsed += 20 * 60 * 1000
            // The resumed command already owns the ID even if this error omits it.
            throw new Error('Error Domain=NSURLErrorDomain Code=-1001 "The request timed out."')
          }
          return { stdout: JSON.stringify({ id: submissionId, status: 'Accepted' }) }
        }
        return { stdout: '', stderr: '' }
      }
      await notarize(f.context, {
        run, env, log, now: () => elapsed,
        sleep: async ms => { elapsed += ms },
      })
      expect(calls.filter(c => c.command === 'ditto')).toHaveLength(1)
      const notary = calls.filter(c => c.args[0] === 'notarytool')
      expect(notary.map(c => c.args[1])).toEqual(['submit', 'wait', 'wait'])
      expect(notary[0].args).toContain('--wait')
      const auth = env.APPLE_NOTARY_PROFILE
        ? ['--keychain-profile', env.APPLE_NOTARY_PROFILE]
        : ['--key', env.APPLE_API_KEY, '--key-id', env.APPLE_API_KEY_ID, '--issuer', env.APPLE_API_ISSUER]
      for (const call of notary) {
        expect(call.args).toEqual(expect.arrayContaining([...auth, '--output-format', 'json', '--timeout']))
        const waitTimeout = call.args[call.args.indexOf('--timeout') + 1]
        expect(waitTimeout).toMatch(/^[1-9][0-9]*s$/)
        expect(call.options.timeout).toBeGreaterThan(Number.parseInt(waitTimeout, 10) * 1000)
      }
      for (let i = 1; i < notary.length; i++) {
        expect(notary[i].args[2]).toBe(submissionId)
        const timeoutSeconds = call => Number.parseInt(call.args[call.args.indexOf('--timeout') + 1], 10)
        expect(timeoutSeconds(notary[i])).toBeLessThan(timeoutSeconds(notary[i - 1]))
        expect(notary[i].options.timeout).toBeLessThan(notary[i - 1].options.timeout)
      }
      expect(calls.filter(c => c.args[0] === 'stapler')).toHaveLength(1)
      expect(log).toHaveBeenCalledWith(expect.stringContaining(submissionId))
      expect(existsSync(path.join(f.root, 'Hermes.zip'))).toBe(false)
    }
  } finally { f.cleanup() }
})

test('wait recovery requires a known submission and stops at failures or its limits', async () => {
  const f = fixture()
  const timeout = 'Error Domain=NSURLErrorDomain Code=-1001 "The request timed out."'
  const submissionUrl = `https://appstoreconnect.apple.com/notary/v2/submissions/${submissionId}?`
  const scenarios = ['missing-id', 'malformed-id', 'auth', 'exhausted', 'budget', 'sleep-budget', 'invalid', 'wrong-id']
  try {
    for (const scenario of scenarios) {
      const calls = []
      let elapsed = 0
      const run = async (command, args, options = {}) => {
        calls.push([command, ...args])
        if (command === 'ditto') writeFileSync(args.at(-1), 'archive fixture')
        if (args[0] === 'notarytool' && args[1] === 'log') return { stdout: 'signature rejected' }
        if (args[0] === 'notarytool') {
          if (scenario === 'auth') throw new Error(`authentication failed for ${submissionUrl}`)
          if (scenario === 'missing-id') throw new Error(`${timeout} LocalDataTask <${submissionId}>`)
          if (scenario === 'malformed-id') throw new Error(`${timeout} ${submissionUrl.replace('?', 'garbage?')}`)
          if (scenario === 'budget') elapsed += options.timeout
          if (args[1] === 'wait' && scenario === 'invalid') {
            throw Object.assign(new Error('notarization rejected'), {
              stdout: JSON.stringify({ id: submissionId, status: 'Invalid' }),
            })
          }
          if (args[1] === 'wait' && scenario === 'wrong-id') {
            return { stdout: JSON.stringify({ id: '00000000-0000-4000-8000-000000000002', status: 'Accepted' }) }
          }
          throw new Error(`${timeout} ${submissionUrl}`)
        }
        return { stdout: '' }
      }
      const sleep = vi.fn(async ms => { elapsed += scenario === 'sleep-budget' ? 24 * 60 * 60 * 1000 : ms })
      const failure = await notarize(f.context, {
        run, env: f.environments[0], sleep, log: vi.fn(), now: () => elapsed,
      }).catch(error => error)
      expect(failure).toBeInstanceOf(Error)
      expect(calls.filter(c => c[2] === 'submit')).toHaveLength(1)
      expect(calls.filter(c => c[1] === 'stapler')).toHaveLength(0)
      const waits = calls.filter(c => c[2] === 'wait')
      if (scenario === 'exhausted') {
        expect(waits).toHaveLength(3)
        expect(sleep).toHaveBeenCalledTimes(waits.length)
      } else if (scenario === 'invalid' || scenario === 'wrong-id') {
        expect(waits).toHaveLength(1)
        expect(failure.message).toContain(scenario === 'invalid' ? 'signature rejected' : 'while waiting for')
      } else {
        expect(waits).toHaveLength(0)
        expect(sleep).toHaveBeenCalledTimes(scenario === 'sleep-budget' ? 1 : 0)
      }
      expect(calls.filter(c => c[2] === 'log')).toHaveLength(scenario === 'invalid' ? 1 : 0)
      expect(existsSync(path.join(f.root, 'Hermes.zip'))).toBe(false)
    }
  } finally { f.cleanup() }
})

test('rejections, unknown failures and exhausted propagation retries remain build failures', async () => {
  const f = fixture()
  try {
    for (const scenario of ['invalid-zero', 'invalid-nonzero', 'malformed', 'auth', 'permanent-staple', 'exhausted']) {
      const calls = []
      const run = vi.fn(async (command, args) => {
        calls.push([command, ...args])
        if (command === 'ditto') writeFileSync(args.at(-1), 'archive fixture')
        if (args[0] === 'notarytool' && args[1] === 'log') return { stdout: '{"issues":[{"message":"signature rejected"}]}' }
        if (args[0] === 'notarytool') {
          if (scenario === 'auth') throw new Error('authentication failed')
          if (scenario === 'malformed') return { stdout: 'not JSON' }
          const stdout = JSON.stringify({ id: submissionId, status: scenario.startsWith('invalid') ? 'Invalid' : 'Accepted' })
          if (scenario === 'invalid-nonzero') throw Object.assign(new Error('submit failed'), { code: 1, stdout })
          return { stdout }
        }
        if (args[0] === 'stapler') {
          if (scenario === 'permanent-staple') throw Object.assign(new Error('invalid bundle format'), { code: 65 })
          throw missingTicket()
        }
        return { stdout: '', stderr: '' }
      })
      const sleep = vi.fn(async () => {})
      const failure = await notarize(f.context, { run, sleep, log: vi.fn(), env: f.environments[0] }).catch(e => e)
      expect(failure).toBeInstanceOf(Error)
      expect(calls.filter(c => c[2] === 'submit')).toHaveLength(1)
      const staples = calls.filter(c => c[1] === 'stapler')
      if (scenario.startsWith('invalid')) {
        expect(failure.message).toContain('Invalid')
        expect(failure.message).toContain('signature rejected')
        expect(calls.filter(c => c[2] === 'log')).toHaveLength(1)
        expect(staples).toHaveLength(0)
      } else if (scenario === 'exhausted') {
        expect(failure.message).toContain('Record not found')
        expect(staples.length).toBe(sleep.mock.calls.length + 1)
        expect(staples.length).toBeGreaterThan(1)
        expect(staples.length).toBeLessThanOrEqual(6)
      } else {
        expect(staples).toHaveLength(scenario === 'permanent-staple' ? 1 : 0)
      }
      if (scenario !== 'exhausted') expect(sleep).not.toHaveBeenCalled()
      expect(existsSync(path.join(f.root, 'Hermes.zip'))).toBe(false)
    }
  } finally {
    f.cleanup()
  }
})

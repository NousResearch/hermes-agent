/**
 * The Desktop SSH token upload must hand the login shell a SINGLE-LINE command (#154114).
 *
 * sshd parses the command string with the remote account's login shell. POSIX shells (and fish)
 * accept a newline inside `sh -c '<payload>'`; csh/tcsh does not, so a multi-line payload killed
 * the launch on a tcsh remote after every probe had passed. Every other launch payload now rides
 * the `sh -s` stdin carrier (ssh-connection.test.ts), but the token upload is the one payload that
 * keeps stdin for the secret itself, so it must carry its program on the argv line instead.
 */
import assert from 'node:assert/strict'

import { test } from 'vitest'

import { spawnRemoteDashboard } from './remote-lifecycle'
import { decodeRemoteProgram } from './ssh-connection'

const OWNERSHIP_ID = '0123456789abcdef0123456789abcdef'

test('the token upload command is one argv line and keeps the token on stdin', async () => {
  const calls: { command: string; stdinData?: string }[] = []

  const ssh = {
    async exec(command: string, opts?: { stdinData?: string }) {
      calls.push({ command, stdinData: opts?.stdinData })

      if (/grep -q ssh-session-token-file/.test(command)) {
        return 'YES\n'
      }

      if (/setsid|nohup/.test(command)) {
        return '4242\n'
      }

      return ''
    }
  }

  const token = 'secret_token_val'

  const { pid } = await spawnRemoteDashboard(ssh as any, {
    hermesPath: '/x/hermes',
    profile: '',
    token,
    ownershipId: OWNERSHIP_ID
  })

  assert.equal(pid, 4242)

  const upload = calls.find(call => call.stdinData === token)
  assert.ok(upload, 'the token must be uploaded on stdin')

  assert.ok(!/[\r\n]/.test(upload.command), `a newline is a csh/tcsh parse error: ${upload.command.slice(0, 80)}`)
  assert.match(upload.command, /^python3 -c '/, 'the program rides the argv line, not stdin')

  // The security intent of the upload is unchanged: it only travels encoded, never weaker.
  const program = decodeRemoteProgram(upload.command)
  assert.match(program, /O_EXCL/)
  assert.match(program, /O_NOFOLLOW/)
  assert.match(program, /dir_fd=dd/)

  for (const call of calls) {
    assert.ok(!call.command.includes(token), `token leaked into argv: ${call.command}`)
  }
})

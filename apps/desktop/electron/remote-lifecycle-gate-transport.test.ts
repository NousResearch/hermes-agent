/**
 * How the relaunch marker gate travels to the remote.
 *
 * `ssh.exec` rides the POSIX ControlMaster mux socket, which hands the client's
 * stdio descriptors over with sendmsg(SCM_RIGHTS) after writing the session
 * request. A request that still fills the socket buffer (8 KB for a unix-stream
 * socket on macOS) leaves no room for that descriptor message, so the pass
 * fails with EMSGSIZE and ssh exits 255 — which the fail-closed gate reports as
 * "could not prove the install is clear" on every reconnect.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { assertRemoteInstallUpdateClear } from './remote-lifecycle'
import { REMOTE_MARKER_GATE_PY } from './remote-update-marker-programs'

test('POSIX relaunch gate reaches a verdict over a transport that refuses a large command', async () => {
  const calls: { command: string; stdinData?: string }[] = []

  const ssh = {
    async exec(command: string, { stdinData }: any = {}) {
      calls.push({ command, stdinData })

      if (command.length > 4096) {
        throw Object.assign(new Error('mm_send_fd: sendmsg(0): Message too long'), { code: 255 })
      }

      return 'CLEAR'
    }
  }

  await assertRemoteInstallUpdateClear(ssh, '/home/alice/.hermes', '~/.local/bin/hermes')
  assert.equal(calls.length, 1)
  assert.equal(calls[0].stdinData, REMOTE_MARKER_GATE_PY)
  assert.ok(!calls[0].command.includes('marker_verdict'), 'the judge must not be inlined in argv')
  assert.ok(calls[0].command.length < 512, `gate command grew to ${calls[0].command.length} bytes`)
})

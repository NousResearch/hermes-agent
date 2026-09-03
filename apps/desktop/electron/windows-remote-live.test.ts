import assert from 'node:assert/strict'

import { test } from 'vitest'

import { pickLocalPort, SshConnection } from './ssh-connection'
import { connectWindowsRemote } from './windows-remote-lifecycle'

// Live test against a real Windows host over SSH. Opt-in: set the env trio to
// your test rig; skipped everywhere else (CI, other machines).
//   HERMES_WIN_SSH_HOST   ssh alias/host of the Windows box
//   HERMES_WIN_SSH_USER   remote user
//   HERMES_WIN_SSH_HERMES absolute path to the remote hermes.exe under test
const liveHost = process.env.HERMES_WIN_SSH_HOST || ''
const liveUser = process.env.HERMES_WIN_SSH_USER || ''
const configuredHermes = process.env.HERMES_WIN_SSH_HERMES || ''
const ownershipId = '89abcdef0123456789abcdef01234567'

function fetchJson(url, token, path) {
  return fetch(`${url}${path}`, { headers: { 'X-Hermes-Session-Token': token } }).then(async response => {
    if (!response.ok) {
      throw new Error(`${response.status}: ${await response.text()}`)
    }

    return response.json()
  })
}

test.skipIf(!liveHost || !liveUser || !configuredHermes)(
  'live Windows remote lifecycle spawns, authenticates, reuses, and cleans exact ownership',
  async () => {
    const ssh = new SshConnection({ host: liveHost, user: liveUser, port: 22, keyPath: '' }, { mux: true })
    await ssh.open()

    const common = {
      ssh,
      ownershipId,
      profile: '',
      remoteHermesPath: configuredHermes,
      pickLocalPort,
      forward: (local, remote) => ssh.forward(local, remote),
      cancelForward: (local, remote) => ssh.cancelForward(local, remote),
      waitForHermes: async (baseUrl, token) => {
        for (let i = 0; i < 40; i++) {
          try {
            await fetchJson(baseUrl, token, '/api/status')

            return
          } catch {
            void 0
          }

          await new Promise(resolve => setTimeout(resolve, 250))
        }

        throw new Error('status timeout')
      },
      probeReuseProof: async (baseUrl, token, nonce) => {
        const body: any = await fetchJson(baseUrl, token, '/api/ssh/ownership')

        return body.sshOwnerNonce === nonce ? 'authenticated-ok' : 'authenticated-stale'
      },
      rememberLog: () => {}
    }

    let first
    let second

    try {
      first = await connectWindowsRemote(common)
      assert.equal(first.platform.os, 'Windows')
      assert.equal(first.reused, false)
      const status: any = await fetchJson(first.baseUrl, first.token, '/api/status')
      assert.ok(status)
      await ssh.cancelForward(first.localPort, first.remotePort)
      second = await connectWindowsRemote({ ...common, reuseToken: first.token })
      assert.equal(second.reused, true)
      assert.equal(second.pid, first.pid)
      assert.equal(second.spawnNonce, first.spawnNonce)
    } finally {
      if (second) {
        await ssh.cancelForward(second.localPort, second.remotePort)
      }

      const runtimeScript = `& '${configuredHermes.replace('hermes.exe', 'python.exe')}' -m hermes_cli.windows_ssh_runtime read-lock '${ownershipId}'`

      const lock: any = JSON.parse(
        await ssh.exec(`powershell.exe -NoProfile -NonInteractive -Command "${runtimeScript}"`)
      )

      if (lock) {
        const python = configuredHermes.replace('hermes.exe', 'python.exe')
        const terminate = `& '${python}' -m hermes_cli.windows_ssh_runtime terminate '${lock.pid}' '${lock.creationTimeNs}' '${lock.hermesPath}' '${lock.spawnNonce}'`
        await ssh.exec(`powershell.exe -NoProfile -NonInteractive -Command "${terminate}"`)
        await ssh.exec(
          `powershell.exe -NoProfile -NonInteractive -Command "& '${python}' -m hermes_cli.windows_ssh_runtime remove-lock '${ownershipId}'"`
        )
        await ssh.exec(
          `powershell.exe -NoProfile -NonInteractive -Command "& '${python}' -m hermes_cli.windows_ssh_runtime remove-log '${ownershipId}' '${lock.spawnNonce}'"`
        )
      }

      await ssh.close()
    }
  },
  90_000
)

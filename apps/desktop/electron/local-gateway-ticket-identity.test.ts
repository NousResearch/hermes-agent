import { execFile, spawn } from 'node:child_process'
import { once } from 'node:events'
import fs from 'node:fs/promises'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'
import readline from 'node:readline'
import { promisify } from 'node:util'

import { expect, test, vi } from 'vitest'

import {
  configurePythonGatewayTicketClient, type GatewayEndpoint, isStaleLocalGatewayError, mintLocalGatewayTicket,
  nativeGatewayHttpHeaders, redialLocalGateway, reservedHome
} from './local-gateway'
import { closeGatewayTicketBridges, mintGatewayTicketWithPython } from './local-gateway-python'
import { shortSocketTmpDir } from './local-gateway.test-helpers'

// A gateway ticket names a profile home by the identity the gateway reserved for it: its
// `Path.resolve()`, which Desktop must reproduce exactly or the gateway refuses the ticket.

const ORIGIN = 'http://127.0.0.1:1234'
const LINK = process.platform === 'win32' ? 'junction' : 'dir'
const PYTHON = process.env.HERMES_TEST_PYTHON || (process.platform === 'win32' ? 'python' : 'python3')

const endpointFor = (profile_id: string): GatewayEndpoint => ({
  profile_id, instance_id: 'owner', authority_epoch: 1, runtime_protocol: 1, api_origin: ORIGIN,
  capabilities: ['session-authority-v1'], supervisor: 'none', control_home: null
})

const descriptorFor = (endpoint: GatewayEndpoint) => ({ gatewayEndpoint: endpoint, baseUrl: ORIGIN })

async function tmpHome(prefix: string): Promise<string> {
  return fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), prefix)))
}

async function sameDirectory(a: string, b: string): Promise<boolean> {
  const [left, right] = await Promise.all([a, b].map(dir => fs.stat(dir, { bigint: true }).catch(() => null)))

  return Boolean(left && right && left.dev === right.dev && left.ino === right.ino)
}

async function pythonResolve(homes: string[]): Promise<string[]> {
  const script = 'import json, sys; from pathlib import Path; print(json.dumps([str(Path(p).resolve()) for p in sys.argv[1:]]))'
  const { stdout } = await promisify(execFile)(PYTHON, ['-c', script, ...homes], { windowsHide: true })

  return JSON.parse(stdout)
}

// Python keeps the given spelling of ordinary components (POSIX) or answers the final on-disk path
// (Windows), and joins a relative link target onto the prefix resolved so far.
test('a profile home resolves to exactly the home the gateway reserves for it', async () => {
  const base = await tmpHome('gw-identity-')
  const home = path.join(base, 'home')
  await Promise.all([fs.mkdir(path.join(home, 'profiles'), { recursive: true }), fs.mkdir(path.join(base, 'other'))])
  await fs.mkdir(path.join(home, 'work-data'))
  // Windows junctions take an absolute target; a POSIX profile link is often relative.
  await fs.symlink(process.platform === 'win32' ? path.join(home, 'work-data') : '../work-data', path.join(home, 'profiles', 'work'), LINK)
  await fs.symlink(path.join(base, 'other'), path.join(base, 'link'), LINK)
  // A letter-case respelling exists only on a case-insensitive volume (Windows, macOS by default).
  const spelled = path.join(base, 'HOME')
  const respelled = await sameDirectory(home, spelled) ? [spelled, path.join(spelled, 'profiles', 'work')] : []
  const homes = [home, path.join(home, 'profiles', 'work'), path.join(base, 'link'), ...respelled]

  try {
    expect(await Promise.all(homes.map(reservedHome))).toEqual(await pythonResolve(homes))
  } finally {
    await fs.rm(base, { recursive: true, force: true })
  }
})

test('a routed profile with no directory is named, not retried as a stale gateway', async () => {
  const root = await tmpHome('gw-gone-')
  const ensure = vi.fn(async () => descriptorFor(endpointFor(root)))
  const forget = vi.fn()

  try {
    const failure = await redialLocalGateway({ ensure, forget, use: descriptor => nativeGatewayHttpHeaders(descriptor, `${ORIGIN}/api/sessions?profile=gone`) }).catch(error => error)
    expect(failure.message).toContain(`Profile 'gone' does not exist`)
    expect(isStaleLocalGatewayError(failure)).toBe(false)
    expect([ensure.mock.calls.length, forget.mock.calls.length]).toEqual([1, 0])
  } finally {
    await fs.rm(root, { recursive: true, force: true })
  }
})

/** A POSIX control socket that mints like the gateway's: only for a profile id it serves, exactly. */
async function controlSocket(home: string, served: string[]) {
  const requests: Array<Record<string, unknown>> = []

  const server = net.createServer(socket => socket.once('data', chunk => {
    const params = JSON.parse(chunk.toString()).params
    requests.push(params)
    socket.end(JSON.stringify(served.includes(params.profile_id)
      ? { protocol: 1, id: 1, ok: true, result: { profile_id: params.profile_id, instance_id: 'owner', runtime_protocol: 1, ticket: `grant-${requests.length}` } }
      : { protocol: 1, id: 1, ok: false, error: 'PermissionError: bootstrap binding rejected' }) + '\n')
  }))

  await new Promise<void>(resolve => server.listen(path.join(home, 'gateway.sock'), resolve))
  await fs.chmod(path.join(home, 'gateway.sock'), 0o600)

  return { requests, close: () => new Promise<void>(resolve => server.close(() => resolve())) }
}

// #138754: on a case-insensitive volume macOS realpath(3) answers with the on-disk letter case,
// while the gateway reserves (and advertises) the spelling HERMES_HOME gave it.
test.skipIf(process.platform === 'win32')('a home advertised in another letter case mints under the gateway spelling, as does a profile linked relative to it', async ctx => {
  const home = await shortSocketTmpDir('gw-case-')
  const spelled = path.join(path.dirname(home), path.basename(home).toUpperCase())

  if (!await sameDirectory(home, spelled)) {
    await fs.rm(home, { recursive: true, force: true })
    ctx.skip('this volume is case-sensitive')
  }

  await fs.mkdir(path.join(home, 'profiles'), { mode: 0o700 })
  await fs.mkdir(path.join(home, 'work-data'), { mode: 0o700 })
  await fs.symlink('../work-data', path.join(home, 'profiles', 'work'))
  // The gateway's Path.resolve() of <spelled>/profiles/work: the link text joined onto the spelled prefix.
  const linked = path.join(spelled, 'work-data')
  const gateway = await controlSocket(home, [spelled, linked])
  const descriptor = descriptorFor(endpointFor(spelled))

  try {
    expect(await nativeGatewayHttpHeaders(descriptor, `${ORIGIN}/api/profiles`)).toEqual({ 'X-Hermes-Gateway-Ticket': 'grant-1' })
    expect(await nativeGatewayHttpHeaders(descriptor, `${ORIGIN}/api/sessions?profile=work`)).toEqual({ 'X-Hermes-Gateway-Ticket': 'grant-2' })
    expect(gateway.requests.map(request => request.profile_id)).toEqual([spelled, linked])
  } finally {
    await gateway.close()
    await fs.rm(home, { recursive: true, force: true })
  }
})

// Security control (passes before and after the fix): identity is not a licence to follow links
// in an advertised home. One that leads to another directory is still refused, without a request.
test.skipIf(process.platform === 'win32')('an advertised home reached through a link to another directory is refused before the gateway is asked', async () => {
  const real = await shortSocketTmpDir('gw-real-')
  const links = await shortSocketTmpDir('gw-links-')
  await fs.symlink(real, path.join(links, 'home'), 'dir')
  await fs.symlink(path.dirname(real), path.join(links, 'parent'), 'dir')
  const gateway = await controlSocket(real, [real])

  try {
    for (const advertised of [path.join(links, 'home'), path.join(links, 'parent', path.basename(real))]) {
      await expect(mintLocalGatewayTicket(endpointFor(advertised), 'native-http')).rejects.toThrow('Noncanonical gateway profile')
      await expect(mintLocalGatewayTicket({ ...endpointFor(real), control_home: advertised }, 'native-http')).rejects.toThrow('Noncanonical gateway profile')
    }

    expect(gateway.requests).toEqual([])
  } finally {
    await gateway.close()
    await Promise.all([real, links].map(dir => fs.rm(dir, { recursive: true, force: true })))
  }
})

// The gateway serves a linked `profiles/<name>` at its resolved home, so it never mints a ticket for
// the joined `profiles/<name>` path.
test.skipIf(process.platform === 'win32')('a symlinked named profile mints for the target the gateway reserved', async () => {
  const [root, target] = await Promise.all(['gw-root-', 'gw-target-'].map(shortSocketTmpDir))
  await fs.mkdir(path.join(root, 'profiles'), { mode: 0o700 })
  await fs.symlink(target, path.join(root, 'profiles', 'work'), 'dir')
  const gateway = await controlSocket(root, [root, target])

  try {
    expect(await nativeGatewayHttpHeaders(descriptorFor(endpointFor(root)), `${ORIGIN}/api/sessions?profile=work`)).toEqual({ 'X-Hermes-Gateway-Ticket': 'grant-1' })
    expect(gateway.requests).toEqual([{ profile_id: target, instance_id: 'owner', purpose: 'native-http' }])
  } finally {
    await gateway.close()
    await Promise.all([root, target].map(dir => fs.rm(dir, { recursive: true, force: true })))
  }
})

// The gateway's real Windows ticket path: its control handler, ticket store and profile reservation
// behind the runtime pipe, reached through the runtime's own ticket bridge. Each stdin line is the
// HTTP boundary's verdict on a minted ticket: redeem it, then check the request's `?profile=`.
const WINDOWS_GATEWAY = `
import json, sys
from pathlib import Path
from types import SimpleNamespace
from gateway.control_socket import GatewayControlServer
from gateway.run_runtime import reserved_profile_homes
from gateway.runtime_bootstrap import TicketStore
from gateway.runtime_bootstrap_windows import NativeControlServer
from hermes_cli.dashboard_auth.native_http import _own_profile
root = Path(sys.argv[1])
runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True, _runtime_profile_homes=[('work', root / 'profiles' / 'work')]))
control = GatewayControlServer(root)
control.ticket_store = TicketStore('owner', frozenset(str(home) for _, home in reserved_profile_homes(runner)))
pipe = NativeControlServer(control._home, control.handle_request_line)
pipe.start()
print('ready', flush=True)
try:
    for line in sys.stdin:
        request = json.loads(line)
        try:
            grant = control.ticket_store.redeem(request['ticket'], profile_id=None, purpose='native-http')
            _own_profile(request['profile'], grant['profile_id'], current=False)
            print('accepted', flush=True)
        except PermissionError as exc:
            print(f'refused: {exc}', flush=True)
finally:
    pipe.close()
`

test.skipIf(process.platform !== 'win32')('a junction-linked named profile mints through the runtime bridge for its reserved target', async () => {
  const [root, target, stray] = await Promise.all(['gw-root-', 'gw-target-', 'gw-stray-'].map(tmpHome))
  await fs.mkdir(path.join(root, 'profiles'))
  await fs.symlink(target, path.join(root, 'profiles', 'work'), 'junction')
  await fs.symlink(stray, path.join(root, 'profiles', 'stray'), 'junction')
  await fs.writeFile(path.join(target, 'config.yaml'), '{}\n')
  const checkout = path.resolve(import.meta.dirname, '../../..')
  const env = { ...process.env, PYTHONPATH: [checkout, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter), PYTHONUTF8: '1' }
  const server = spawn(PYTHON, ['-c', WINDOWS_GATEWAY, root], { env: { ...env, HERMES_HOME: root }, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] })
  let stderr = ''
  server.stderr.on('data', chunk => { stderr += chunk })
  const replies = readline.createInterface({ input: server.stdout })[Symbol.asyncIterator]()
  configurePythonGatewayTicketClient((endpoint, purpose) => mintGatewayTicketWithPython({ command: PYTHON, env }, checkout, endpoint, purpose))
  const descriptor = descriptorFor(endpointFor(root))

  try {
    expect((await replies.next()).value, stderr).toBe('ready')
    const ticket = (await nativeGatewayHttpHeaders(descriptor, `${ORIGIN}/api/sessions?profile=work`))['X-Hermes-Gateway-Ticket']
    server.stdin.write(JSON.stringify({ ticket, profile: 'work' }) + '\n')
    expect((await replies.next()).value).toBe('accepted')
    // The gateway's exact membership still decides: a junction to a directory it does not serve.
    await expect(nativeGatewayHttpHeaders(descriptor, `${ORIGIN}/api/sessions?profile=stray`)).rejects.toThrow('Gateway ticket bootstrap failed')
  } finally {
    closeGatewayTicketBridges()
    configurePythonGatewayTicketClient(undefined as never)
    const stopped = server.exitCode === null ? once(server, 'close') : undefined
    server.stdin.end()
    await stopped
    await Promise.all([root, target, stray].map(dir => fs.rm(dir, { recursive: true, force: true })))
  }
}, 60_000)

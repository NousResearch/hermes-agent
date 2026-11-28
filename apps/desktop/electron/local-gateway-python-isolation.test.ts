import type * as ChildProcessAPI from 'node:child_process'
import { EventEmitter } from 'node:events'

import { afterEach, beforeEach, expect, test, vi } from 'vitest'

// A real timeout is a verdict on one request. The helper answers in order, so a request written a
// moment ago has not had its time when an older one runs out: it must not be failed with it, and the
// helper that may answer it must not be killed. (The idea of rejecting only the timed-out request is
// from a fork commit linked on issue #151351, kvnloo d380a6d; this code and these tests were written
// separately, and unlike that commit a helper that stays silent is still retired.)
class FakeHelper extends EventEmitter {
  requests: { id: number; purpose: string }[] = []
  stdin = Object.assign(new EventEmitter(), {
    write: (line: string) => { this.requests.push(JSON.parse(line));

 return true }
  })
  stdout = Object.assign(new EventEmitter(), { setEncoding: () => undefined })
  stderr = Object.assign(new EventEmitter(), { resume: () => undefined })
  kill = vi.fn(() => { queueMicrotask(() => this.emit('close', null, 'SIGKILL'));

 return true })

  reply(index: number, ticket = 'grant') {
    this.stdout.emit('data', JSON.stringify({ id: this.requests[index].id, ticket }) + '\n')
  }
}

const helpers: FakeHelper[] = []

vi.mock('node:child_process', async importOriginal => ({
  ...await importOriginal<typeof ChildProcessAPI>(),
  spawn: vi.fn(() => { const helper = new FakeHelper(); helpers.push(helper);

 return helper })
}))

const { closeGatewayTicketBridges, mintGatewayTicketWithPython } = await import('./local-gateway-python')
const { GatewayTicketTimeoutError, isStaleLocalGatewayError } = await import('./local-gateway')

const backend = { command: 'python', env: {} }
const endpoint = { profile_id: '/home', instance_id: 'owner', runtime_protocol: 1, control_home: null }
const mint = () => mintGatewayTicketWithPython(backend, '/cwd', endpoint, 'native-http')
const settled = (ticket: Promise<string>) => ticket.catch((error: unknown) => error)
let clock = 0

/** A running loop: the clock and the timers move together, then one turn for the pipe to be read. */
const run = async (ms: number) => {
  clock += ms
  vi.advanceTimersByTime(ms)
  await new Promise(resolve => setImmediate(resolve))
}

beforeEach(() => {
  helpers.length = 0
  clock = 0
  vi.spyOn(performance, 'now').mockImplementation(() => clock)
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
})

afterEach(() => {
  closeGatewayTicketBridges()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

test('a timeout rejects only the request that ran out; a younger one resolves on the same helper', async () => {
  const older = settled(mint())
  const [helper] = helpers

  await run(6_000)

  const younger = mint()

  await run(4_001)

  const failure = await older
  expect(failure).toBeInstanceOf(GatewayTicketTimeoutError)
  expect(isStaleLocalGatewayError(failure)).toBe(false)
  expect(helper.kill).not.toHaveBeenCalled()

  helper.reply(1, 'grant-younger')
  await expect(younger).resolves.toBe('grant-younger')
  expect(helpers).toHaveLength(1)
})

test('a late reply to a request that timed out is dropped, and the helper serves the next mint', async () => {
  const first = settled(mint())
  const [helper] = helpers

  await run(10_001)
  expect(await first).toBeInstanceOf(GatewayTicketTimeoutError)

  helper.reply(0, 'too-late')
  expect(helper.kill).not.toHaveBeenCalled()

  const next = mint()
  helper.reply(1, 'grant-next')
  await expect(next).resolves.toBe('grant-next')
  expect(helpers).toHaveLength(1)
})

test('two timeouts in a row with no reply between them retire the helper as hung', async () => {
  const first = settled(mint())
  const [helper] = helpers

  await run(6_000)

  const second = settled(mint())

  await run(4_001)
  expect(await first).toBeInstanceOf(GatewayTicketTimeoutError)
  expect(helper.kill).not.toHaveBeenCalled()

  await run(2_000)

  const third = settled(mint())

  await run(4_000)

  // The second ran out too and the helper never spoke: it is hung. What still waits fails now, as a
  // timeout (the owner is not known to be gone), and the next mint starts a fresh helper.
  const [b, c] = await Promise.all([second, third])
  expect(b).toBeInstanceOf(GatewayTicketTimeoutError)
  expect(c).toBeInstanceOf(GatewayTicketTimeoutError)
  expect(isStaleLocalGatewayError(c)).toBe(false)
  expect(helper.kill).toHaveBeenCalledWith('SIGKILL')

  const fresh = mint()
  expect(helpers).toHaveLength(2)
  helpers[1].reply(0)
  await expect(fresh).resolves.toBe('grant')
})

test('a reply between two timeouts shows a live helper: it is kept', async () => {
  const first = settled(mint())
  const [helper] = helpers

  await run(10_001)
  expect(await first).toBeInstanceOf(GatewayTicketTimeoutError)

  const second = mint()
  helper.reply(1, 'grant-2')
  await expect(second).resolves.toBe('grant-2')

  const third = settled(mint())

  await run(10_001)
  expect(await third).toBeInstanceOf(GatewayTicketTimeoutError)
  expect(helper.kill).not.toHaveBeenCalled()
  expect(helpers).toHaveLength(1)
})

test('a helper left with nothing pending after a timeout still idles out', async () => {
  const first = settled(mint())
  const [helper] = helpers

  await run(10_001)
  expect(await first).toBeInstanceOf(GatewayTicketTimeoutError)
  expect(helper.kill).not.toHaveBeenCalled()

  await run(5 * 60_000)
  expect(helper.kill).toHaveBeenCalledWith('SIGKILL')
})

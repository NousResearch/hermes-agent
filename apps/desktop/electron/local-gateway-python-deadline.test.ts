import type * as ChildProcessAPI from 'node:child_process'
import { EventEmitter } from 'node:events'

import { afterEach, beforeEach, expect, test, vi } from 'vitest'

// The ticket helper is a long-lived child; a fake one lets each test decide exactly when a reply
// reaches the pipe relative to the request deadline.
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
const { GatewayTicketTimeoutError, isStaleLocalGatewayError, redialLocalGateway } = await import('./local-gateway')

const backend = { command: 'python', env: {} }
const endpoint = { profile_id: '/home', instance_id: 'owner', runtime_protocol: 1, control_home: null }
const mint = () => mintGatewayTicketWithPython(backend, '/cwd', endpoint, 'native-http')
const nextTurn = () => new Promise(resolve => setImmediate(resolve))
let clock = 0

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

test('a reply that reached the pipe during a main-process stall resolves, and the helper is kept', async () => {
  const first = mint()
  const [helper] = helpers

  // Main stalls 10 s: the deadline is due when the loop resumes, the helper's reply is in the pipe
  // but unread. Timers run first, then the poll phase delivers the reply.
  clock = 10_001
  vi.advanceTimersByTime(10_001)
  helper.reply(0)

  await expect(first).resolves.toBe('grant')
  expect(helper.kill).not.toHaveBeenCalled()

  const second = mint()
  helper.reply(1, 'grant-2')
  await expect(second).resolves.toBe('grant-2')
  expect(helpers).toHaveLength(1)
})

test('a deadline that fires late gives the helper a grace on a running loop before it is judged', async () => {
  const pending = mint().catch(error => error)
  const [helper] = helpers

  // The loop (or the machine) was stopped for 25 s: the helper's silence was never measured.
  clock = 35_000
  vi.advanceTimersByTime(10_001)
  await nextTurn()
  expect(helper.kill).not.toHaveBeenCalled()

  helper.reply(0)
  await expect(pending).resolves.toBe('grant')
  expect(helper.kill).not.toHaveBeenCalled()
})

test('a helper silent past its deadline on a running loop is retired with a timeout, not a dead owner', async () => {
  const first = mint().catch(error => error)
  const second = mint().catch(error => error)
  const [helper] = helpers

  clock = 10_001
  vi.advanceTimersByTime(10_001)
  await nextTurn()

  const [a, b] = await Promise.all([first, second])
  // Each request gets its own verdict; the second one in a row with no reply is what retires the helper
  // (local-gateway-python-isolation.test.ts).
  expect(a).toBeInstanceOf(GatewayTicketTimeoutError)
  expect(b).toBeInstanceOf(GatewayTicketTimeoutError)
  expect(isStaleLocalGatewayError(a)).toBe(false)
  expect(helper.kill).toHaveBeenCalledWith('SIGKILL')

  // A timeout keeps the cached owner: no forget, no second `gateway ensure`.
  let ensures = 0
  let forgets = 0

  await expect(redialLocalGateway({
    ensure: async () => { ensures++

      return endpoint },
    forget: () => { forgets++ },
    use: async () => { throw a }
  })).rejects.toBe(a)
  expect({ ensures, forgets }).toEqual({ ensures: 1, forgets: 0 })

  // The next mint starts a fresh helper.
  const third = mint()
  expect(helpers).toHaveLength(2)
  helpers[1].reply(0)
  await expect(third).resolves.toBe('grant')
})

test('a helper idle for a steady poll interval is reused; it retires only once the app has gone quiet', async () => {
  const first = mint()
  helpers[0].reply(0)
  await expect(first).resolves.toBe('grant')

  // The open app's slowest steady poll leaves the helper idle ~30 s; a little jitter must not cost a cold start.
  vi.advanceTimersByTime(30_100)
  const second = mint()
  helpers[0].reply(1, 'grant-2')
  await expect(second).resolves.toBe('grant-2')
  expect(helpers).toHaveLength(1)
  expect(helpers[0].kill).not.toHaveBeenCalled()

  vi.advanceTimersByTime(5 * 60_000)
  expect(helpers[0].kill).toHaveBeenCalledWith('SIGKILL')
})

test('a stall that ends with the helper still silent is judged after the grace, on the running loop', async () => {
  const pending = mint().catch(error => error)
  const [helper] = helpers

  clock = 35_000
  vi.advanceTimersByTime(10_001)
  await nextTurn()
  expect(helper.kill).not.toHaveBeenCalled()

  // The grace runs on a live loop: on time, and the pipe is read before the verdict.
  clock = 37_001
  vi.advanceTimersByTime(2_000)
  await nextTurn()
  expect(await pending).toBeInstanceOf(GatewayTicketTimeoutError)
  // One request ran out: that is its verdict, not yet the helper's.
  expect(helper.kill).not.toHaveBeenCalled()
})

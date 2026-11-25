// Session-wire readiness for HermesGateway's dials. A native dial is canonical by provenance and
// ready at once; a public dial is ready at the first `gateway.ready` its own socket delivers. The
// gate owns each dial's settle, its timeout, the `open` it holds back from state subscribers, and
// a request's cancellable wait; HermesGateway connects it to its transport events.
import type { ConnectionState } from '@hermes/shared'

import { TimeoutError } from '@/lib/with-timeout'

export const GATEWAY_NOT_CONNECTED_MESSAGE = 'Hermes gateway is not connected'

// A public dial counts its readiness wait inside the shared client's 15 s connect budget, so a
// peer that opens but never sends `gateway.ready` fails as fast as one that never opens.
const GATEWAY_READINESS_TIMEOUT_MS = 15_000

type StateHandler = (state: ConnectionState) => void

interface Dial {
  generation: number
  opened: boolean
  promise: Promise<void>
  ready: boolean
  reject: (error: Error) => void
  resolve: () => void
  settled: boolean
  timer?: ReturnType<typeof setTimeout>
}

function createDial(generation: number, ready: boolean): Dial {
  let resolve = () => {}

  let reject: (error: Error) => void = () => {}

  const promise = new Promise<void>((res, rej) => {
    resolve = res
    reject = rej
  })

  // connect() awaits it after the transport; a failure first seen there must not be unhandled.
  promise.catch(() => {})
  const dial = { generation, opened: false, promise, ready: false, reject, resolve, settled: false }

  if (ready) {
    dial.ready = dial.settled = true
    resolve()
  }

  return dial
}

export function remainingRequestBudget(deadline: number, signal?: AbortSignal): number {
  signal?.throwIfAborted()

  if (deadline === Number.POSITIVE_INFINITY) { return 0 }
  const remaining = deadline - Date.now()

  if (remaining <= 0) { throw new Error('Hermes gateway request timed out') }

  return remaining
}

export class DialReadiness {
  private dial = createDial(0, true)
  // State listeners told `open` before readiness are held until the wire is known: an open
  // socket that never becomes ready is not a healthy connection to reconnect ladders.
  private readonly held = new Set<StateHandler>()

  /** `transportOpen` reads the shared client's own state, below the `open` this gate holds. */
  constructor(private readonly transportOpen: () => boolean) {}

  get ready(): boolean {
    return this.dial.ready
  }

  get promise(): Promise<void> {
    return this.dial.promise
  }

  /** Start a dial's readiness; a public one fails after the readiness timeout. */
  begin(generation: number, native: boolean): Promise<void> {
    const dial = createDial(generation, native)
    this.dial = dial

    if (!dial.settled) {
      dial.timer = setTimeout(() => this.settle(dial, new TimeoutError('Hermes gateway protocol readiness timed out')), GATEWAY_READINESS_TIMEOUT_MS)
    }

    return dial.promise
  }

  /** The first `gateway.ready` of the current dial: `select` picks the wire before any held
   *  subscriber hears `open`; a later frame on the same socket cannot flip it. */
  adoptReady(generation: number, select: () => void): void {
    const dial = this.dial

    if (dial.settled || dial.generation !== generation) { return }
    select()
    this.settle(dial)
  }

  /** Every transport transition, seen before any subscriber, so a subscriber's redial from its
   *  `closed` handler starts after the old dial has failed. */
  trackTransport(state: ConnectionState): void {
    if (state === 'open') { this.dial.opened = true }

    if (state === 'closed' || state === 'error') { this.fail() }
  }

  /** Fail the current dial with one text whoever closed the socket; it names the closed
   *  connection, as the transport does. */
  fail(): void {
    const dial = this.dial

    this.settle(dial, new Error(dial.opened ? 'Hermes gateway connection closed before protocol readiness' : GATEWAY_NOT_CONNECTED_MESSAGE))
  }

  /** Fail the dial `promise` belongs to (a connect() whose transport rejected). */
  failDial(promise: Promise<void>, error: Error): void {
    if (this.dial.promise === promise) { this.settle(this.dial, error) }
  }

  /** Wrap a state subscriber so it hears `open` only once the current dial is ready. */
  hold(handler: StateHandler): { deliver: StateHandler; release: () => void } {
    return {
      deliver: state => {
        if (state === 'open' && !this.dial.ready) {
          this.held.add(handler)

          return
        }

        this.held.delete(handler)
        handler(state)
      },
      release: () => { this.held.delete(handler) }
    }
  }

  /** A request made before readiness waits for it inside its own deadline and abort signal. A
   *  failed dial answers as the transport would: not connected, which the request layer's
   *  reconnect and the connectors' retry key on. */
  async wait(deadline: number, signal: AbortSignal | undefined, timedOut: () => Error): Promise<void> {
    remainingRequestBudget(deadline, signal)
    const stops: Array<() => void> = []

    const exits = new Promise<never>((_, reject) => {
      if (signal) {
        const onAbort = () => reject(signal.reason)
        signal.addEventListener('abort', onAbort, { once: true })
        stops.push(() => signal.removeEventListener('abort', onAbort))
      }

      if (deadline !== Number.POSITIVE_INFINITY) {
        const timer = setTimeout(() => reject(timedOut()), Math.max(0, deadline - Date.now()))
        stops.push(() => clearTimeout(timer))
      }
    })

    const ready = this.dial.promise.catch(() => { throw new Error(GATEWAY_NOT_CONNECTED_MESSAGE) })

    try { await Promise.race([ready, exits]) } finally { for (const stop of stops) { stop() } }
  }

  private settle(dial: Dial, error?: Error): void {
    if (dial.settled) { return }
    dial.settled = true
    clearTimeout(dial.timer)

    if (error) {
      this.held.clear()
      dial.reject(error)

      return
    }

    dial.ready = true
    dial.resolve()

    // Rechecked per handler: an earlier one may unsubscribe a later one, or close or redial the
    // client. A handler that already heard `closed` has left the held set.
    for (const handler of [...this.held]) {
      if (this.dial !== dial || !this.transportOpen()) { return }

      if (this.held.delete(handler)) { handler('open') }
    }
  }
}

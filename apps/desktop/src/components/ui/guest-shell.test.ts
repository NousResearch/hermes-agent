// @vitest-environment jsdom
import { once } from 'node:events'
import { readFileSync } from 'node:fs'
import { ReadableStream, TransformStream, WritableStream } from 'node:stream/web'
import { runInNewContext } from 'node:vm'
import { MessageChannel, MessagePort } from 'node:worker_threads'

import { afterEach, describe, expect, it, vi } from 'vitest'

const source = readFileSync('../../hermes_cli/webapp_previews.py', 'utf8')
const html = source.match(/PREVIEW_HTML = """([\s\S]*?)"""/)![1]
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)]

expect(scripts).toHaveLength(1)

interface RelayEvent {
  data: unknown
  origin: string
  ports: MessagePort[]
  source: unknown
}

const ports: MessagePort[] = []

afterEach(() => {
  ports.splice(0).forEach(port => port.close())
})

function channel() {
  const result = new MessageChannel()

  ports.push(result.port1, result.port2)

  return result
}

function shell(direction: 'down' | 'up') {
  const parent = { postMessage: vi.fn<(data: unknown, origin: string, transfers?: object[]) => void>() }
  const guest = { postMessage: vi.fn<(data: unknown, origin: string, transfers?: object[]) => void>() }
  const frame: { contentWindow: typeof guest | null } = { contentWindow: guest }
  let receive!: (event: RelayEvent) => void

  runInNewContext(scripts[0][1], {
    document: { createElement: () => frame },
    parent,
    window: {},
    location: { href: 'https://hermes.test/assets/webapp-guest.html' },
    addEventListener: (_type: string, listener: typeof receive) => {
      receive = listener
    },
    URL,
    Object,
    Array,
    Map,
    Set,
    Error,
    ArrayBuffer,
    SharedArrayBuffer,
    MessagePort,
    ReadableStream,
    WritableStream,
    TransformStream
  })

  return {
    frame,
    post: direction === 'down' ? guest.postMessage : parent.postMessage,
    send: (data: unknown, transferredPorts: MessagePort[] = []) =>
      receive({
        data,
        origin: direction === 'down' ? 'https://hermes.test' : 'null',
        ports: transferredPorts,
        source: direction === 'down' ? parent : guest
      })
  }
}

function clone<T>(data: T, transfers: object[]) {
  return structuredClone(data, { transfer: transfers as Transferable[] })
}

function expectTransfer(post: ReturnType<typeof shell>['post'], data: unknown, expected: object[]) {
  expect(post).toHaveBeenCalledTimes(1)
  expect(post.mock.calls[0][0]).toBe(data)
  expect(post.mock.calls[0][1]).toBe('*')
  const actual = post.mock.calls[0][2]!

  expect(actual).toHaveLength(expected.length)
  expected.forEach((value, index) => expect(actual[index]).toBe(value))
}

function stream() {
  return new ReadableStream<string>({
    start(controller) {
      controller.enqueue('delivered')
      controller.close()
    }
  })
}

describe.each(['down', 'up'] as const)('guest shell relay %s', direction => {
  it('delivers ports and streams across both structured-clone hops', async () => {
    const { send, post } = shell(direction)
    const { port1, port2 } = channel()
    const original = { port: port1, stream: stream() }
    const incoming = clone(original, [port1, original.stream])
    let delivered!: typeof incoming

    ports.push(incoming.port)
    post.mockImplementation((data, _origin, transfers) => {
      delivered = clone(data as typeof incoming, transfers!)
      ports.push(delivered.port)
    })
    send(incoming, [incoming.port])
    expectTransfer(post, incoming, [incoming.port, incoming.stream])
    expect(await delivered.stream.getReader().read()).toEqual({ done: false, value: 'delivered' })
    const message = once(delivered.port, 'message')

    port2.postMessage('connected')
    expect(await message).toEqual(['connected'])
  })

  it('walks cloneable containers with cycles and duplicates, but never inside transferables or shared buffers', () => {
    const { send, post } = shell(direction)
    const { port1, port2 } = channel()
    const named = stream()
    const cause = stream()
    const errors = stream()
    const key = new ArrayBuffer(8)
    const value = new WritableStream()
    const member = new TransformStream()
    const array = Object.assign([port1], { named })

    const data = {
      array,
      error: new Error('cause', { cause }),
      aggregate: new AggregateError([errors]),
      map: new Map([[key, value]]),
      set: new Set([member]),
      plain: Object.assign(Object.create(null), { port1 }),
      shared: new SharedArrayBuffer(8)
    }

    Object.assign(array, { cycle: data })
    Object.assign(port1, { ignored: stream() })
    send(data, [port1, port2, port1])
    expectTransfer(post, data, [port1, port2, named, cause, errors, key, value, member])
  })

  it.each(['array property', 'Error.cause'] as const)(
    'delivers a stream under %s across two real clone hops',
    async shape => {
      const { send, post } = shell(direction)
      const readable = stream()

      const original =
        shape === 'array property' ? Object.assign([], { stream: readable }) : new Error('wrapped', { cause: readable })

      const incoming = clone(original, [readable])
      let delivered!: typeof incoming

      post.mockImplementation((data, _origin, transfers) => {
        delivered = clone(data as typeof incoming, transfers!)
      })
      send(incoming)
      expect(post).toHaveBeenCalledTimes(1)
      const received = delivered instanceof Error ? (delivered.cause as ReadableStream) : delivered.stream

      expect(await received.getReader().read()).toEqual({ done: false, value: 'delivered' })
    }
  )

  it('walks a payload nested deeper than the call stack', () => {
    const { send, post } = shell(direction)
    const { port1 } = channel()
    let data: unknown = { port1 }

    for (let depth = 0; depth < 20000; depth += 1) {
      data = new Map([['next', data]])
    }

    send(data)
    expectTransfer(post, data, [port1])
  })

  it.each([false, true])('retries without transfers and closes ports only if both posts fail (both=%s)', both => {
    const { send, post } = shell(direction)
    const { port1, port2 } = channel()
    const close1 = vi.spyOn(port1, 'close')
    const close2 = vi.spyOn(port2, 'close')
    const data = { port: port2, buffer: new ArrayBuffer(8) }

    const fail = () => {
      throw new DOMException('cannot transfer', 'DataCloneError')
    }

    post.mockImplementationOnce(fail)

    if (both) {
      post.mockImplementationOnce(fail)
    }

    expect(() => send(data, [port1])).not.toThrow()
    expect(post.mock.calls).toEqual([
      [data, '*', [port1, port2, data.buffer]],
      [data, '*']
    ])
    expect(close1).toHaveBeenCalledTimes(both ? 1 : 0)
    expect(close2).toHaveBeenCalledTimes(both ? 1 : 0)
  })
})

it('closes the ports of a parent message that arrives before the guest frame exists', () => {
  const { frame, send } = shell('down')
  const { port1, port2 } = channel()
  const close = vi.spyOn(port1, 'close')

  frame.contentWindow = null
  expect(() => send({ early: true }, [port1])).not.toThrow()
  expect(close).toHaveBeenCalledOnce()
  port2.close()
})

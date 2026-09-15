import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { afterEach, beforeEach, expect, test, vi } from 'vitest'

const { spawn } = vi.hoisted(() => ({ spawn: vi.fn() }))
vi.mock('node:child_process', () => ({ spawn }))

import { RealSessionBuilder } from './real-session-builder'

class FakeChild extends EventEmitter {
  stdin = new PassThrough()
  stdout = new PassThrough()
  stderr = new PassThrough()
  exitCode: number | null = null
  signalCode: NodeJS.Signals | null = null
  signals: string[] = []

  kill(signal: string): boolean {
    this.signals.push(signal)

    return true
  }

  exit(): void {
    this.exitCode = 0
    this.emit('exit', 0, null)
  }
}

beforeEach(() => {
  vi.useFakeTimers()
  spawn.mockReset()
})
afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
})

async function readyBuilder(child: FakeChild): Promise<RealSessionBuilder> {
  spawn.mockReturnValueOnce(child)
  const pending = RealSessionBuilder.start('/synthetic/bot-profile')
  child.stdout.write(JSON.stringify({ method: 'event', params: { type: 'gateway.ready' } }) + '\n')

  return pending
}

test('READY failure closes the owned seeding child before rejecting start', async () => {
  const child = new FakeChild()
  spawn.mockReturnValueOnce(child)
  child.stdin.on('finish', () => child.exit())
  const result = RealSessionBuilder.start('/synthetic/bot-profile').catch(error => error)
  child.emit('error', new Error('Synthetic READY failure'))
  await vi.advanceTimersByTimeAsync(0)

  expect(child.stdin.writableEnded).toBe(true)
  expect(child.exitCode).toBe(0)
  expect(String(await result)).toContain('Synthetic READY failure')
})

test('READY failure preserves the original error when owned-child cleanup also fails', async () => {
  const child = new FakeChild()
  spawn.mockReturnValueOnce(child)
  const result = RealSessionBuilder.start('/synthetic/bot-profile').catch(error => error)
  child.emit('error', new Error('Synthetic READY failure'))
  await vi.advanceTimersByTimeAsync(5_000)
  const failure = await result

  expect(failure).toBeInstanceOf(AggregateError)
  expect(String(failure.errors[0])).toContain('Synthetic READY failure')
  expect(String(failure.errors[1])).toContain('exit')
  expect(child.signals).toEqual(['SIGTERM'])
})

test('close rejects when SIGTERM was sent but owned-child exit is still unconfirmed', async () => {
  const child = new FakeChild()
  const builder = await readyBuilder(child)
  const result = builder.close().then(() => 'resolved', error => error)
  await vi.advanceTimersByTimeAsync(5_000)

  expect(await result).toBeInstanceOf(Error)
  expect(String(await result)).toContain('exit')
  expect(child.exitCode).toBeNull()
  expect(child.signals).toEqual(['SIGTERM'])
})

test('close resolves immediately for an already-exited owned child', async () => {
  const child = new FakeChild()
  const builder = await readyBuilder(child)
  child.exit()
  let finished = false
  void builder.close().then(() => { finished = true })
  await vi.advanceTimersByTimeAsync(0)

  expect(finished).toBe(true)
  expect(child.signals).toEqual([])
})

test('concurrent close calls both wait for observed owned-child exit', async () => {
  const child = new FakeChild()
  const builder = await readyBuilder(child)
  const first = builder.close()
  let secondFinished = false
  const second = builder.close().then(() => { secondFinished = true })
  await vi.advanceTimersByTimeAsync(0)

  expect(secondFinished).toBe(false)
  child.exit()
  await Promise.all([first, second])
  expect(secondFinished).toBe(true)
  expect(child.signals).toEqual([])
})

import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createLinkTitleRenderQueue } from './link-title-render-queue'

test('renderer queue wait consumes the absolute renderer deadline', async () => {
  let finishFirst: ((value: string) => void) | undefined

  const run = vi.fn(
    () =>
      new Promise<string>(resolve => {
        finishFirst = resolve
      })
  )

  const queue = createLinkTitleRenderQueue({ concurrency: 1, run, timeoutMs: 25 })

  const first = queue.enqueue('https://first.example/')
  const second = queue.enqueue('https://second.example/')

  assert.equal(await second, '')
  assert.equal(run.mock.calls.length, 1)

  finishFirst?.('First title')
  assert.equal(await first, 'First title')
  assert.equal(run.mock.calls.length, 1)
})

test('closing the renderer queue settles queued and future work without starting it', async () => {
  let finishFirst: ((value: string) => void) | undefined

  const run = vi.fn(
    () =>
      new Promise<string>(resolve => {
        finishFirst = resolve
      })
  )

  const queue = createLinkTitleRenderQueue({ concurrency: 1, run, timeoutMs: 1_000 })

  const first = queue.enqueue('https://first.example/')
  const queued = queue.enqueue('https://queued.example/')

  queue.close()

  assert.equal(await queued, '')
  assert.equal(await queue.enqueue('https://future.example/'), '')
  assert.equal(run.mock.calls.length, 1)

  finishFirst?.('First title')
  assert.equal(await first, 'First title')
})

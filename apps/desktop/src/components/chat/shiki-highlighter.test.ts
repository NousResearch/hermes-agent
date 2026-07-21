import { afterEach, describe, expect, it, vi } from 'vitest'

import { chunkByLines, exceedsHighlightBudget } from '@/components/chat/shiki-highlighter'
import { startShikiHighlight } from '@/components/chat/shiki-worker-client'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('exceedsHighlightBudget', () => {
  it('highlights normal-sized blocks', () => {
    expect(exceedsHighlightBudget('const x = 1\n'.repeat(100))).toBe(false)
  })

  it('skips highlighting past the line budget', () => {
    expect(exceedsHighlightBudget('x\n'.repeat(5_000))).toBe(true)
  })

  it('skips highlighting past the char budget on few lines', () => {
    expect(exceedsHighlightBudget('a'.repeat(200_000))).toBe(true)
  })

  it('short-circuits on char budget before line loop', () => {
    expect(exceedsHighlightBudget('y\n'.repeat(250_000))).toBe(true)
  })
})

describe('chunkByLines', () => {
  it('keeps a small block as a single chunk', () => {
    const code = 'a\nb\nc'
    expect(chunkByLines(code, 200)).toEqual([{ text: code, lines: 3 }])
  })

  it('splits a large block and reconstructs it losslessly', () => {
    const code = Array.from({ length: 1000 }, (_, i) => `line ${i}`).join('\n')
    const chunks = chunkByLines(code, 200)

    expect(chunks).toHaveLength(5)
    expect(chunks.map(chunk => chunk.text).join('\n')).toBe(code)
    expect(chunks.reduce((sum, chunk) => sum + chunk.lines, 0)).toBe(1000)
  })
})

describe('Shiki worker client lifecycle', () => {
  it('shares one worker and terminates it after the final consumer disposes', async () => {
    class TestWorker {
      static instances: TestWorker[] = []
      messages: Array<{ code: string; id: number; language: string }> = []
      onerror: ((event: ErrorEvent) => void) | null = null
      onmessage: ((event: MessageEvent) => void) | null = null
      terminated = false

      constructor() {
        TestWorker.instances.push(this)
      }

      postMessage(message: { code: string; id: number; language: string }) {
        this.messages.push(message)
      }

      terminate() {
        this.terminated = true
      }
    }

    vi.stubGlobal('Worker', TestWorker)
    const first = startShikiHighlight('const a = 1', 'ts')
    const second = startShikiHighlight('print(1)', 'python')
    const worker = TestWorker.instances[0]

    expect(TestWorker.instances).toHaveLength(1)
    expect(worker.messages).toHaveLength(2)

    for (const message of worker.messages) {
      worker.onmessage?.({ data: { id: message.id, tokens: [[{ content: message.code }]] } } as MessageEvent)
    }

    await expect(first.promise).resolves.toEqual([[{ content: 'const a = 1' }]])
    await expect(second.promise).resolves.toEqual([[{ content: 'print(1)' }]])

    first.dispose()
    expect(worker.terminated).toBe(false)
    second.dispose()
    expect(worker.terminated).toBe(true)
  })

  it('returns a plain-code-compatible rejection when Worker is unavailable', async () => {
    vi.stubGlobal('Worker', undefined)
    const job = startShikiHighlight('const a = 1', 'ts')

    await expect(job.promise).rejects.toThrow('Web Workers are unavailable')
    expect(() => job.dispose()).not.toThrow()
  })

  it('rejects pending jobs and recreates the worker after a runtime error', async () => {
    class FailingWorker {
      static instances: FailingWorker[] = []
      messages: Array<{ code: string; id: number; language: string }> = []
      onerror: ((event: ErrorEvent) => void) | null = null
      onmessage: ((event: MessageEvent) => void) | null = null

      constructor() {
        FailingWorker.instances.push(this)
      }

      postMessage(message: { code: string; id: number; language: string }) {
        this.messages.push(message)
      }

      terminate() {}
    }

    vi.stubGlobal('Worker', FailingWorker)
    const first = startShikiHighlight('const a = 1', 'ts')
    const second = startShikiHighlight('print(1)', 'python')
    const failedWorker = FailingWorker.instances[0]
    const firstRejection = expect(first.promise).rejects.toThrow('worker crashed')
    const secondRejection = expect(second.promise).rejects.toThrow('worker crashed')

    failedWorker.onerror?.({ message: 'worker crashed' } as ErrorEvent)
    await Promise.all([firstRejection, secondRejection])
    first.dispose()
    second.dispose()

    const recovered = startShikiHighlight('echo ok', 'bash')
    const recoveredWorker = FailingWorker.instances[1]
    const message = recoveredWorker.messages[0]

    expect(FailingWorker.instances).toHaveLength(2)
    recoveredWorker.onmessage?.({ data: { id: message.id, tokens: [[{ content: message.code }]] } } as MessageEvent)
    await expect(recovered.promise).resolves.toEqual([[{ content: 'echo ok' }]])
    recovered.dispose()
  })
})

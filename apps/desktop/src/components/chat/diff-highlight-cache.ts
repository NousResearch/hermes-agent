import type { GrammarState, ThemedToken } from 'shiki'

export interface DiffHighlightInput {
  code: string
  grammarState?: GrammarState
  language: string
  theme: string
}

export interface DiffHighlightResult {
  grammarState?: GrammarState
  tokens: ThemedToken[][]
}

export interface DiffHighlightRequest {
  promise: Promise<DiffHighlightResult>
  release: () => void
}

type Highlight = (input: DiffHighlightInput, signal: AbortSignal) => Promise<DiffHighlightResult>
type Schedule = (task: () => void) => void

interface DiffHighlightCacheOptions {
  highlight: Highlight
  maxCompleted?: number
  schedule?: Schedule
}

interface InFlightHighlight {
  controller: AbortController
  key: string
  observers: number
  promise: Promise<DiffHighlightResult>
  reject: (reason?: unknown) => void
}

function abortError(): Error {
  const error = new Error('Highlight request no longer has an observer')

  error.name = 'AbortError'

  return error
}

export class DiffHighlightCache {
  private readonly completed = new Map<string, DiffHighlightResult>()
  private readonly grammarStateIds = new WeakMap<GrammarState, number>()
  private readonly highlight: Highlight
  private readonly inFlight = new Map<string, InFlightHighlight>()
  private readonly maxCompleted: number
  private nextGrammarStateId = 1
  private readonly schedule: Schedule

  constructor({
    highlight,
    maxCompleted = 64,
    schedule = task => void setTimeout(task, 0)
  }: DiffHighlightCacheOptions) {
    this.highlight = highlight
    this.maxCompleted = Math.max(0, maxCompleted)
    this.schedule = schedule
  }

  request(input: DiffHighlightInput): DiffHighlightRequest {
    const key = this.cacheKey(input)
    const completed = this.completed.get(key)

    if (completed) {
      // Map insertion order gives us a compact LRU: move cache hits to the end.
      this.completed.delete(key)
      this.completed.set(key, completed)

      return { promise: Promise.resolve(completed), release: () => undefined }
    }

    const current = this.inFlight.get(key)

    if (current) {
      current.observers += 1

      return this.handle(current)
    }

    let resolve!: (result: DiffHighlightResult) => void
    let reject!: (reason?: unknown) => void

    const promise = new Promise<DiffHighlightResult>((onResolve, onReject) => {
      resolve = onResolve
      reject = onReject
    })

    const entry: InFlightHighlight = {
      controller: new AbortController(),
      key,
      observers: 1,
      promise,
      reject
    }

    this.inFlight.set(key, entry)
    this.schedule(() => {
      // A window can move again before this yielded task gets CPU time. Do not
      // tokenize a chunk that every observer has already released.
      if (entry.controller.signal.aborted) {
        if (this.inFlight.get(key) === entry) {
          this.inFlight.delete(key)
        }

        return
      }

      let work: Promise<DiffHighlightResult>

      try {
        work = this.highlight(input, entry.controller.signal)
      } catch (error) {
        if (this.inFlight.get(key) === entry) {
          this.inFlight.delete(key)
        }

        reject(error)

        return
      }

      void work.then(
        result => {
          if (this.inFlight.get(key) === entry) {
            this.inFlight.delete(key)
          }

          if (entry.controller.signal.aborted) {
            reject(abortError())

            return
          }

          if (this.maxCompleted > 0) {
            this.completed.set(key, result)

            while (this.completed.size > this.maxCompleted) {
              const oldest = this.completed.keys().next().value

              if (oldest === undefined) {
                break
              }

              this.completed.delete(oldest)
            }
          }

          resolve(result)
        },
        error => {
          if (this.inFlight.get(key) === entry) {
            this.inFlight.delete(key)
          }

          reject(error)
        }
      )
    })

    return this.handle(entry)
  }

  private cacheKey({ code, grammarState, language, theme }: DiffHighlightInput): string {
    let grammarStateId = 0

    if (grammarState) {
      const known = this.grammarStateIds.get(grammarState)

      if (known) {
        grammarStateId = known
      } else {
        grammarStateId = this.nextGrammarStateId
        this.nextGrammarStateId += 1
        this.grammarStateIds.set(grammarState, grammarStateId)
      }
    }

    return `${language}\u0000${theme}\u0000${grammarStateId}\u0000${code}`
  }

  private handle(entry: InFlightHighlight): DiffHighlightRequest {
    let released = false

    return {
      promise: entry.promise,
      release: () => {
        if (!released) {
          released = true
          entry.observers -= 1

          if (entry.observers === 0) {
            entry.controller.abort()

            if (this.inFlight.get(entry.key) === entry) {
              this.inFlight.delete(entry.key)
            }

            entry.reject(abortError())
          }
        }
      }
    }
  }
}

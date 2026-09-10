export type BackendConnectionAttempt<TConnection> = {
  generation: number
  promise: Promise<TConnection> | null
}

export type BackendProcessOwner<TProcess> = {
  generation: number
  process: TProcess
}

interface PendingBackendStop<TProcess> {
  process: TProcess
  completion: Promise<void>
  failed: boolean
}

export function createBackendConnectionState<TProcess, TConnection>() {
  let generation = 0
  let process: TProcess | null = null
  let promise: Promise<TConnection> | null = null
  let stopping: PendingBackendStop<TProcess> | null = null

  function invalidate(): TProcess | null {
    const currentProcess = process
    generation += 1
    process = null
    promise = null

    return currentProcess
  }

  return {
    startAttempt(): BackendConnectionAttempt<TConnection> {
      if (stopping) {
        throw new Error('The previous backend has not stopped. Retry its shutdown before starting a replacement.')
      }

      return { generation, promise: null }
    },

    setPromise(attempt: BackendConnectionAttempt<TConnection>, nextPromise: Promise<TConnection>): boolean {
      if (attempt.generation !== generation) {
        return false
      }

      attempt.promise = nextPromise
      promise = nextPromise

      return true
    },

    isCurrentAttempt(attempt: BackendConnectionAttempt<TConnection>): boolean {
      return attempt.generation === generation
    },

    attachProcess(
      attempt: BackendConnectionAttempt<TConnection>,
      nextProcess: TProcess
    ): BackendProcessOwner<TProcess> | null {
      if (attempt.generation !== generation) {
        return null
      }

      process = nextProcess

      return { generation, process: nextProcess }
    },

    async claimProcess(
      attempt: BackendConnectionAttempt<TConnection>,
      nextProcess: TProcess,
      claim: (current: TProcess) => Promise<unknown>
    ): Promise<BackendProcessOwner<TProcess> | null> {
      const owner = this.attachProcess(attempt, nextProcess)

      if (!owner) {
        return null
      }

      await claim(nextProcess)

      return owner.generation === generation && process === nextProcess ? owner : null
    },

    clearForCurrentProcess(owner: BackendProcessOwner<TProcess>): boolean {
      if (owner.generation !== generation || owner.process !== process) {
        return false
      }

      process = null
      promise = null

      return true
    },

    clearPromiseForAttempt(attempt: BackendConnectionAttempt<TConnection>): boolean {
      if (attempt.generation !== generation || (promise !== null && attempt.promise !== promise)) {
        return false
      }

      promise = null

      return true
    },

    getProcess(): TProcess | null {
      return process
    },

    getPromise(): Promise<TConnection> | null {
      return promise
    },

    invalidate,

    stopProcess(stop: (current: TProcess) => Promise<void>): Promise<void> {
      if (stopping && !stopping.failed) {
        return stopping.completion
      }

      const current = stopping?.process ?? invalidate()

      if (current === null) {
        return Promise.resolve()
      }

      const completion = Promise.resolve().then(() => stop(current)).then(
        () => { stopping = null },
        error => { stopping!.failed = true; throw error }
      )

      stopping = { process: current, completion, failed: false }

      return completion
    }
  }
}

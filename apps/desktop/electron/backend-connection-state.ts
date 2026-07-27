export type BackendConnectionAttempt<TConnection> = {
  generation: number
  promise: Promise<TConnection> | null
}

export type BackendProcessOwner<TProcess> = {
  generation: number
  process: TProcess
}

export function createBackendConnectionState<TProcess, TConnection>() {
  let generation = 0
  let process: TProcess | null = null
  let promise: Promise<TConnection> | null = null

  return {
    startAttempt(): BackendConnectionAttempt<TConnection> {
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

    invalidate(): TProcess | null {
      const currentProcess = process

      generation += 1
      process = null
      promise = null

      return currentProcess
    }
  }
}

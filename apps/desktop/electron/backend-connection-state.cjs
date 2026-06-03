function createBackendConnectionState() {
  let generation = 0
  let process = null
  let promise = null

  return {
    attachProcess(nextProcess) {
      process = nextProcess

      return { generation, process: nextProcess }
    },

    clearForCurrentProcess(owner) {
      if (!owner || owner.generation !== generation || owner.process !== process) {
        return false
      }

      process = null
      promise = null

      return true
    },

    clearPromiseForAttempt(attempt) {
      if (!attempt || attempt.generation !== generation || attempt.promise !== promise) {
        return false
      }

      promise = null

      return true
    },

    clearPromise() {
      promise = null
    },

    getProcess() {
      return process
    },

    getPromise() {
      return promise
    },

    invalidate() {
      generation += 1
      process = null
      promise = null

      return generation
    },

    setPromise(attempt, nextPromise) {
      if (attempt) {
        attempt.promise = nextPromise
      }

      promise = nextPromise

      return nextPromise
    },

    startAttempt() {
      return { generation, promise: null }
    }
  }
}

module.exports = { createBackendConnectionState }

export function createBootstrapInstallCoordinator<T>() {
  let active: { controller: AbortController; promise: Promise<T> } | null = null
  let terminationFailure: unknown = null

  function assertTerminationVerified(): void {
    if (terminationFailure !== null) {throw terminationFailure}
  }

  async function run(start: (signal: AbortSignal) => Promise<T>, assertCurrent: () => void = () => {}): Promise<T> {
    while (active?.controller.signal.aborted) {
      await active.promise.catch(() => undefined)
    }

    assertTerminationVerified()
    assertCurrent()

    if (active) {return active.promise}

    const controller = new AbortController()
    let promise: Promise<T>

    try {
      promise = Promise.resolve(start(controller.signal))
    } catch (error) {
      promise = Promise.reject(error)
    }

    const entry = { controller, promise }
    active = entry
    void promise.then(
      () => { if (active === entry) {active = null} },
      error => {
        if (error && typeof error === 'object' && error.installTerminationUnverified === true) {
          terminationFailure = error
        }

        if (active === entry) {active = null}
      }
    )

    return promise
  }

  function cancel(): boolean {
    if (!active) {return false}
    active.controller.abort()

    return true
  }

  async function cancelAndWait(): Promise<void> {
    const entry = active

    if (entry) {
      entry.controller.abort()
      await entry.promise.catch(() => undefined)
    }

    assertTerminationVerified()
  }

  async function waitForSafeRuntime(): Promise<void> {
    while (active) {
      await active.promise.catch(() => undefined)
    }

    assertTerminationVerified()
  }

  return { run, cancel, cancelAndWait, waitForSafeRuntime }
}

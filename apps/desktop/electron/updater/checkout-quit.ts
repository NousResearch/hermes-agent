interface QuitEvent {
  preventDefault: () => void
}

interface CheckoutQuitDeps<Event extends QuitEvent> {
  preflightStateDb: (home: string, log: (message: string) => void) => Promise<void>
  heldQuitForActiveWork: (event: Event) => boolean
  teardown: (event: Event) => void
  quit: () => void
}

/** The checkout preflight and Electron before-quit share one lifecycle owner. */
export function createCheckoutQuitLifecycle<Event extends QuitEvent>(deps: CheckoutQuitDeps<Event>) {
  let pending: Promise<void> | null = null
  let quitting = false
  let waitingForPreflight = false
  let quitGeneration = 0

  const cancelled = (): Error => new Error('Update cancelled because Desktop is quitting.')

  return {
    async preflightStateDb(home: string, log: (message: string) => void): Promise<void> {
      if (quitting) {
        throw cancelled()
      }

      // Include config/launcher resolution, not just the snapshot child. Keep
      // ownership until the caller's continuation has been admitted or refused.
      const generation = quitGeneration
      pending = Promise.resolve().then(() => deps.preflightStateDb(home, log))

      try {
        await pending

        // A later Keep Running releases future work, not this cancelled update.
        if (generation !== quitGeneration) {
          throw cancelled()
        }
      } finally {
        pending = null
      }
    },
    beforeQuit(event: Event): void {
      // A newly held prompt revokes any earlier admitted quit. Keep Running
      // will not re-enter; Quit Anyway makes a new request through this seam.
      if (deps.heldQuitForActiveWork(event)) {
        quitting = false
        waitingForPreflight = false

        return
      }

      if (!quitting) {
        quitting = true
        quitGeneration += 1
      }

      if (pending) {
        event.preventDefault()

        if (!waitingForPreflight) {
          waitingForPreflight = true
          const generation = quitGeneration

          const resumeQuit = () => {
            if (quitting && generation === quitGeneration) {
              deps.quit()
            }
          }

          // Don't kill the snapshot or allow a stale updater handoff. Even a
          // failed snapshot may release an explicitly requested ordinary quit;
          // the update itself still rejects before any update teardown/spawn.
          void pending.then(resumeQuit, resumeQuit)
        }

        return
      }

      deps.teardown(event)
    }
  }
}

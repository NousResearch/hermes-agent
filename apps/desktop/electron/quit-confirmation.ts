interface QuitEvent {
  preventDefault: () => void
}

interface QuitConfirmationOptions {
  shouldSkip: () => boolean
  requestQuit: () => void
  /** Null means no confirmation is needed or no visible parent is available. */
  showPrompt: () => Promise<{ response: number }> | null
}

/** Keep cancellation ahead of irreversible backend teardown. */
export function createQuitConfirmation(options: QuitConfirmationOptions): (event: QuitEvent) => boolean {
  let promptOpen = false
  let admitted = false

  return event => {
    if (admitted || options.shouldSkip()) {
      admitted = true

      return false
    }

    if (promptOpen) {
      event.preventDefault()

      return true
    }

    const prompt = options.showPrompt()

    if (!prompt) {
      // An idle/no-window quit is just as final as an explicitly confirmed one.
      // Teardown seals SSH permanently, so a later work report cannot veto it.
      admitted = true

      return false
    }

    event.preventDefault()
    promptOpen = true

    void prompt
      .then(({ response }) => {
        promptOpen = false

        if (response === 1) {
          admitted = true
          options.requestQuit()
        }
      })
      .catch(() => {
        // Preserve the existing fallback when the native dialog fails.
        promptOpen = false
        admitted = true
        options.requestQuit()
      })

    return true
  }
}

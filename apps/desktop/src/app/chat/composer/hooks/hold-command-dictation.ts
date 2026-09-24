type HoldCommandAction = 'arm' | 'cancel' | 'ignore' | 'start' | 'stop'

type ModifierKeyEvent = Pick<KeyboardEvent, 'altKey' | 'ctrlKey' | 'key' | 'repeat' | 'shiftKey'>

/** Keeps hold-Command dictation distinct from ordinary Command shortcuts. */
export function createHoldCommandDictation() {
  let state: 'idle' | 'armed' | 'recording' = 'idle'

  const cancel = (): HoldCommandAction => {
    if (state === 'recording') {
      state = 'idle'

      return 'stop'
    }

    if (state === 'armed') {
      state = 'idle'

      return 'cancel'
    }

    return 'ignore'
  }

  return {
    begin(): HoldCommandAction {
      if (state !== 'armed') {return 'ignore'}
      state = 'recording'

      return 'start'
    },
    cancel,
    keyDown(event: ModifierKeyEvent): HoldCommandAction {
      if (event.key === 'Meta') {
        if (state === 'idle' && !event.repeat && !event.altKey && !event.ctrlKey && !event.shiftKey) {
          state = 'armed'

          return 'arm'
        }

        return 'ignore'
      }

      // A key pressed while Command is held is an ordinary shortcut, never a
      // dictation request. Stop too if a hold had already crossed its delay.
      return cancel()
    },
    keyUp(event: Pick<KeyboardEvent, 'key'>): HoldCommandAction {
      return event.key === 'Meta' ? cancel() : 'ignore'
    }
  }
}

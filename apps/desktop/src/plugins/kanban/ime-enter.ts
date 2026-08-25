export interface ImeKeyEvent {
  key: string
  nativeEvent: {
    isComposing?: boolean
    keyCode?: number
  }
}

/**
 * Enter confirms an IME conversion before it should act as a submit shortcut.
 * Chromium can report the legacy 229 keyCode around composition boundaries,
 * so keep that fallback in addition to the standard isComposing signal.
 */
export function shouldSubmitOnEnter(event: ImeKeyEvent): boolean {
  return event.key === 'Enter' && !event.nativeEvent.isComposing && event.nativeEvent.keyCode !== 229
}

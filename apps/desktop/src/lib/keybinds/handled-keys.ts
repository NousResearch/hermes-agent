// Keydowns the window keybind dispatcher ran as a Hermes action. It cancels
// every chord it matches in capture, including one whose passthrough handler
// then declines, so `defaultPrevented` alone doesn't say an action ran. xterm
// ignores `defaultPrevented`; the terminal asks here before sending a key to
// the PTY.
const handled = new WeakSet<Event>()

export function markKeybindHandled(event: Event): void {
  handled.add(event)
}

export function keybindHandled(event: Event): boolean {
  return handled.has(event)
}

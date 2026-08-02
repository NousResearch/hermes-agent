interface SelectionIpcRenderer {
  on: (channel: string, listener: (...args: unknown[]) => void) => void
  removeListener: (channel: string, listener: (...args: unknown[]) => void) => void
}

export function createIpcSelectionSubscription(ipc: SelectionIpcRenderer, channel: string) {
  return (callback: (text: string) => void) => {
    const listener = (_event: unknown, text: unknown) => callback(typeof text === 'string' ? text : '')

    ipc.on(channel, listener)

    return () => ipc.removeListener(channel, listener)
  }
}

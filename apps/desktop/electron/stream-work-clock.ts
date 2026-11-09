import type { BrowserWindow, IpcMain, WebContents } from 'electron'

interface WakeRequest {
  token: string
  delayMs: number | null
}

function isWakeRequest(value: unknown): value is WakeRequest {
  if (!value || typeof value !== 'object' || !('token' in value) || !('delayMs' in value)) {
    return false
  }

  return (
    typeof value.token === 'string' &&
    value.token.length > 0 &&
    value.token.length <= 80 &&
    (value.delayMs === null ||
      (typeof value.delayMs === 'number' &&
        Number.isFinite(value.delayMs) &&
        value.delayMs >= 0 &&
        value.delayMs <= 400))
  )
}

/** One pending functional wake per owning renderer; never enables painting. */
export function registerStreamWorkClock(
  ipc: Pick<IpcMain, 'on'>,
  getOwner: (sender: WebContents) => BrowserWindow | null
): void {
  const clocks = new WeakMap<
    WebContents,
    { cancel: () => void; token: string; timer?: ReturnType<typeof setTimeout> }
  >()

  ipc.on('hermes:stream-work-wake', (event, payload: unknown) => {
    const sender = event.sender
    const owner = getOwner(sender)

    if (
      !owner ||
      owner.isDestroyed() ||
      owner.webContents !== sender ||
      event.senderFrame !== sender.mainFrame ||
      !isWakeRequest(payload)
    ) {
      return
    }

    let clock = clocks.get(sender)

    if (!clock) {
      const state = {
        token: '',
        timer: undefined as ReturnType<typeof setTimeout> | undefined,
        cancel: () => {
          clearTimeout(state.timer)
          state.timer = undefined
          state.token = ''
        }
      }

      clock = state
      clocks.set(sender, state)
      sender.on('render-process-gone', state.cancel)
      sender.on('did-navigate', state.cancel)
      sender.once('destroyed', () => {
        state.cancel()
        sender.off('render-process-gone', state.cancel)
        sender.off('did-navigate', state.cancel)
        clocks.delete(sender)
      })
    }

    if (payload.delayMs === null) {
      if (clock.token === payload.token) {
        clock.cancel()
      }

      return
    }

    clock.cancel()
    clock.token = payload.token
    const current = clock
    const frame = event.senderFrame
    current.timer = setTimeout(() => {
      current.timer = undefined
      current.token = ''

      if (frame && !sender.isDestroyed() && !frame.isDestroyed() && !frame.detached) {
        frame.send('hermes:stream-work-ready', payload.token)
      }
    }, Math.ceil(payload.delayMs))
  })
}

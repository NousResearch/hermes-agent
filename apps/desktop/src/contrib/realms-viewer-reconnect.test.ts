import { afterEach, expect, it, vi } from 'vitest'

const { clients } = vi.hoisted(() => ({
  clients: [] as Array<EventTarget & { viewOnly: boolean; disconnect: () => void }>
}))

vi.mock('../../../../plugins/hermes-realms/realms/web/vendor/novnc/core/rfb.js', () => ({
  default: class extends EventTarget {
    viewOnly = true
    constructor() {
      super()
      clients.push(this)
    }
    disconnect() {
      this.dispatchEvent(new Event('disconnect'))
    }
  }
}))

async function openViewer() {
  vi.useFakeTimers()
  vi.resetModules()
  document.body.innerHTML =
    '<div id="screen"></div><span id="state"></span><button id="control"></button><div id="error" hidden></div>'
  history.replaceState(null, '', '/realms/fixture/view#ticket=' + 'a'.repeat(43))
  // @ts-expect-error The bundled viewer is a plain JavaScript module.
  await import('../../../../plugins/hermes-realms/realms/web/viewer.js')
}

afterEach(() => {
  window.dispatchEvent(new Event('pagehide'))
  vi.clearAllTimers()
  vi.useRealTimers()
  clients.length = 0
  document.body.innerHTML = ''
})

it('reconnects the existing viewer in view-only mode without automatically reclaiming control', async () => {
  await openViewer()
  clients[0].dispatchEvent(new Event('connect'))
  document.querySelector<HTMLButtonElement>('#control')!.click()
  expect(clients[1].viewOnly).toBe(false)
  clients[1].dispatchEvent(new Event('connect'))
  clients[1].dispatchEvent(new Event('disconnect'))
  await vi.advanceTimersByTimeAsync(1000)
  expect(clients).toHaveLength(3)
  expect(clients[2].viewOnly).toBe(true)
  clients[2].dispatchEvent(new Event('connect'))
  expect(document.querySelector('#state')!.textContent).toBe('Live · view only')
  clients[2].dispatchEvent(new Event('disconnect'))
  window.dispatchEvent(new Event('pagehide'))
  await vi.advanceTimersByTimeAsync(30000)
  expect(clients).toHaveLength(3)
})

it('bounds failed reconnects and leaves authorization failures disconnected', async () => {
  await openViewer()

  for (let attempt = 0; attempt < 6; attempt++) {
    clients.at(-1)!.dispatchEvent(new Event('disconnect'))
    await vi.advanceTimersByTimeAsync(30000)
  }

  const finalCount = clients.length
  await vi.advanceTimersByTimeAsync(300000)
  expect(clients.length).toBe(finalCount)
  expect(document.querySelector('#state')!.textContent).toBe('Disconnected')
  expect(document.querySelector('#error')!.textContent).toContain('Watch')
})

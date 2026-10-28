import { vi } from 'vitest'

import { announceWebappWindow } from '@/lib/browser-window'

/**
 * BroadcastChannel as a browser delivers it within one origin: asynchronously, to every other open
 * channel with the same name, never to the sender, and never after either side closes.
 */
class Channel {
  static open = new Set<Channel>()
  onmessage: ((event: MessageEvent) => void) | null = null

  constructor(readonly name: string) {
    Channel.open.add(this)
  }

  postMessage(data: unknown) {
    if (!Channel.open.has(this)) {
      throw new DOMException('BroadcastChannel is closed', 'InvalidStateError')
    }

    for (const peer of Channel.open) {
      if (peer !== this && peer.name === this.name) {
        queueMicrotask(() => {
          if (Channel.open.has(peer)) {
            peer.onmessage?.(new MessageEvent('message', { data }))
          }
        })
      }
    }
  }

  close() {
    Channel.open.delete(this)
  }
}

/** Every open channel, for asserting what is still listening. */
export function stubBroadcastChannels(): ReadonlySet<Channel> {
  Channel.open.clear()
  vi.stubGlobal('BroadcastChannel', Channel)

  return Channel.open
}

/**
 * The browser's side of a Webapp pop-out. As in a browser, `window.open` returns null when the
 * popup blocker stops the window, and for a `noopener` or `noreferrer` open even when the window
 * opens. Unless `blocked`, each window opens and boots the Webapp at its address, which runs the
 * boot hello (`announceWebappWindow`). This test has one jsdom window, so the boot borrows its
 * address and puts it back.
 */
export function browserOpensWebappWindows({ blocked = false } = {}) {
  const channels = stubBroadcastChannels()

  const open = vi.spyOn(window, 'open').mockImplementation((url, _target, features) => {
    if (blocked) {
      return null
    }

    queueMicrotask(() => {
      const opener = window.location.href
      window.history.replaceState(null, '', String(url))
      announceWebappWindow()
      window.history.replaceState(null, '', opener)
    })

    return /noopener|noreferrer/.test(features ?? '') ? null : ({ opener: window } as Window)
  })

  return { channels, open }
}

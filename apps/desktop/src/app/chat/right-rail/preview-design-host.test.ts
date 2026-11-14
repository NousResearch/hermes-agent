import {
  type DesignSelection,
  PREVIEW_DESIGN_COMMAND_CHANNEL,
  PREVIEW_DESIGN_EVENT_CHANNEL,
  type PreviewDesignBridgeEvent,
  type PreviewDesignCommand
} from '@hermes/shared/preview-design'
import { describe, expect, it, vi } from 'vitest'

import { installGuestDesignBridge } from '../../../../electron/preview-guest-preload'

import { installDesignOverlay, type PreviewDesignGuest } from './preview-design-host'

class FakeGuest extends EventTarget implements PreviewDesignGuest {
  readonly commands: PreviewDesignCommand[] = []

  send(command: PreviewDesignCommand): void {
    this.commands.push(command)

    if (command.type === 'start') {
      queueMicrotask(() => this.emit({ session: command.session + 1, type: 'ready' }))
      queueMicrotask(() => this.emit({ session: command.session, type: 'ready' }))
    } else if (command.type === 'stop') {
      queueMicrotask(() => this.emit({ session: command.session, type: 'stopped' }))
    }
  }

  emit(payload: PreviewDesignBridgeEvent, channel = PREVIEW_DESIGN_EVENT_CHANNEL): void {
    const event = new Event('ipc-message') as Event & { args?: unknown[]; channel?: string }
    event.channel = channel
    event.args = [payload]
    this.dispatchEvent(event)
  }
}

const selection: DesignSelection = {
  css: { 'background-color': 'rgb(255, 255, 255)', color: 'rgb(0, 0, 0)', 'font-size': '16px' },
  layers: [
    { segment: 'body', tag: 'body' },
    { segment: 'button:nth-of-type(1)', tag: 'button' }
  ],
  rect: { height: 30, width: 80, x: 10, y: 20 },
  selectedLayer: 1,
  selector: 'body > button:nth-of-type(1)',
  tag: 'button'
}

describe('Preview Design isolated guest host', () => {
  it('keeps the controller out of the page world and scopes preload commands to the active session', async () => {
    const frame = document.createElement('iframe')
    document.body.appendChild(frame)

    try {
      const doc = frame.contentDocument!
      const button = doc.createElement('button')
      doc.body.appendChild(button)
      doc.elementFromPoint = () => button
      const listeners = new Map<string, (_event: unknown, payload: unknown) => void>()
      const events: PreviewDesignBridgeEvent[] = []

      installGuestDesignBridge(doc, {
        on: (channel, listener) => listeners.set(channel, listener),
        sendToHost: (channel, payload) => {
          expect(channel).toBe(PREVIEW_DESIGN_EVENT_CHANNEL)
          events.push(payload)
        }
      })

      const send = (payload: PreviewDesignCommand) => listeners.get(PREVIEW_DESIGN_COMMAND_CHANNEL)?.({}, payload)
      send({ session: 41, type: 'start' })
      await vi.waitFor(() => expect(events).toContainEqual({ session: 41, type: 'ready' }))
      expect((frame.contentWindow as unknown as Record<string, unknown>).__hermesDesign).toBeUndefined()
      const outline = doc.querySelector('hermes-design')!.shadowRoot!.querySelector<HTMLElement>('[data-design-selected]')!
      expect(outline.style.display).toBe('none')

      button.dispatchEvent(new MouseEvent('click', { bubbles: true, button: 0, cancelable: true }))
      await vi.waitFor(() => expect(events.some(event => event.type === 'select' && event.session === 41)).toBe(true))

      send({ session: 40, type: 'stop' })
      await Promise.resolve()
      expect(events).not.toContainEqual({ session: 40, type: 'stopped' })
      send({ session: 41, type: 'stop' })
      await vi.waitFor(() => expect(events).toContainEqual({ session: 41, type: 'stopped' }))
      expect(doc.querySelector('hermes-design')).toBeNull()
    } finally {
      frame.remove()
    }
  })

  it('ignores stale sessions, validates guest events, and routes layer selection without page script injection', async () => {
    const guest = new FakeGuest()
    const lifetime = new AbortController()
    const session = await installDesignOverlay(guest, lifetime.signal)
    const start = guest.commands[0]

    expect(start).toMatchObject({ type: 'start' })

    if (!start) {
      throw new Error('Expected a start command')
    }

    guest.emit({ selection, session: start.session + 1, type: 'select' })
    guest.emit({ selection, session: start.session, type: 'select' })
    await expect(session.wait()).resolves.toEqual({ selection, type: 'select' })
    expect(session.selectLayer(1)).toBe(true)
    expect(guest.commands.at(-1)).toEqual({ index: 1, session: start.session, type: 'select-layer' })
    expect(session.selectLayer(16)).toBe(false)
    await session.stop()
    expect(guest.commands.at(-1)).toEqual({ session: start.session, type: 'stop' })
  })

  it('ignores unrelated ipc channels and rejects malformed DOM context', async () => {
    const unrelated = new FakeGuest()

    unrelated.send = command => {
      unrelated.commands.push(command)

      if (command.type === 'start') {
        queueMicrotask(() => unrelated.emit({ session: command.session, type: 'ready' }, 'page-controlled-channel'))
        queueMicrotask(() => unrelated.emit({ session: command.session, type: 'ready' }))
      } else if (command.type === 'stop') {
        queueMicrotask(() => unrelated.emit({ session: command.session, type: 'stopped' }))
      }
    }

    const session = await installDesignOverlay(unrelated, new AbortController().signal)
    await session.stop()

    const malformed = new FakeGuest()

    malformed.send = command => {
      malformed.commands.push(command)

      if (command.type === 'start') {
        queueMicrotask(() => {
          const event = new Event('ipc-message') as Event & { args?: unknown[]; channel?: string }
          event.channel = PREVIEW_DESIGN_EVENT_CHANNEL
          event.args = [{ extra: true, session: command.session, type: 'ready' }]
          malformed.dispatchEvent(event)
        })
      } else if (command.type === 'stop') {
        queueMicrotask(() => malformed.emit({ session: command.session, type: 'stopped' }))
      }
    }

    await expect(installDesignOverlay(malformed, new AbortController().signal)).rejects.toThrow(
      'invalid DOM context'
    )
  })
})

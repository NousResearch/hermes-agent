/**
 * Unit tests for the Hyprland window provider. The socket itself needs a live
 * compositor, so what is covered here is everything that decides whether the
 * answer is right once the bytes arrive: which instance we talk to, and the two
 * corrections that turn `j/clients` into a front-to-back list of the windows
 * actually on screen.
 */

import { describe, expect, it } from 'vitest'

import { hyprlandSocketPath, parseHyprlandClients } from './hyprland'

const SELF_PID = 42

const client = (over: Record<string, unknown> = {}) => ({
  address: '0x55d1',
  at: [0, 0],
  class: 'app',
  focusHistoryID: 0,
  pid: 1,
  size: [800, 600],
  title: 'a window',
  workspace: { id: 1 },
  ...over
})

const payload = (...clients: Array<Record<string, unknown>>) => JSON.stringify(clients)

describe('hyprlandSocketPath', () => {
  it('is null when Hyprland is not the compositor', () => {
    expect(hyprlandSocketPath({}, 1000)).toBeNull()
    expect(hyprlandSocketPath({ XDG_RUNTIME_DIR: '/run/user/1000' }, 1000)).toBeNull()
  })

  it('addresses the running instance, not just the user', () => {
    const path = hyprlandSocketPath({ HYPRLAND_INSTANCE_SIGNATURE: 'abc123', XDG_RUNTIME_DIR: '/run/user/1000' }, 1000)

    expect(path).toBe('/run/user/1000/hypr/abc123/.socket.sock')
  })

  it('falls back to the uid runtime dir when XDG_RUNTIME_DIR is unset', () => {
    const path = hyprlandSocketPath({ HYPRLAND_INSTANCE_SIGNATURE: 'abc123' }, 1000)

    expect(path).toBe('/run/user/1000/hypr/abc123/.socket.sock')
  })
})

describe('parseHyprlandClients', () => {
  it('orders by focus history, so the window the user came from is first', () => {
    const parsed = parseHyprlandClients(
      payload(
        client({ class: 'kitty', focusHistoryID: 3, pid: 3 }),
        client({ class: 'firefox', focusHistoryID: 1, pid: 2 }),
        client({ class: 'spotify', focusHistoryID: 2, pid: 4 })
      ),
      SELF_PID
    )

    expect(parsed.map(w => w.app)).toEqual(['firefox', 'spotify', 'kitty'])
  })

  // The HUD floats always-on-top while the user works underneath it, so it sits
  // well down a focus-ordered list. Keeping it in would make the caller skip
  // past the focused app — the very window it is trying to report.
  it('leaves our own windows out, even when the HUD itself is focused', () => {
    const parsed = parseHyprlandClients(
      payload(
        client({ class: 'hermes', focusHistoryID: 0, pid: SELF_PID }),
        client({ class: 'firefox', focusHistoryID: 1, pid: 2 })
      ),
      SELF_PID
    )

    expect(parsed.map(w => w.app)).toEqual(['firefox'])
  })

  // Windows on a hidden workspace share coordinates with the visible ones, so
  // left in they win the overlap test and the tool reports a window the user
  // cannot see.
  it('drops windows on workspaces other than ours', () => {
    const parsed = parseHyprlandClients(
      payload(
        client({ class: 'hermes', pid: SELF_PID, workspace: { id: 2 } }),
        client({ class: 'visible', focusHistoryID: 1, pid: 2, workspace: { id: 2 } }),
        client({ class: 'elsewhere', focusHistoryID: 1, pid: 3, workspace: { id: 7 } })
      ),
      SELF_PID
    )

    expect(parsed.map(w => w.app)).toEqual(['visible'])
  })

  it('keeps everything when it cannot find our own window', () => {
    const parsed = parseHyprlandClients(
      payload(
        client({ class: 'a', pid: 2, workspace: { id: 1 } }),
        client({ class: 'b', pid: 3, workspace: { id: 7 } })
      ),
      SELF_PID
    )

    expect(parsed.map(w => w.app)).toEqual(['a', 'b'])
  })

  it('maps position and size into bounds', () => {
    const [window] = parseHyprlandClients(payload(client({ at: [120, 80], pid: 2, size: [640, 320] })), SELF_PID)

    expect(window.bounds).toEqual({ x: 120, y: 80, width: 640, height: 320 })
  })

  it('skips zero-sized windows', () => {
    const parsed = parseHyprlandClients(payload(client({ pid: 2, size: [0, 0] })), SELF_PID)

    expect(parsed).toEqual([])
  })

  it('survives a payload that is not the client list', () => {
    for (const bad of ['', 'not json', '{}', 'null']) {
      expect(parseHyprlandClients(bad, SELF_PID)).toEqual([])
    }
  })
})

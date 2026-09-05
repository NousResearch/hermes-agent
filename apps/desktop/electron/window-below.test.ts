import { describe, expect, it } from 'vitest'

import { type EnumeratedWindow, enumerationFailureNote, pickWindowBelow, resolveOutsideAsar } from './window-below'

const win = (pid: number, x = 0, y = 0, width = 800, height = 600, app = `app-${pid}`): EnumeratedWindow => ({
  app,
  bounds: { x, y, width, height },
  id: pid * 10,
  pid,
  title: `${app} window`
})

const SELF_PID = 42
const SELF_BOUNDS = { x: 100, y: 100, width: 800, height: 600 }

describe('pickWindowBelow', () => {
  it('picks the first overlapping window behind ours in z-order', () => {
    const chrome = win(1, 120, 120)
    const spotify = win(2, 130, 130)

    const { below, frontmost } = pickWindowBelow([win(SELF_PID, 100, 100), chrome, spotify], SELF_PID, SELF_BOUNDS)

    expect(below).toBe(chrome)
    expect(frontmost).toBe(chrome)
  })

  it('skips windows behind ours that do not overlap', () => {
    const elsewhere = win(1, 5000, 5000)
    const covered = win(2, 200, 200)

    const { below } = pickWindowBelow([win(SELF_PID, 100, 100), elsewhere, covered], SELF_PID, SELF_BOUNDS)

    expect(below).toBe(covered)
  })

  it('skips our own other windows (same pid) while walking down', () => {
    const secondHermesWindow = win(SELF_PID, 150, 150)
    const target = win(7, 160, 160)

    const { below } = pickWindowBelow([win(SELF_PID, 100, 100), secondHermesWindow, target], SELF_PID, SELF_BOUNDS)

    expect(below).toBe(target)
  })

  it('reports frontmost even when nothing overlaps', () => {
    const elsewhere = win(1, 5000, 5000)

    const { below, frontmost } = pickWindowBelow([win(SELF_PID, 100, 100), elsewhere], SELF_PID, SELF_BOUNDS)

    expect(below).toBeNull()
    expect(frontmost).toBe(elsewhere)
  })

  it('windows in front of ours are never "below", even overlapping', () => {
    const inFront = win(3, 110, 110)
    const behind = win(4, 120, 120)

    const { below, frontmost } = pickWindowBelow([inFront, win(SELF_PID, 100, 100), behind], SELF_PID, SELF_BOUNDS)

    expect(below).toBe(behind)
    expect(frontmost).toBe(inFront)
  })

  it('falls back to overlap-only when our own window is not in the list', () => {
    // macOS omits windows the enumerator cannot see; still answer usefully.
    const chrome = win(1, 120, 120)
    const { below } = pickWindowBelow([chrome], SELF_PID, SELF_BOUNDS)

    expect(below).toBe(chrome)
  })

  it('returns nulls for an empty enumeration', () => {
    const { below, frontmost } = pickWindowBelow([], SELF_PID, SELF_BOUNDS)

    expect(below).toBeNull()
    expect(frontmost).toBeNull()
  })

  it('edge-adjacent bounds do not count as overlap', () => {
    const adjacent = win(1, 900, 100) // starts exactly at our right edge

    const { below } = pickWindowBelow([win(SELF_PID, 100, 100), adjacent], SELF_PID, SELF_BOUNDS)

    expect(below).toBeNull()
  })
})

describe('enumerationFailureNote', () => {
  it('tells a Wayland user the session is the problem', () => {
    for (const env of [{ XDG_SESSION_TYPE: 'wayland' }, { WAYLAND_DISPLAY: 'wayland-0' }]) {
      expect(enumerationFailureNote('linux', env)).toMatch(/Wayland/)
    }
  })

  // Hyprland is asked over its own IPC, so the X11 tooling advice would be a
  // wrong turn — reaching here means the compositor didn't answer.
  it('points a Hyprland user at their compositor, not at xprop', () => {
    const note = enumerationFailureNote('linux', { HYPRLAND_INSTANCE_SIGNATURE: 'abc', XDG_SESSION_TYPE: 'wayland' })

    expect(note).toMatch(/Hyprland/)
    expect(note).not.toMatch(/xprop|X11\/Xorg/)
  })

  it('tells an X11 user which commands are missing', () => {
    const note = enumerationFailureNote('linux', { XDG_SESSION_TYPE: 'x11', DISPLAY: ':0' })

    expect(note).toMatch(/xprop/)
    expect(note).not.toMatch(/Wayland/)
  })

  // XWayland can still answer through xprop, so the fix is the tooling, not
  // switching session type.
  it('treats Wayland with an X display as X11', () => {
    const note = enumerationFailureNote('linux', { WAYLAND_DISPLAY: 'wayland-0', DISPLAY: ':0' })

    expect(note).toMatch(/xprop/)
    expect(note).not.toMatch(/Wayland/)
  })

  it('does not offer Linux advice on other platforms', () => {
    for (const platform of ['darwin', 'win32']) {
      const note = enumerationFailureNote(platform, {})

      expect(note).not.toMatch(/xprop|Wayland/)
      expect(note.length).toBeGreaterThan(0)
    }
  })

  // The macOS/Windows note used to be one fixed sentence, so a real report came
  // back saying only "could not enumerate windows on this system" — the module
  // failing to load, the helper failing to spawn, and the OS answering with
  // nothing are three different fixes and all three said that.
  it('carries the enumerator\u2019s own reason where there is no environmental fork', () => {
    for (const platform of ['darwin', 'win32']) {
      expect(enumerationFailureNote(platform, {}, 'the helper failed: spawn EACCES')).toMatch(/spawn EACCES/)
    }
  })

  // Linux's notes name the fix (change session type, install xprop); the raw
  // exception underneath would only bury it.
  it('keeps the actionable Linux advice instead of the raw reason', () => {
    const note = enumerationFailureNote('linux', { XDG_SESSION_TYPE: 'x11', DISPLAY: ':0' }, 'ENOENT')

    expect(note).toMatch(/xprop/)
    expect(note).not.toMatch(/ENOENT/)
  })
})

describe('resolveOutsideAsar', () => {
  // The helper binary get-windows execs cannot run from inside the archive
  // (execFile on a path through app.asar fails ENOTDIR), so the import must
  // land on the unpacked copy electron-builder ships beside it.
  it('redirects a packaged specifier into app.asar.unpacked', () => {
    expect(
      resolveOutsideAsar('file:///Applications/Hermes.app/Contents/Resources/app.asar/dist/node_modules/get-windows/index.js')
    ).toBe('file:///Applications/Hermes.app/Contents/Resources/app.asar.unpacked/dist/node_modules/get-windows/index.js')
  })

  it('leaves a dev-tree specifier alone', () => {
    const dev = 'file:///Users/dev/hermes-agent/node_modules/get-windows/index.js'

    expect(resolveOutsideAsar(dev)).toBe(dev)
  })

  // Only the exact archive segment counts — a directory that merely starts
  // with the name must not be rewritten.
  it('requires app.asar to be a complete path segment', () => {
    const lookalike = 'file:///opt/app.asar-tools/node_modules/get-windows/index.js'

    expect(resolveOutsideAsar(lookalike)).toBe(lookalike)
  })
})

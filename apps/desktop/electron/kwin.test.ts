/**
 * Unit tests for the KWin window provider. The compositor itself needs a live
 * Plasma session, so what is covered here is everything that decides whether
 * the answer is right once the bytes arrive: whether this is KWin's session at
 * all, and the corrections that turn `workspace.stackingOrder` into a
 * front-to-back list of the windows actually on screen.
 *
 * A handful of these cases exist because the KWin scripting host is quiet about
 * its own limits — a script that fails to parse still reports a successful
 * `loadScript` and sends nothing, which from the outside looks exactly like a
 * desktop with no windows. The script text is asserted against the constructs
 * that are known to break it, so a later edit cannot quietly reintroduce the
 * worst possible failure mode: a confident empty answer.
 */

import { describe, expect, it } from 'vitest'

import { KWIN_PROBE_SCRIPT, kwinSessionLikely, parseKwinWindows } from './kwin'

const SELF_PID = 42

const kwinSession = {
  XDG_CURRENT_DESKTOP: 'KDE',
  XDG_SESSION_TYPE: 'wayland',
  WAYLAND_DISPLAY: 'wayland-0'
}

const row = (over: Record<string, unknown> = {}) => ({
  active: false,
  appletPopup: false,
  caption: 'a window',
  class: 'app',
  comboBox: false,
  criticalNotification: false,
  deleted: false,
  desktopWindow: false,
  desktops: ['d1'],
  dock: false,
  dndIcon: false,
  dropdownMenu: false,
  height: 600,
  hidden: false,
  id: '{3bda1724-941e-4e1b-94ba-9cf668cafd72}',
  menu: false,
  minimized: false,
  notification: false,
  onAllDesktops: false,
  onScreenDisplay: false,
  pid: 1,
  popupMenu: false,
  popupWindow: false,
  stackingOrder: 0,
  toolbar: false,
  tooltip: false,
  width: 800,
  x: 0,
  y: 0,
  ...over
})

const payload = (rows: Array<Record<string, unknown>>, current = 'd1') => JSON.stringify({ current, rows })

describe('kwinSessionLikely', () => {
  it('accepts a Plasma Wayland session', () => {
    expect(kwinSessionLikely(kwinSession)).toBe(true)
  })

  it('accepts the session without DISPLAY set, which is the usual Plasma case', () => {
    expect(kwinSessionLikely({ ...kwinSession, DISPLAY: undefined })).toBe(true)
  })

  it('rejects X11, where the X11 enumerator is already correct', () => {
    expect(kwinSessionLikely({ ...kwinSession, XDG_SESSION_TYPE: 'x11' })).toBe(false)
  })

  it('rejects another Wayland desktop', () => {
    expect(kwinSessionLikely({ ...kwinSession, XDG_CURRENT_DESKTOP: 'GNOME' })).toBe(false)
    expect(kwinSessionLikely({ XDG_SESSION_TYPE: 'wayland' })).toBe(false)
  })

  // Hyprland's provider answers first, so this path must not claim the session
  // and shadow it.
  it('rejects Hyprland even though it is a Wayland session', () => {
    expect(kwinSessionLikely({ ...kwinSession, HYPRLAND_INSTANCE_SIGNATURE: 'abc123' })).toBe(false)
  })
})

describe('parseKwinWindows', () => {
  // stackingOrder runs bottom-to-top, the caller wants front-to-back.
  it('reverses the stacking order, so the frontmost window is first', () => {
    const parsed = parseKwinWindows(
      payload([
        row({ class: 'bottom', pid: 2, stackingOrder: 0 }),
        row({ class: 'middle', pid: 3, stackingOrder: 1 }),
        row({ class: 'top', pid: 4, stackingOrder: 2 })
      ]),
      SELF_PID
    )

    expect(parsed.map(w => w.app)).toEqual(['top', 'middle', 'bottom'])
  })

  // The real shape of a Plasma desktop: the wallpapers and the two panels make
  // up four of the six entries on the stacking order. Reporting those instead of
  // the window underneath is the everyday failure.
  it('drops the desktop, panels and other compositor chrome', () => {
    const parsed = parseKwinWindows(
      payload(
        [
          row({ class: 'plasmashell', desktopWindow: true, onAllDesktops: true, pid: 9, stackingOrder: 0 }),
          row({ class: 'plasmashell', desktopWindow: true, onAllDesktops: true, pid: 9, stackingOrder: 1 }),
          row({ class: 'vivaldi-stable', pid: 7, stackingOrder: 2 }),
          row({ class: 'plasmashell', dock: true, onAllDesktops: true, pid: 9, stackingOrder: 3 }),
          row({ class: 'plasmashell', dock: true, onAllDesktops: true, pid: 9, stackingOrder: 4 }),
          row({ class: 'krunner', popupWindow: true, pid: 8, stackingOrder: 5 }),
          row({ class: 'tooltip', tooltip: true, pid: 8, stackingOrder: 6 })
        ],
        'd1'
      ),
      SELF_PID
    )

    expect(parsed.map(w => w.app)).toEqual(['vivaldi-stable'])
  })

  // Unlike the Hyprland provider, ours stays in: real z-order lets the caller
  // walk past our own window and take the next one, and that still works when
  // the HUD is not the frontmost window.
  it('keeps our own window, because real z-order lets the caller skip it', () => {
    const parsed = parseKwinWindows(
      payload([
        row({ class: 'com.nousresearch.hermes', pid: SELF_PID, stackingOrder: 0 }),
        row({ class: 'firefox', pid: 2, stackingOrder: 1 })
      ]),
      SELF_PID
    )

    expect(parsed.map(w => w.app)).toEqual(['firefox', 'com.nousresearch.hermes'])
  })

  // Windows on a hidden virtual desktop share coordinates with the visible ones,
  // so left in they win the overlap test and the tool reports what the user
  // cannot see.
  it('keeps only the desktop our own window is on, plus the always-visible ones', () => {
    const parsed = parseKwinWindows(
      payload(
        [
          row({ class: 'hermes', desktops: ['d2'], pid: SELF_PID, stackingOrder: 0 }),
          row({ class: 'visible', desktops: ['d2'], pid: 2, stackingOrder: 1 }),
          row({ class: 'elsewhere', desktops: ['d7'], pid: 3, stackingOrder: 2 }),
          row({ class: 'everywhere', desktops: [], onAllDesktops: true, pid: 4, stackingOrder: 3 })
        ],
        'd2'
      ),
      SELF_PID
    )

    expect(parsed.map(w => w.app)).toEqual(['everywhere', 'visible', 'hermes'])
  })

  it('falls back to the compositor’s current desktop when our own window is missing', () => {
    const parsed = parseKwinWindows(
      payload(
        [
          row({ class: 'here', desktops: ['d1'], pid: 2, stackingOrder: 0 }),
          row({ class: 'elsewhere', desktops: ['d7'], pid: 3, stackingOrder: 1 })
        ],
        'd1'
      ),
      SELF_PID
    )

    expect(parsed.map(w => w.app)).toEqual(['here'])
  })

  it('drops minimized, hidden and vanishing windows', () => {
    const parsed = parseKwinWindows(
      payload([
        row({ class: 'ok', pid: 2, stackingOrder: 0 }),
        row({ class: 'minimized', minimized: true, pid: 3, stackingOrder: 1 }),
        row({ class: 'hidden', hidden: true, pid: 4, stackingOrder: 2 }),
        row({ class: 'deleted', deleted: true, pid: 5, stackingOrder: 3 })
      ]),
      SELF_PID
    )

    expect(parsed.map(w => w.app)).toEqual(['ok'])
  })

  it('maps geometry into bounds and keeps the title', () => {
    const [window] = parseKwinWindows(
      payload([row({ caption: 'Deskmodder – News', height: 1034, pid: 2, width: 1920, x: 1920, y: 0 })]),
      SELF_PID
    )

    expect(window.bounds).toEqual({ x: 1920, y: 0, width: 1920, height: 1034 })
    expect(window.title).toBe('Deskmodder – News')
  })

  it('turns the compositor UUID into a stable numeric handle', () => {
    const first = parseKwinWindows(payload([row({ pid: 2 })]), SELF_PID)[0]
    const again = parseKwinWindows(payload([row({ pid: 2 })]), SELF_PID)[0]

    expect(first.id).toBe(again.id)
    expect(Number.isInteger(first.id)).toBe(true)
  })

  it('skips zero-sized windows', () => {
    expect(parseKwinWindows(payload([row({ height: 0, pid: 2, width: 0 })]), SELF_PID)).toEqual([])
  })

  it('reports nothing when the script reported an error', () => {
    expect(parseKwinWindows(JSON.stringify({ error: 'ReferenceError: workspace is not defined' }), SELF_PID)).toEqual([])
  })

  it('survives a payload that is not the window list', () => {
    for (const bad of ['', 'not json', '{}', 'null', '[]', JSON.stringify({ rows: 'nope' })]) {
      expect(parseKwinWindows(bad, SELF_PID)).toEqual([])
    }
  })
})

describe('the probe script KWin has to run', () => {
  // The scripting host parses this with an old engine. `for…of` and arrow
  // functions make the whole script fail to parse — `loadScript` still returns
  // successfully and `start()` still returns, but nothing is ever sent, which
  // is indistinguishable from a desktop with no windows.
  it('avoids the constructs the scripting host cannot parse', () => {
    expect(KWIN_PROBE_SCRIPT).not.toMatch(/for\s*\([^)]*\bof\b/)
    expect(KWIN_PROBE_SCRIPT).not.toMatch(/=>/)
  })

  it('reads the real stacking order and hands each window its own fields', () => {
    expect(KWIN_PROBE_SCRIPT).toContain('workspace.stackingOrder')
    expect(KWIN_PROBE_SCRIPT).toContain('resourceClass')
    expect(KWIN_PROBE_SCRIPT).toContain('frameGeometry')
    expect(KWIN_PROBE_SCRIPT).toContain('desktopWindow')
  })

  it('sends exactly one message, so the listener has one complete value to parse', () => {
    expect(KWIN_PROBE_SCRIPT.match(/__hermesSend\(/g)?.length).toBe(3)
  })
})

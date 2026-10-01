// Listen overlay — the global-hotkey live PC-audio transcription surface.
//
// A small frameless always-on-top window that a global shortcut (Ctrl+Shift+L)
// toggles from anywhere. While it is open, the main process runs the
// `pc-audio-monitor` skill's `live_listen.py` engine (WASAPI loopback capture
// + rolling faster-whisper transcription) as a child process and forwards its
// JSON-line output to the overlay so it renders rolling captions. On stop,
// the engine emits one authoritative transcript + an mp3 path.
//
// Structured after wake-indicator-window.ts: a single self-contained
// controller factory owns the BrowserWindow, the IPC, the global shortcut and
// the engine child process. The overlay window carries NO gateway connection
// of its own (same split as the pet overlay / wake indicator) — it renders
// state pushed from main over `hermes:listen-overlay:state`.
//
// The transcript is DELIBERATELY never handed to any chat session (Kirsin or
// otherwise): it stays local to this window, exactly the way the pet overlay
// or wake indicator never submit a prompt on their own. Feeding it to a
// session as a prompt made the agent "answer" what was just a spoken note —
// wrong for a subtitle/dictation surface. The window itself IS the delivery:
// it shows the rolling live captions while capturing, then the final
// corrected pass once you stop, and stays open (you close it with ✕, or start
// a fresh capture with the shortcut) so you can read or copy it.
import { type ChildProcess, spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { app, BrowserWindow, globalShortcut, ipcMain, screen } from 'electron'

import { createHudDragSession } from './hud-drag'

const LISTEN_SHORTCUT = 'CommandOrControl+Shift+L'

// The engine prints accented/non-ASCII text (Italian captions, app names
// from the OS). Python on Windows defaults stdout to the system codepage
// (e.g. cp1252) unless told otherwise; Node decodes a spawned child's stdout
// as UTF-8 by default, and the two disagree on multi-byte bytes — every
// accented character arrived as "?" until this forced the child's stdio to
// real UTF-8 regardless of the machine's locale.
const ENGINE_ENV = { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' }

// A title row, a target picker, a scrollable caption area, a status row.
// Wide enough to read a line of Italian without wrapping awkwardly; tall
// enough that a 30s+ recording's transcript has real room to scroll instead
// of clipping after 3 lines.
const LISTEN_WINDOW_WIDTH = 460
const LISTEN_WINDOW_HEIGHT = 320

// Spotlight-ish placement: centered on the active display, a fraction down
// from the top (same instinct as Quick Entry) rather than dead center.
const LISTEN_TOP_FRACTION = 0.18

/**
 * Resolve the Hermes venv python that the engine runs under, then the engine
 * script. Mirrors the app's own venv-root resolution: the venv lives inside
 * the active checkout, at `<HERMES_HOME>/hermes-agent/venv` (see
 * ACTIVE_HERMES_ROOT / VENV_ROOT in main.ts) — NOT directly under HERMES_HOME.
 * Returns null when the interpreter or the skill script is missing — the
 * shortcut then surfaces an error state instead of spawning a dead process.
 */
export function resolveListenEngine(hermesHome: string): { engine: string; python: string } | null {
  const activeRoot = path.join(hermesHome, 'hermes-agent')
  const venvPython = path.join(activeRoot, 'venv', 'Scripts', 'python.exe')
  const fallbackPython = path.join(activeRoot, 'venv', 'bin', 'python')
  const python = existsSync(venvPython) ? venvPython : existsSync(fallbackPython) ? fallbackPython : ''
  const engine = path.join(hermesHome, 'skills', 'pc-audio-monitor', 'scripts', 'live_listen.py')

  if (!python || !existsSync(engine)) {
    return null
  }

  return { engine, python }
}

export interface ListenTarget {
  pid: number
  name: string
  label: string
  hasAudio: boolean
}

/** User's device preference, persisted across sessions (main.ts owns the file). */
export type ListenDevicePreference = 'auto' | 'gpu' | 'cpu'

export function sanitizeListenDevicePreference(raw: unknown): ListenDevicePreference {
  return raw === 'gpu' || raw === 'cpu' ? raw : 'auto'
}

/** What happens to the final transcript: 'subtitle' never leaves this window
 *  (the default, and the only mode until the user explicitly opts in each
 *  time); 'dictate' delivers it as a real prompt to the Kirsin window, the
 *  same submitText path the composer's own dictation uses. Deliberately NOT
 *  persisted across a close/reopen — this is a per-use choice, not a standing
 *  preference, so a stale "dictate" setting can never silently ship a future
 *  transcript into chat (see the module comment: this bit the user once
 *  already, unconditionally, before mode selection existed). */
export type ListenMode = 'subtitle' | 'dictate'

export function sanitizeListenMode(raw: unknown): ListenMode {
  return raw === 'dictate' ? 'dictate' : 'subtitle'
}

/** Parse one JSON-line from the engine. Returns null on a bad line. */
export function parseEngineLine(line: string): null | Record<string, unknown> {
  const trimmed = line.trim()

  if (!trimmed) {
    return null
  }

  try {
    const value: unknown = JSON.parse(trimmed)

    return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
  } catch {
    return null
  }
}

export interface ListenOverlayState {
  /** 'idle' | 'starting' | 'listening' | 'stopping' | 'done' | 'error' */
  status: string
  /** Rolling caption: the newest transcribed chunk. */
  caption: string
  /** Language the live model most recently reported (e.g. 'it'). */
  lang: string
  /** Elapsed capture seconds, from the engine's `t` fields. */
  t: number
  /** Error detail when status === 'error'. */
  error: string
  /** Final transcript + mp3 path once the stop pass completes. */
  finalText: string
  finalMp3: string
  finalLang: string
  finalDur: number
  /** Apps currently holding an audio session, refreshed each time the overlay opens. */
  targets: ListenTarget[]
  /** 0 = system loopback (all apps); otherwise a pid from `targets`. */
  selectedPid: number
  /** User's device preference ('auto' tries GPU then falls back to CPU). */
  devicePref: ListenDevicePreference
  /** What the engine actually used for the last/current capture ('' before the first). */
  device: string
  /** Why the engine fell back from the preference (empty when it got what it asked for). */
  deviceNote: string
  /** 'subtitle' (default, stays in this window) or 'dictate' (final transcript
   *  is delivered to Kirsin as a real prompt). See ListenMode's doc comment —
   *  this is a per-open choice, never persisted. */
  mode: ListenMode
}

const INITIAL_STATE: ListenOverlayState = {
  status: 'idle',
  caption: '',
  lang: '',
  t: 0,
  error: '',
  finalText: '',
  finalMp3: '',
  finalLang: '',
  finalDur: 0,
  targets: [],
  selectedPid: 0,
  devicePref: 'auto',
  device: '',
  deviceNote: '',
  mode: 'subtitle'
}

interface ListenOverlayWindowOptions {
  devServer?: string
  getDevicePreference: () => ListenDevicePreference
  /** Resolves the live Kirsin BrowserWindow, or null if it isn't open. Used
   *  ONLY when mode === 'dictate' on the final transcript — see the
   *  'dictate' branch in the engine's 'final' handler. */
  getKirsinWindow: () => BrowserWindow | null
  hermesHome: string
  loadWindowUrl: (window: BrowserWindow, url: string, label: string) => void
  log: (message: string) => void
  /** Opens (or focuses) the Kirsin window — called before delivering a
   *  dictated transcript so it lands somewhere visible, mirroring how a
   *  manually-typed prompt always has a window to appear in. */
  openKirsinWindow: () => void
  preloadPath: string
  rendererIndex: () => string
  setDevicePreference: (pref: ListenDevicePreference) => void
  wireWindow: (window: BrowserWindow) => void
}

/**
 * Owns the overlay BrowserWindow, the engine child process, the caption feed
 * and the global Ctrl+Shift+L toggle. One controller instance for the app's
 * lifetime, created once from main.ts (mirrors createWakeIndicatorWindowController).
 */
export function createListenOverlayWindowController({
  devServer,
  getDevicePreference,
  getKirsinWindow,
  hermesHome,
  loadWindowUrl,
  log,
  openKirsinWindow,
  preloadPath,
  rendererIndex,
  setDevicePreference,
  wireWindow
}: ListenOverlayWindowOptions) {
  const drag = createHudDragSession()
  let window: BrowserWindow | null = null
  let child: null | ChildProcess = null
  let state: ListenOverlayState = { ...INITIAL_STATE, devicePref: getDevicePreference() }

  const url = () => {
    if (devServer) {
      return `${devServer.endsWith('/') ? devServer.slice(0, -1) : devServer}/?win=listen#/`
    }

    return `${pathToFileURL(rendererIndex()).toString()}?win=listen#/`
  }

  const windowBounds = () => {
    const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint())
    const { x: wx, y: wy, width: ww } = display.workArea

    return {
      x: Math.round(wx + (ww - LISTEN_WINDOW_WIDTH) / 2),
      y: Math.round(wy + display.workArea.height * LISTEN_TOP_FRACTION),
      width: LISTEN_WINDOW_WIDTH,
      height: LISTEN_WINDOW_HEIGHT
    }
  }

  const sendState = (next: ListenOverlayState): void => {
    state = next

    if (window && !window.isDestroyed()) {
      window.webContents.send('hermes:listen-overlay:state', state)
    }
  }

  const killEngine = (): void => {
    if (child && !child.killed) {
      try {
        child.kill()
      } catch {
        // Best effort — a dead child must not block a restart.
      }
    }

    child = null
  }

  /**
   * One-shot `--list-targets` call: near-instant (no model load), returns the
   * processes currently holding an audio session so the picker can offer them.
   * Best-effort — resolves to [] on any failure/timeout rather than throwing,
   * since a stale/empty picker must never block starting a system-wide capture.
   */
  const listTargets = (): Promise<ListenTarget[]> =>
    new Promise((resolve) => {
      const resolved = resolveListenEngine(hermesHome)

      if (!resolved) {
        resolve([])

        return
      }

      let proc: ChildProcess
      let settled = false
      let buf = ''

      const finish = (targets: ListenTarget[]): void => {
        if (settled) {
          return
        }

        settled = true
        clearTimeout(timer)

        try {
          proc.kill()
        } catch {
          // Already exited — nothing to do.
        }

        resolve(targets)
      }

      const timer = setTimeout(() => finish([]), 4000)

      try {
        proc = spawn(resolved.python, [resolved.engine, '--list-targets'], {
          cwd: os.tmpdir(),
          env: ENGINE_ENV,
          stdio: ['ignore', 'pipe', 'ignore'],
          windowsHide: true
        })
      } catch {
        clearTimeout(timer)
        resolve([])

        return
      }

      proc.stdout?.on('data', (data: Buffer) => {
        buf += data.toString()
        const lines = buf.split('\n')

        buf = lines.pop() ?? ''

        for (const line of lines) {
          const msg = parseEngineLine(line)

          if (msg?.type === 'targets' && Array.isArray(msg.targets)) {
            const raw = msg.targets as Array<Record<string, unknown>>

            finish(raw.map((t) => ({
              pid: Number(t.pid) || 0,
              name: typeof t.name === 'string' ? t.name : '',
              label: typeof t.label === 'string' ? t.label : '',
              hasAudio: t.has_audio === true
            })))
          }
        }
      })

      proc.on('exit', () => finish([]))
      proc.on('error', () => finish([]))
    })

  let targetsTimer: NodeJS.Timeout | null = null

  const refreshTargets = async (): Promise<void> => {
    const targets = await listTargets()

    if (!window || window.isDestroyed()) {
      return
    }

    sendState({ ...state, targets })
  }

  // Apps come and go while the picker is open, so poll rather than snapshot
  // once. list-targets loads no model — cheap enough for a 4s cadence.
  const startTargetPolling = (): void => {
    stopTargetPolling()
    void refreshTargets()
    targetsTimer = setInterval(() => void refreshTargets(), 4000)
  }

  const stopTargetPolling = (): void => {
    if (targetsTimer) {
      clearInterval(targetsTimer)
      targetsTimer = null
    }
  }

  const handleEngineLine = (line: string): void => {
    const msg = parseEngineLine(line)

    if (!msg) {
      return
    }

    const type = String(msg.type ?? '')

    switch (type) {
      case 'ready':
        sendState({
          ...state,
          status: 'listening',
          device: typeof msg.device === 'string' ? msg.device : state.device,
          deviceNote: typeof msg.device_note === 'string' ? msg.device_note : ''
        })

        break

      case 'alive':
        // Liveness ping — keep the elapsed clock honest, no caption change.
        if (typeof msg.t === 'number') {
          sendState({ ...state, t: msg.t })
        }

        break
      case 'chunk': {
        const text = typeof msg.text === 'string' ? msg.text : ''

        sendState({
          ...state,
          status: 'listening',
          caption: text || state.caption,
          lang: typeof msg.lang === 'string' ? msg.lang : state.lang,
          t: typeof msg.t === 'number' ? msg.t : state.t
        })

        break
      }

      case 'final': {
        const text = typeof msg.text === 'string' ? msg.text : ''
        const mp3 = typeof msg.mp3 === 'string' ? msg.mp3 : ''
        const finalLang = typeof msg.lang === 'string' ? msg.lang : ''

        sendState({
          ...state,
          status: 'done',
          finalText: text,
          finalMp3: mp3,
          finalLang,
          // The header's language badge should reflect the FINAL (medium
          // model, longer window, free detection) pass, not a stale early
          // live-chunk guess from a few seconds of noisy audio — that guess
          // is frequently wrong (e.g. a few seconds of Italian misheard as
          // Portuguese) and the badge never updated once capture stopped.
          lang: finalLang || state.lang,
          finalDur: typeof msg.dur === 'number' ? msg.dur : 0,
          t: typeof msg.dur === 'number' ? msg.dur : state.t
        })

        // Dictate mode ONLY: hand the corrected final transcript to Kirsin as
        // a real prompt, the same submitText path a manually-typed message
        // uses (see 'hermes:listen-overlay:dictate' in main.ts). Subtitle
        // mode (the default) never reaches this branch — the transcript
        // stays in this window, full stop. Empty/whitespace-only text is not
        // worth opening Kirsin over.
        if (state.mode === 'dictate' && text.trim()) {
          openKirsinWindow()
          const kirsin = getKirsinWindow()

          if (kirsin && !kirsin.isDestroyed()) {
            kirsin.webContents.send('hermes:listen-overlay:dictate', text.trim())
          } else {
            log('[listen-overlay] dictate: no Kirsin window to deliver to')
          }
        }

        break
      }

      case 'error':
        sendState({ ...state, status: 'error', error: String(msg.msg ?? 'engine error') })

        break

      default:
        // Unknown message type — ignore rather than error (forward compat).
        break
    }
  }

  const startEngine = (): void => {
    killEngine()

    const resolved = resolveListenEngine(hermesHome)
    const targetPid = state.selectedPid
    const devicePref = state.devicePref
    const mode = state.mode
    const carry = { targets: state.targets, selectedPid: targetPid, devicePref, mode }

    if (!resolved) {
      sendState({ ...INITIAL_STATE, ...carry, status: 'error', error: 'audio engine unavailable (venv or skill script missing)' })

      return
    }

    sendState({ ...INITIAL_STATE, ...carry, status: 'starting' })

    let proc: ChildProcess

    try {
      // The engine writes its wav/mp3 relative to its cwd (`live_listen.wav`
      // next to it) — run it from a scratch temp dir, never the app's own
      // install/working directory. targetPid 0 means system-wide loopback;
      // a nonzero pid captures only that process's own render audio.
      proc = spawn(resolved.python, [resolved.engine, '5', '1', String(targetPid), `--device=${devicePref}`], {
        cwd: os.tmpdir(),
        env: ENGINE_ENV,
        stdio: ['pipe', 'pipe', 'pipe'],
        windowsHide: true
      })
    } catch (err) {
      sendState({ ...INITIAL_STATE, ...carry, status: 'error', error: `engine failed to start: ${String(err)}` })

      return
    }

    child = proc
    let buf = ''

    proc.stdout?.on('data', (data: Buffer) => {
      buf += data.toString()
      const lines = buf.split('\n')

      buf = lines.pop() ?? ''

      for (const line of lines) {
        handleEngineLine(line)
      }
    })

    proc.stderr?.on('data', (data: Buffer) => {
      // The engine logs model-download noise to stderr; surface only genuine
      // failures (a crash line) and keep the caption feed clean otherwise.
      const text = data.toString().trim()

      if (text && /error|traceback|exception/i.test(text)) {
        log(`listen-overlay engine stderr: ${text.slice(-300)}`)
      }
    })

    proc.on('exit', (code) => {
      const wasListening = state.status === 'listening' || state.status === 'stopping'

      if (wasListening && code !== 0 && !state.finalText) {
        sendState({ ...state, status: 'error', error: `engine exited ${code ?? 'unexpectedly'}` })
      }

      killEngine()
    })

    proc.stdin?.on('error', () => {
      // Broken stdin pipe on a killed child — nothing to do.
    })
  }

  const stopEngine = (): void => {
    if (!child) {
      return
    }

    sendState({ ...state, status: 'stopping' })

    try {
      child.stdin?.write('stop\n')
    } catch {
      // Stdin already closed — just wait for the exit handler.
    }
  }

  const spawnWindow = (): BrowserWindow => {
    const next = new BrowserWindow({
      ...windowBounds(),
      alwaysOnTop: true,
      backgroundColor: '#00000000',
      // A REAL interactive window (Start/Device/Mode/close buttons all need
      // reliable OS-level click delivery), not an ambient overlay like the
      // pet sprite — focusable: false + showInactive() there is deliberate
      // (it must never steal focus while drifting over other apps), but on
      // this window it silently broke mouse-click hit-testing on Windows:
      // every STATE transition still worked because the global Ctrl+Shift+L
      // shortcut bypasses window focus entirely, so the bug went unnoticed
      // until the user tried clicking the overlay's own buttons directly
      // (e.g. after opening it from the Kirsin mic button). Every other
      // clickable utility window in this app (Kirsin, Quick Entry, the HUD
      // chat) is a normal focusable window — this one now matches them.
      focusable: true,
      frame: false,
      fullscreenable: false,
      hasShadow: false,
      hiddenInMissionControl: true,
      maximizable: false,
      minimizable: false,
      movable: false,
      resizable: false,
      show: false,
      skipTaskbar: true,
      transparent: true,
      type: 'panel',
      webPreferences: {
        backgroundThrottling: false,
        contextIsolation: true,
        devTools: true,
        nodeIntegration: false,
        preload: preloadPath,
        sandbox: true
      }
    })

    next.setAlwaysOnTop(true, 'screen-saver')

    try {
      next.setVisibleOnAllWorkspaces(true, { skipTransformProcessType: true, visibleOnFullScreen: true })
    } catch {
      // Best effort on older Electron/macOS combinations.
    }

    wireWindow(next)

    next.on('closed', () => {
      if (window === next) {
        window = null
      }

      stopEngine()
    })

    next.once('ready-to-show', () => {
      if (!next.isDestroyed()) {
        next.show()
      }
    })

    loadWindowUrl(next, url(), 'Listen overlay')

    return next
  }

  const open = (): void => {
    if (!window || window.isDestroyed()) {
      window = spawnWindow()
    } else {
      window.setBounds(windowBounds())
      window.show()
    }

    startTargetPolling()
  }

  const close = (): void => {
    stopTargetPolling()

    if (window && !window.isDestroyed()) {
      window.close()
    }

    window = null
  }

  const toggle = (): void => {
    if (child) {
      // Actively capturing (or already told to stop and finishing the final
      // pass) → stop. The window stays open so the caption area fills in with
      // the corrected final transcript once the engine exits.
      stopEngine()

      return
    }

    if (!window || window.isDestroyed()) {
      // First press: just open the picker, idle. The user picks a source
      // (or leaves it on system audio) and starts explicitly — either the
      // overlay's Start control or pressing the shortcut again — rather than
      // capturing from whatever was selected last before they could look.
      open()

      return
    }

    // Window already open and idle (nothing captured yet, or a previous
    // capture's result is still on screen) → this press starts fresh.
    // startEngine() resets to INITIAL_STATE, clearing any old transcript.
    startEngine()
  }

  // The controller is constructed at module scope, ahead of app.whenReady() —
  // globalShortcut throws if called before the app is ready, so registration
  // must wait for it instead of running inline here (#swap-script-incident).
  if (app.isReady()) {
    globalShortcut.register(LISTEN_SHORTCUT, toggle)
  } else {
    app.whenReady().then(() => globalShortcut.register(LISTEN_SHORTCUT, toggle))
  }

  // ---- IPC: overlay window control + state ---------------------------------
  const isFromOverlay = (event: Electron.IpcMainEvent): boolean => Boolean(window) && !window!.isDestroyed() && event.sender === window!.webContents

  ipcMain.on('hermes:listen-overlay:begin-move', (event) => {
    if (!isFromOverlay(event) || !window) {
      return
    }

    const [x, y] = window.getPosition()

    drag.begin(screen.getCursorScreenPoint(), { x, y })
  })

  ipcMain.on('hermes:listen-overlay:end-move', (event) => {
    if (!isFromOverlay(event)) {
      return
    }

    drag.end()
  })

  ipcMain.on('hermes:listen-overlay:move-by', (event, size: unknown) => {
    if (!isFromOverlay(event) || !window) {
      return
    }

    const s = (size && typeof size === 'object' ? size : {}) as { width?: unknown; height?: unknown }
    const width = Number(s.width)
    const height = Number(s.height)

    if (!Number.isFinite(width) || !Number.isFinite(height)) {
      return
    }

    const origin = drag.origin(screen.getCursorScreenPoint())

    if (!origin) {
      return
    }

    window.setBounds({ x: origin.x, y: origin.y, width: Math.round(width), height: Math.round(height) })
  })

  // The overlay asks to close itself (its ✕ button). Stop the engine too so a
  // dismissed overlay never keeps the mic running.
  ipcMain.on('hermes:listen-overlay:close', () => {
    stopEngine()
    close()
  })

  // The overlay's Start button — same effect as pressing the shortcut while
  // idle, for anyone who'd rather click than remember the chord.
  ipcMain.on('hermes:listen-overlay:start', (event) => {
    if (!isFromOverlay(event) || child) {
      return
    }

    startEngine()
  })

  ipcMain.handle('hermes:listen-overlay:get-state', () => state)

  // The overlay's dropdown picked a target (0 = system-wide loopback). Only
  // takes effect on the NEXT capture — switching mid-listen would orphan the
  // running engine's stream, so this just updates the pending selection.
  ipcMain.on('hermes:listen-overlay:select-target', (event, pid: unknown) => {
    if (!isFromOverlay(event)) {
      return
    }

    const next = Number(pid)

    if (!Number.isFinite(next)) {
      return
    }

    sendState({ ...state, selectedPid: Math.max(0, Math.round(next)) })
  })

  // The overlay's device picker (Auto / GPU / CPU). Persists immediately so
  // it survives a restart, and — like the target picker — only takes effect
  // on the NEXT capture (switching mid-listen would orphan a loaded model).
  ipcMain.on('hermes:listen-overlay:select-device', (event, pref: unknown) => {
    if (!isFromOverlay(event)) {
      return
    }

    const next = sanitizeListenDevicePreference(pref)

    setDevicePreference(next)
    sendState({ ...state, devicePref: next })
  })

  // The overlay's Subtitle/Dictate toggle. Deliberately NOT persisted (see
  // ListenMode's doc comment) — every open starts back at 'subtitle', so a
  // silent leftover "dictate" setting from a past session can never surprise
  // the user by piping a future transcript into chat. Takes effect on the
  // NEXT capture's final pass, same as the other pickers.
  ipcMain.on('hermes:listen-overlay:select-mode', (event, mode: unknown) => {
    if (!isFromOverlay(event)) {
      return
    }

    sendState({ ...state, mode: sanitizeListenMode(mode) })
  })

  return {
    close,
    dispose: (): void => {
      globalShortcut.unregister(LISTEN_SHORTCUT)
      stopTargetPolling()
      drag.end()
      killEngine()
      close()
    },
    open,
    toggle
  }
}

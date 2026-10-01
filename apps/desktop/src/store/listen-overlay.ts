/**
 * Renderer-side types + bridge for the listen overlay (global Ctrl+Shift+L
 * live PC-audio transcription HUD). The overlay window itself is a separate
 * gateway-less BrowserWindow (`?win=listen`, created in
 * electron/listen-overlay.ts) — this module exists so the type shape is
 * importable from the renderer (electron/ is excluded from the renderer
 * tsconfig, so the main-process module's own type can't be imported directly).
 *
 * Unlike the pet overlay, this module carries no store/bridge logic of its
 * own: the overlay window reads state directly via `window.hermesDesktop
 * .listenOverlay.onState`. The transcript is DELIBERATELY never delivered to
 * any other window or session — it stays in this HUD, which is the delivery
 * surface itself (see listen-overlay.ts's module comment for why). Kept here
 * purely for the shared type.
 */
export interface ListenTarget {
  pid: number
  name: string
  label: string
  hasAudio: boolean
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
  /** Apps currently holding an audio session, refreshed while the overlay is open. */
  targets: ListenTarget[]
  /** 0 = system loopback (all apps); otherwise a pid from `targets`. */
  selectedPid: number
  /** User's device preference ('auto' tries GPU then falls back to CPU). */
  devicePref: 'auto' | 'gpu' | 'cpu'
  /** What the engine actually used for the last/current capture ('' before the first). */
  device: string
  /** Why the engine fell back from the preference (empty when it got what it asked for). */
  deviceNote: string
  /** 'subtitle' (default, stays in this window) or 'dictate' (final transcript
   *  is delivered to Kirsin as a real prompt). Never persisted across opens. */
  mode: 'subtitle' | 'dictate'
}

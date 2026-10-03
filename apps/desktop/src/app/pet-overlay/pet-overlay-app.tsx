import { useStore } from '@nanostores/react'
import { type CSSProperties, useCallback, useEffect, useRef, useState } from 'react'

import { blobToDataUrl } from '@/app/session/hooks/use-prompt-actions/utils'
import { PetHeartField, playVibeHearts } from '@/components/chat/vibe-hearts'
import { PetSprite } from '@/components/pet/pet-sprite'
import { type PetZoomAnchor, usePetZoomGesture } from '@/components/pet/use-pet-zoom-gesture'
import { ArrowUp, AudioLines, ChevronDown, Mail, Pencil, Square } from '@/lib/icons'
import { isSubmitEnter } from '@/lib/ime'
import { $petActivity, $petInfo, setPetInfo } from '@/store/pet'
import { overlayWindowSize, type PetOverlayHeard, type PetOverlayNotice, type PetOverlayTurn } from '@/store/pet-overlay'
import { setAwaitingResponse, setBusy } from '@/store/session'

// Fallbacks mirror pet-sprite's defaults; the gateway normally sends real values.
const DEFAULT_FRAME_W = 192
const DEFAULT_FRAME_H = 208
const DEFAULT_SCALE = 0.33

// Must match the root's paddingBottom — the sprite renders bottom-centered, this
// many px above the window's bottom edge. Used to anchor the resize.
// Room under the pet for the companion pill (composer) and the reply card that
// hangs below it, Codex-style. The pet's feet sit this far above the bottom.
const COMPANION_AREA_H = 420
const PET_PADDING_BOTTOM = COMPANION_AREA_H

// Frosted-glass chrome for the companion (Codex-style). The overlay window is
// transparent, so the translucency shows the desktop through it.
const GLASS: CSSProperties = {
  backdropFilter: 'blur(18px) saturate(140%)',
  background: 'linear-gradient(180deg, rgba(92,104,116,0.62), rgba(52,62,72,0.62))',
  border: '1px solid rgba(255,255,255,0.22)',
  boxShadow: '0 8px 28px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.18)',
  color: '#fff',
  textShadow: '0 1px 2px rgba(0,0,0,0.35)'
}

const TOOL_BUTTON: CSSProperties = {
  alignItems: 'center',
  background: 'transparent',
  border: 'none',
  color: '#fff',
  cursor: 'pointer',
  display: 'inline-flex',
  height: 34,
  justifyContent: 'center',
  padding: 0,
  width: 40
}

const TOOL_DIVIDER: CSSProperties = {
  alignSelf: 'stretch',
  background: 'rgba(255,255,255,0.28)',
  margin: '8px 0',
  width: 1
}

// A sprite pixel counts as "solid" (interactive) at/above this alpha (0-255).
// Low enough to catch anti-aliased edges, high enough that the faint halo around
// the art still clicks through.
const ALPHA_HIT_THRESHOLD = 16

/**
 * The pop-out overlay's only view: a transparent, draggable mascot with a mini
 * composer.
 *
 * This runs in a separate, gateway-less BrowserWindow (`?win=overlay`). It is a
 * pure puppet — the main renderer pushes the live pet state over IPC and we
 * mirror it into the same atoms the in-window pet reads, so `PetSprite` /
 * `PetBubble` render identically with zero extra logic.
 *
 * The window is a full rectangle but mostly transparent; we toggle OS-level
 * mouse click-through so only the sprite (or the open composer) is interactive
 * and the empty margins pass clicks through to whatever is behind.
 *
 * Gestures on the pet: drag to move it anywhere on screen (even outside the
 * app), shift-click to pop it back into the window, single-click to open a small
 * composer, double-click to toggle the app window (minimize ↔ restore). A mail
 * icon (shown only when a turn finished while you were away) raises the app on
 * the most recent thread.
 */

// Below this much pointer travel, a press counts as a click, not a drag.
const CLICK_SLOP_PX = 3
// A second click within this window is a double-click (raise app) and cancels
// the deferred single-click (open composer), so a double never flashes it open.
const DOUBLE_CLICK_MS = 250

interface DragState {
  startX: number
  startY: number
  offX: number
  offY: number
  width: number
  height: number
  moved: boolean
}

export function PetOverlayApp() {
  const info = useStore($petInfo)
  const [composerOpen, setComposerOpen] = useState(false)
  const [draft, setDraft] = useState('')
  // Mirrored from the main renderer: a finish landed while you were away.
  const [unread, setUnread] = useState(false)
  // Companion: what dictation heard (or why it failed), echoed into the card.
  const [heard, setHeard] = useState<PetOverlayHeard | null>(null)
  const [recording, setRecording] = useState(false)
  // Companion state: the text field, and whether the agent is mid-turn (the
  // card then shows "Thinking…" under the recent conversation).
  const [writing, setWriting] = useState(false)
  const [working, setWorking] = useState(false)
  const [thread, setThread] = useState<PetOverlayTurn[]>([])
  // A reminder / "needs your attention" notice: unfolds the balloon on arrival
  // (without taking the keyboard) and stays until dismissed.
  const [notice, setNotice] = useState<PetOverlayNotice | null>(null)
  const lastNoticeIdRef = useRef<string | null>(null)
  // Turns hidden by "New chat": the balloon starts clean even if the app
  // keeps the old session around.
  const [clearedIds, setClearedIds] = useState<ReadonlySet<string>>(() => new Set())
  const visibleThread = thread.filter(turn => !clearedIds.has(turn.id))
  const threadEndRef = useRef<HTMLDivElement | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const companionRef = useRef<HTMLDivElement | null>(null)

  const dragRef = useRef<DragState | null>(null)
  // Last Alt+wheel anchor, consumed by the resize effect to zoom toward the
  // cursor; null means a non-wheel scale change (slider) → anchor bottom-center.
  const zoomAnchorRef = useRef<PetZoomAnchor | null>(null)
  const petRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  // Last mirrored reaction id — a bump means the main window fired a reaction.
  const lastReactionRef = useRef<number | null>(null)
  const ignoreRef = useRef(true)
  const composerOpenRef = useRef(false)
  const clickTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  const setIgnore = (ignore: boolean) => {
    if (ignoreRef.current !== ignore) {
      ignoreRef.current = ignore
      window.hermesDesktop?.petOverlay?.setIgnoreMouse(ignore)
    }
  }

  // Mirror pushed state into the shared atoms so PetSprite/PetBubble just work.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const off = window.hermesDesktop?.petOverlay?.onState(payload => {
      setPetInfo(payload.info)
      $petActivity.set(payload.activity ?? {})
      setBusy(Boolean(payload.busy))
      setWorking(Boolean(payload.busy))
      setAwaitingResponse(Boolean(payload.awaiting))
      setUnread(Boolean(payload.unread))
      setThread(payload.thread ?? [])

      const incoming = payload.notice ?? null

      if (incoming && incoming.id !== lastNoticeIdRef.current) {
        lastNoticeIdRef.current = incoming.id
        setNotice(incoming)
        setComposerOpen(true)
      }

      setHeard(prev => {
        const next = payload.heard ?? null

        // A fresh dictation result (or error) unfolds the companion.
        if (next && next.id !== prev?.id) {
          setComposerOpen(true)
        }

        return next
      })

      // Play a reaction on a new id (ignore the first sync, which just primes it).
      const reaction = payload.reaction ?? null

      if (lastReactionRef.current === null) {
        lastReactionRef.current = reaction?.id ?? 0
      } else if (reaction && reaction.id > lastReactionRef.current) {
        lastReactionRef.current = reaction.id

        if (reaction.kind === 'vibe') {
          playVibeHearts()
        }
      }
    })

    // Tell the main renderer we're mounted so it pushes the current frame (the
    // subscribe-time pushes during open() can land before this view exists).
    window.hermesDesktop?.petOverlay?.control({ type: 'ready' })

    return off
  }, [])

  // Click-through: make only the *solid* sprite pixels (plus the bubble / mail
  // button / open composer) interactive — clicks on the transparent rectangle
  // around the art pass through to whatever's behind. With ignore+forward, the
  // renderer still receives mousemove so we can re-arm the moment the cursor
  // returns to a solid pixel.
  useEffect(() => {
    setIgnore(true)

    // True when the point sits on a solid sprite pixel or on the pet's other
    // interactive chrome (bubble, mail button). Over the canvas we sample the
    // rendered alpha; elsewhere inside the pet (bubble/button) we trust DOM
    // hit-testing. Anything else is transparent backdrop.
    const isInteractiveAt = (x: number, y: number): boolean => {
      const pet = petRef.current
      const target = document.elementFromPoint(x, y)

      if (!pet || !target || !pet.contains(target)) {
        return false
      }

      if (!(target instanceof HTMLCanvasElement)) {
        return true
      }

      const rect = target.getBoundingClientRect()

      if (rect.width === 0 || rect.height === 0) {
        return true
      }

      const ctx = target.getContext('2d')

      if (!ctx) {
        return true
      }

      const px = Math.floor((x - rect.left) * (target.width / rect.width))
      const py = Math.floor((y - rect.top) * (target.height / rect.height))

      try {
        return ctx.getImageData(px, py, 1, 1).data[3] >= ALPHA_HIT_THRESHOLD
      } catch {
        // Tainted/zero-size read — fail open so the pet stays grabbable.
        return true
      }
    }

    const onMove = (ev: MouseEvent) => {
      if (dragRef.current) {
        setIgnore(false)

        return
      }

      // The companion pill and reply card are interactive; the transparent rest
      // of the (large) window always clicks through, so an open composer never
      // turns the area around the pet into a dead zone.
      const companion = companionRef.current
      const target = document.elementFromPoint(ev.clientX, ev.clientY)

      if (companion && target && companion.contains(target) && target !== companion) {
        setIgnore(false)

        return
      }

      setIgnore(!isInteractiveAt(ev.clientX, ev.clientY))
    }

    window.addEventListener('mousemove', onMove)

    return () => {
      window.removeEventListener('mousemove', onMove)
      clearTimeout(clickTimerRef.current)
    }
  }, [])

  // The whole window must stay interactive while the composer is open (so the
  // input keeps focus); focus it on open. The overlay is a non-activating panel
  // (so it never steals the app's cmd/alt-tab anchor) — flip it focusable while
  // the composer needs the keyboard, then back to non-activating when it closes.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    composerOpenRef.current = composerOpen

    // Only the text field needs the keyboard; the toolbar and card work with
    // the overlay left non-activating.
    window.hermesDesktop?.petOverlay?.setFocusable(writing)

    if (writing) {
      // The OS window has to become key first (setFocusable + focus happen in
      // the main process), so focus the input on the next frame.
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [composerOpen, writing])

  const onPetPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) {
      return
    }

    ;(e.target as Element).setPointerCapture?.(e.pointerId)
    dragRef.current = {
      height: window.outerHeight,
      moved: false,
      offX: e.screenX - window.screenX,
      offY: e.screenY - window.screenY,
      startX: e.screenX,
      startY: e.screenY,
      width: window.outerWidth
    }
  }

  const onPetPointerMove = (e: React.PointerEvent) => {
    const drag = dragRef.current

    if (!drag) {
      return
    }

    if (Math.hypot(e.screenX - drag.startX, e.screenY - drag.startY) > CLICK_SLOP_PX) {
      drag.moved = true
    }

    window.hermesDesktop?.petOverlay?.setBounds({
      height: drag.height,
      width: drag.width,
      x: e.screenX - drag.offX,
      y: e.screenY - drag.offY
    })
  }

  const onPetPointerUp = (e: React.PointerEvent) => {
    const drag = dragRef.current
    dragRef.current = null
    ;(e.target as Element).releasePointerCapture?.(e.pointerId)

    if (!drag) {
      return
    }

    if (drag.moved) {
      // A drag cancels any deferred single-click so the composer can't pop open
      // after you reposition the pet.
      clearTimeout(clickTimerRef.current)
      clickTimerRef.current = undefined

      // Remember the spot on the desktop (screen coords) so the pet reopens here
      // next time / after a restart.
      window.hermesDesktop?.petOverlay?.control({
        bounds: { height: drag.height, width: drag.width, x: e.screenX - drag.offX, y: e.screenY - drag.offY },
        type: 'bounds'
      })

      return
    }

    // Shift-click always pops the pet back in (no double-click ambiguity).
    if (e.shiftKey) {
      window.hermesDesktop?.petOverlay?.control({ type: 'pop-in' })

      return
    }

    // Double-click toggles the app window (minimize ↔ restore); defer the
    // single-click composer toggle so a double never flashes the composer open.
    if (clickTimerRef.current) {
      clearTimeout(clickTimerRef.current)
      clickTimerRef.current = undefined
      window.hermesDesktop?.petOverlay?.control({ type: 'toggle-app' })

      return
    }

    clickTimerRef.current = setTimeout(() => {
      clickTimerRef.current = undefined

      if (composerOpenRef.current) {
        cancelRecording()
      }

      setComposerOpen(open => {
        setWriting(!open)

        return !open
      })
    }, DOUBLE_CLICK_MS)
  }

  // Drop an in-flight take without transcribing it (typing or collapsing wins).
  const cancelRecording = () => {
    const current = recorderRef.current

    if (current) {
      current.ondataavailable = null

      current.onstop = () => {
        for (const track of current.stream.getTracks()) {
          track.stop()
        }
      }

      if (current.state !== 'inactive') {
        current.stop()
      }

      recorderRef.current = null
    }

    setRecording(false)
  }

  const send = () => {
    const text = draft.trim()

    cancelRecording()

    if (text) {
      window.hermesDesktop?.petOverlay?.control({ text, type: 'submit' })
      setHeard(null)
    }

    setDraft('')
  }

  // Dictation: record here (the overlay owns the mic while it's in front), hand
  // the audio to the main renderer, which transcribes and sends it.
  const toggleRecording = async () => {
    const current = recorderRef.current

    if (current) {
      current.stop()

      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      const chunks: Blob[] = []

      recorder.ondataavailable = event => {
        if (event.data.size > 0) {
          chunks.push(event.data)
        }
      }

      recorder.onstop = () => {
        for (const track of stream.getTracks()) {
          track.stop()
        }

        recorderRef.current = null
        setRecording(false)

        const audio = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' })

        if (audio.size === 0) {
          return
        }

        setHeard({ id: -1, text: 'Transcribing…' })
        void blobToDataUrl(audio).then(dataUrl =>
          window.hermesDesktop?.petOverlay?.control({ dataUrl, mime: audio.type, type: 'dictate' })
        )
      }

      recorderRef.current = recorder
      recorder.start()
      setRecording(true)
      // Safety cap: a take never runs past a minute even if the stop click is lost.
      setTimeout(() => {
        if (recorderRef.current === recorder && recorder.state === 'recording') {
          recorder.stop()
        }
      }, 60_000)
    } catch {
      setComposerOpen(true)
      setHeard({ error: true, id: -1, text: 'No microphone access. Allow Hermes in System Settings → Privacy → Microphone.' })
    }
  }

  const openApp = () => {
    // The envelope means "an answer landed": show it in the balloon right here
    // instead of throwing the main window over the desktop.
    setUnread(false)
    setComposerOpen(true)
    setWriting(true)
    window.hermesDesktop?.petOverlay?.control({ type: 'mark-read' })
  }


  // Alt+wheel over the popped-out pet resizes it. The overlay has no gateway,
  // so paint the new scale locally for instant feedback, then ask the main
  // renderer to persist it (it pushes the reconciled scale back). Stash the
  // cursor anchor for the resize effect; the window itself is grown to fit there.
  const onScale = useCallback((next: number, anchor: PetZoomAnchor) => {
    zoomAnchorRef.current = anchor
    setPetInfo({ ...$petInfo.get(), scale: next })
    window.hermesDesktop?.petOverlay?.control({ scale: next, type: 'scale' })
  }, [])

  usePetZoomGesture(petRef, onScale, Boolean(info.enabled && info.spritesheetBase64))

  // Grow/shrink the OS overlay window to fit the pet at its current scale so the
  // sprite is never cropped — covers both the wheel gesture here and a scale
  // changed from the app's settings slider (pushed in as a state update). With a
  // wheel anchor we zoom toward the cursor (keep the pixel under it fixed);
  // otherwise we anchor the bottom-center (the pet's feet stay planted). New
  // bounds are persisted so the pet reopens at the right size.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (!info.enabled || !info.spritesheetBase64) {
      return
    }

    const { width, height } = overlayWindowSize(
      info.frameW ?? DEFAULT_FRAME_W,
      info.frameH ?? DEFAULT_FRAME_H,
      info.scale ?? DEFAULT_SCALE
    )

    const curW = window.outerWidth
    const curH = window.outerHeight

    if (width === curW && height === curH) {
      zoomAnchorRef.current = null

      return
    }

    const anchor = zoomAnchorRef.current
    zoomAnchorRef.current = null

    // The sprite scales about its bottom-center, at window-local (curW/2,
    // curH - paddingBottom). Hold the anchor pixel fixed on screen as it scales;
    // with no wheel anchor we pin the bottom-center itself (ratio 1 ⇒ no shift).
    const ratio = anchor?.ratio ?? 1
    const ax = anchor?.clientX ?? curW / 2
    const ay = anchor?.clientY ?? curH - PET_PADDING_BOTTOM

    const bounds = {
      height,
      width,
      x: Math.round(window.screenX + ax - (ax - curW / 2) * ratio - width / 2),
      y: Math.round(window.screenY + ay - (ay - (curH - PET_PADDING_BOTTOM)) * ratio - (height - PET_PADDING_BOTTOM))
    }

    window.hermesDesktop?.petOverlay?.setBounds(bounds)
    window.hermesDesktop?.petOverlay?.control({ bounds, type: 'bounds' })
  }, [info.enabled, info.spritesheetBase64, info.scale, info.frameW, info.frameH])

   
  useEffect(() => {
    if (!heard?.error) {
      return
    }

    const timer = setTimeout(() => setHeard(current => (current === heard ? null : current)), 5000)

    return () => clearTimeout(timer)
  }, [heard])

  // Keep the newest turn in view as the conversation grows.
   
  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ block: 'end' })
  }, [thread, working, recording, heard])

  if (!info.enabled || !info.spritesheetBase64) {
    return null
  }


  return (
    <div
      style={{
        alignItems: 'center',
        background: 'transparent',
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        justifyContent: 'flex-end',
        paddingBottom: PET_PADDING_BOTTOM,
        position: 'relative',
        userSelect: 'none',
        width: '100vw'
      }}
    >
      <div
        onPointerDown={onPetPointerDown}
        onPointerMove={onPetPointerMove}
        onPointerUp={onPetPointerUp}
        ref={petRef}
        style={{
          alignItems: 'center',
          cursor: 'grab',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          touchAction: 'none'
        }}
      >
        <div style={{ lineHeight: 0, position: 'relative' }}>
          <PetSprite info={info} pauseWhenUnfocused={false} />

          {/* Hearts on the popped-out pet — identical to in-window. */}
          <PetHeartField
            petH={(info.frameH ?? DEFAULT_FRAME_H) * (info.scale ?? DEFAULT_SCALE)}
            petW={(info.frameW ?? DEFAULT_FRAME_W) * (info.scale ?? DEFAULT_SCALE)}
          />

          {unread && (
            <button
              aria-label="Open in Hermes"
              onClick={openApp}
              onPointerDown={e => e.stopPropagation()}
              onPointerUp={e => e.stopPropagation()}
              style={{
                alignItems: 'center',
                background: 'var(--ui-bg-elevated)',
                border: '1px solid var(--ui-stroke-secondary)',
                borderRadius: 999,
                boxShadow: '0 4px 14px rgba(0,0,0,0.22)',
                color: 'var(--foreground)',
                cursor: 'pointer',
                display: 'inline-flex',
                height: 24,
                justifyContent: 'center',
                padding: 0,
                position: 'absolute',
                right: 0,
                top: 0,
                width: 24
              }}
              type="button"
            >
              <Mail style={{ height: 13, width: 13 }} />
            </button>
          )}
        </div>
      </div>

      {/* Companion area under the pet: a rounded pill to type or dictate, and
          the reply card hanging below it. Clicking the pet folds it away. */}
      <div
        ref={companionRef}
        style={{
          alignItems: 'center',
          bottom: 0,
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
          height: COMPANION_AREA_H,
          left: 0,
          paddingTop: 10,
          position: 'absolute',
          right: 0
        }}
      >
        {composerOpen && (
          <div
            style={{
              ...GLASS,
              alignItems: 'center',
              borderRadius: 999,
              display: 'flex',
              height: 38,
              padding: '0 4px'
            }}
          >
            <button
              aria-label="New chat"
              onClick={() => {
                cancelRecording()
                setHeard(null)
                setClearedIds(new Set(thread.map(turn => turn.id)))
                setDraft('')
                setWriting(true)
                window.hermesDesktop?.petOverlay?.control({ type: 'new-chat' })
              }}
              style={TOOL_BUTTON}
              title="New chat"
              type="button"
            >
              <Pencil style={{ height: 17, width: 17 }} />
            </button>
            <span style={TOOL_DIVIDER} />
            <button
              aria-label={recording ? 'Stop and send' : 'Dictate'}
              onClick={() => void toggleRecording()}
              style={{ ...TOOL_BUTTON, color: recording ? '#ff6b6b' : '#fff' }}
              title={recording ? 'Stop and send' : 'Dictate'}
              type="button"
            >
              {recording ? (
                <Square style={{ height: 15, width: 15 }} />
              ) : (
                <AudioLines style={{ height: 18, width: 18 }} />
              )}
            </button>
            <span style={TOOL_DIVIDER} />
            <button
              aria-label="Collapse"
              onClick={() => {
                cancelRecording()
                setWriting(false)
                setComposerOpen(false)
              }}
              style={TOOL_BUTTON}
              title="Collapse"
              type="button"
            >
              <ChevronDown style={{ height: 17, width: 17 }} />
            </button>
          </div>
        )}

        {composerOpen && writing && (
          <div
            style={{
              ...GLASS,
              alignItems: 'center',
              borderRadius: 999,
              display: 'flex',
              gap: 6,
              height: 42,
              padding: '0 5px 0 16px',
              width: 330
            }}
          >
            <input
              onChange={e => setDraft(e.target.value)}
              onKeyDown={e => {
                if (isSubmitEnter(e) && !e.shiftKey) {
                  e.preventDefault()
                  send()
                } else if (e.key === 'Escape') {
                  setWriting(false)
                }
              }}
              placeholder="Ask Hermes"
              ref={inputRef}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#fff',
                flex: 1,
                fontSize: 13.5,
                minWidth: 0,
                outline: 'none'
              }}
              value={draft}
            />
            <button
              aria-label="Send"
              disabled={!draft.trim()}
              onClick={send}
              style={{
                alignItems: 'center',
                background: draft.trim() ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.25)',
                border: 'none',
                borderRadius: 999,
                color: '#1d2b36',
                cursor: draft.trim() ? 'pointer' : 'default',
                display: 'inline-flex',
                flex: 'none',
                height: 32,
                justifyContent: 'center',
                padding: 0,
                width: 32
              }}
              type="button"
            >
              <ArrowUp style={{ height: 16, width: 16 }} />
            </button>
          </div>
        )}

        {composerOpen && (visibleThread.length > 0 || recording || heard || notice) && (
          <div
            style={{
              ...GLASS,
              borderRadius: 22,
              cursor: 'default',
              display: 'flex',
              flexDirection: 'column',
              fontSize: 13.5,
              gap: 8,
              lineHeight: 1.4,
              maxHeight: COMPANION_AREA_H - 110,
              overflowY: 'auto',
              padding: '10px 18px 11px',
              userSelect: 'text',
              whiteSpace: 'pre-wrap',
              width: 340
            }}
          >
            {notice && (
              <div
                style={{
                  background: 'rgba(255, 214, 102, 0.18)',
                  border: '1px solid rgba(255, 214, 102, 0.45)',
                  borderRadius: 14,
                  display: 'flex',
                  gap: 8,
                  padding: '7px 8px 8px 10px'
                }}
              >
                <div style={{ flex: 1 }}>{notice.text}</div>
                <button
                  aria-label="Dismiss notice"
                  onClick={() => setNotice(null)}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: 'inherit',
                    cursor: 'pointer',
                    flex: 'none',
                    opacity: 0.7,
                    padding: 0
                  }}
                  title="Dismiss"
                  type="button"
                >
                  ×
                </button>
              </div>
            )}
            {visibleThread.map(turn => (
              <div
                key={turn.id}
                style={turn.role === 'user' ? { fontWeight: 650 } : { opacity: 0.88 }}
              >
                {turn.text}
              </div>
            ))}
            {recording ? (
              <div style={{ fontStyle: 'italic', opacity: 0.85 }}>Listening… click ■ to send</div>
            ) : heard?.error ? (
              <div style={{ color: '#ffd0d0', fontWeight: 600 }}>{heard.text}</div>
            ) : heard && heard.id < 0 ? (
              <div style={{ fontStyle: 'italic', opacity: 0.85 }}>{heard.text}</div>
            ) : working || visibleThread.at(-1)?.role === 'user' ? (
              <div style={{ fontStyle: 'italic', opacity: 0.7 }}>Thinking…</div>
            ) : null}
            <div ref={threadEndRef} />
          </div>
        )}
      </div>
    </div>
  )
}

import { useCallback, useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'

import type { ListenOverlayState, ListenTarget } from '@/store/listen-overlay'

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

interface DragState {
  pointerId: number
  width: number
  height: number
}

/**
 * Header-drag hook for the listen overlay, mirroring useKirsinHeaderDrag: a
 * plain primary-button press-and-drag on the (non-button) header, driving
 * begin-move / move-by / end-move so main re-pins the window at
 * cursor-minus-grab-offset on every move.
 */
function useListenOverlayDrag() {
  const stateRef = useRef<DragState | null>(null)
  const targetRef = useRef<HTMLElement | null>(null)

  const end = useCallback((sendEndMove: boolean) => {
    const state = stateRef.current

    if (!state) {
      return
    }

    if (sendEndMove) {
      window.hermesDesktop?.listenOverlay?.endMove?.()
    }

    try {
      if (targetRef.current?.hasPointerCapture?.(state.pointerId)) {
        targetRef.current.releasePointerCapture?.(state.pointerId)
      }
    } catch {
      // Pointer cancellation may invalidate the id before React cleans up.
    }

    stateRef.current = null
    targetRef.current = null
  }, [])

  const onPointerDown = useCallback((event: React.PointerEvent<HTMLElement>) => {
    if (event.button !== 0 || (event.target as HTMLElement).closest?.('[data-listen-ctl]')) {
      return
    }

    const state = { width: window.outerWidth, height: window.outerHeight, pointerId: event.pointerId }

    stateRef.current = state
    targetRef.current = event.currentTarget

    try {
      event.currentTarget.setPointerCapture?.(state.pointerId)
    } catch {
      // A failed capture must not abort the drag — the window listeners below
      // keep it alive regardless.
    }

    window.hermesDesktop?.listenOverlay?.beginMove?.()
  }, [])

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      const state = stateRef.current

      if (!state || event.pointerId !== state.pointerId) {
        return
      }

      event.preventDefault()
      window.hermesDesktop?.listenOverlay?.moveBy?.({ width: state.width, height: state.height })
    }

    const onUp = (event: PointerEvent | MouseEvent) => {
      const state = stateRef.current
      const pointerId = 'pointerId' in event ? event.pointerId : state?.pointerId

      if (!state || pointerId !== state.pointerId) {
        return
      }

      end(true)
    }

    window.addEventListener('pointermove', onMove, true)
    window.addEventListener('pointerup', onUp, true)
    window.addEventListener('mouseup', onUp, true)

    return () => {
      window.removeEventListener('pointermove', onMove, true)
      window.removeEventListener('pointerup', onUp, true)
      window.removeEventListener('mouseup', onUp, true)
    }
  }, [end])

  useEffect(() => () => end(false), [end])

  return { onPointerDown }
}

const STATUS_LABEL: Record<string, string> = {
  idle: 'Idle',
  starting: 'Starting…',
  listening: 'Listening',
  stopping: 'Finishing…',
  done: 'Done',
  error: 'Error'
}

function SourceOption({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }) {
  const [hover, setHover] = useState(false)

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onPointerDown={e => e.stopPropagation()}
      style={{
        background: active
          ? 'color-mix(in srgb, var(--color-accent) 35%, transparent)'
          : hover
            ? 'var(--color-border)'
            : 'transparent',
        cursor: 'pointer',
        fontSize: 12,
        overflow: 'hidden',
        padding: '6px 8px',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap'
      }}
    >
      {label}
    </div>
  )
}

interface SourcePickerProps {
  disabled: boolean
  onChange: (pid: number) => void
  targets: ListenTarget[]
  value: number
}

/**
 * Custom dropdown instead of a native `<select>`: a native select's open
 * popup is drawn by the OS/Chromium popup layer, which ignores this
 * (frameless, fixed-size) window's own bounds — the list rendered outside
 * the overlay entirely, spilling past its edges. This one is a plain
 * absolutely-positioned list inside the app's own clipped
 * (`overflow: hidden`) container, so it can never spill past the HUD.
 */
function SourcePicker({ disabled, onChange, targets, value }: SourcePickerProps) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const selected = value === 0 ? null : targets.find(t => t.pid === value)
  const label = selected?.label ?? 'System audio (everything)'

  useEffect(() => {
    if (!open) {
      return
    }

    const onDocPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false)
      }
    }

    document.addEventListener('pointerdown', onDocPointerDown, true)

    return () => document.removeEventListener('pointerdown', onDocPointerDown, true)
  }, [open])

  const pick = (pid: number) => {
    onChange(pid)
    setOpen(false)
  }

  return (
    <div data-listen-ctl ref={rootRef} style={{ flex: 1, minWidth: 0, position: 'relative' }}>
      <button
        disabled={disabled}
        onClick={() => setOpen(v => !v)}
        onPointerDown={e => e.stopPropagation()}
        style={{
          alignItems: 'center',
          background: 'var(--color-card)',
          border: '1px solid var(--color-border)',
          borderRadius: 6,
          color: 'var(--color-foreground)',
          cursor: disabled ? 'default' : 'pointer',
          display: 'flex',
          fontSize: 12,
          justifyContent: 'space-between',
          opacity: disabled ? 0.6 : 1,
          overflow: 'hidden',
          padding: '3px 6px',
          textAlign: 'left',
          width: '100%'
        }}
        type="button"
      >
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>
        <span style={{ color: 'var(--color-muted-foreground)', flexShrink: 0, marginLeft: 6 }}>▾</span>
      </button>
      {open && (
        <div
          style={{
            background: 'var(--color-card)',
            border: '1px solid var(--color-border)',
            borderRadius: 6,
            boxShadow: '0 8px 20px rgba(0,0,0,0.4)',
            left: 0,
            maxHeight: 160,
            overflowY: 'auto',
            position: 'absolute',
            right: 0,
            top: 'calc(100% + 4px)',
            zIndex: 10
          }}
        >
          <SourceOption active={value === 0} label="System audio (everything)" onClick={() => pick(0)} />
          {targets.map(t => (
            <SourceOption active={value === t.pid} key={t.pid} label={t.label} onClick={() => pick(t.pid)} />
          ))}
        </div>
      )}
    </div>
  )
}

const DEVICE_OPTIONS: Array<{ value: 'auto' | 'gpu' | 'cpu'; label: string }> = [
  { value: 'auto', label: 'Auto (GPU, falls back to CPU)' },
  { value: 'gpu', label: 'GPU' },
  { value: 'cpu', label: 'CPU' }
]

interface DevicePickerProps {
  disabled: boolean
  onChange: (pref: 'auto' | 'gpu' | 'cpu') => void
  value: 'auto' | 'gpu' | 'cpu'
}

/** Same clipped-popup pattern as SourcePicker, for the fixed 3-way device choice. */
function DevicePicker({ disabled, onChange, value }: DevicePickerProps) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const label = DEVICE_OPTIONS.find(o => o.value === value)?.label ?? 'Auto'

  useEffect(() => {
    if (!open) {
      return
    }

    const onDocPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false)
      }
    }

    document.addEventListener('pointerdown', onDocPointerDown, true)

    return () => document.removeEventListener('pointerdown', onDocPointerDown, true)
  }, [open])

  const pick = (pref: 'auto' | 'gpu' | 'cpu') => {
    onChange(pref)
    setOpen(false)
  }

  return (
    <div data-listen-ctl ref={rootRef} style={{ flex: 1, minWidth: 0, position: 'relative' }}>
      <button
        disabled={disabled}
        onClick={() => setOpen(v => !v)}
        onPointerDown={e => e.stopPropagation()}
        style={{
          alignItems: 'center',
          background: 'var(--color-card)',
          border: '1px solid var(--color-border)',
          borderRadius: 6,
          color: 'var(--color-foreground)',
          cursor: disabled ? 'default' : 'pointer',
          display: 'flex',
          fontSize: 12,
          justifyContent: 'space-between',
          opacity: disabled ? 0.6 : 1,
          overflow: 'hidden',
          padding: '3px 6px',
          textAlign: 'left',
          width: '100%'
        }}
        type="button"
      >
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>
        <span style={{ color: 'var(--color-muted-foreground)', flexShrink: 0, marginLeft: 6 }}>▾</span>
      </button>
      {open && (
        <div
          style={{
            background: 'var(--color-card)',
            border: '1px solid var(--color-border)',
            borderRadius: 6,
            boxShadow: '0 8px 20px rgba(0,0,0,0.4)',
            left: 0,
            overflowY: 'auto',
            position: 'absolute',
            right: 0,
            top: 'calc(100% + 4px)',
            zIndex: 10
          }}
        >
          {DEVICE_OPTIONS.map(o => (
            <SourceOption active={value === o.value} key={o.value} label={o.label} onClick={() => pick(o.value)} />
          ))}
        </div>
      )}
    </div>
  )
}

interface ModeToggleProps {
  disabled: boolean
  onChange: (mode: 'subtitle' | 'dictate') => void
  value: 'subtitle' | 'dictate'
}

/**
 * Subtitle/Dictate — a two-way segmented control (not a dropdown: there are
 * only ever two choices, and the distinction is important enough to always
 * be visible without opening anything). Subtitle keeps the transcript in
 * this window only; Dictate sends the corrected final transcript to Kirsin
 * as a real prompt, the moment capture stops. Deliberately NOT persisted
 * (see ListenMode's doc comment in electron/listen-overlay.ts) — every open
 * starts back at Subtitle.
 */
function ModeToggle({ disabled, onChange, value }: ModeToggleProps) {
  const optionStyle = (active: boolean): CSSProperties => ({
    background: active ? 'var(--color-accent)' : 'transparent',
    border: 'none',
    borderRadius: 5,
    color: active ? 'var(--color-foreground)' : 'var(--color-muted-foreground)',
    cursor: disabled ? 'default' : 'pointer',
    flex: 1,
    fontSize: 12,
    fontWeight: active ? 600 : 400,
    padding: '3px 8px'
  })

  return (
    <div
      data-listen-ctl
      style={{
        background: 'var(--color-card)',
        border: '1px solid var(--color-border)',
        borderRadius: 6,
        display: 'flex',
        flex: 1,
        gap: 2,
        padding: 2
      }}
    >
      <button
        disabled={disabled}
        onClick={() => onChange('subtitle')}
        onPointerDown={e => e.stopPropagation()}
        style={optionStyle(value === 'subtitle')}
        type="button"
      >
        Subtitle
      </button>
      <button
        disabled={disabled}
        onClick={() => onChange('dictate')}
        onPointerDown={e => e.stopPropagation()}
        style={optionStyle(value === 'dictate')}
        type="button"
      >
        Dictate
      </button>
    </div>
  )
}

/**
 * The listen overlay's only view: a compact always-on-top HUD. Ctrl+Shift+L
 * opens it idle so you can pick a source first; Start (or the shortcut
 * again) begins rolling live captions, then the corrected final transcript
 * once you stop. This IS the delivery surface — the transcript is never
 * submitted anywhere as a prompt, so it stays here to read, scroll or copy
 * until you close the window or start a fresh capture (mirrors state pushed
 * from main, electron/listen-overlay.ts, over IPC. No gateway of its own).
 * The background is a deliberately visible tint (not full opacity, so the
 * HUD still reads as an overlay) but strong enough to read against a plain
 * white window behind it, not just dark ones.
 */
export function ListenOverlayApp() {
  const [state, setState] = useState<ListenOverlayState>(INITIAL_STATE)
  const { onPointerDown } = useListenOverlayDrag()
  const transcriptRef = useRef<HTMLDivElement>(null)
  const stickToBottomRef = useRef(true)

  useEffect(() => {
    const off = window.hermesDesktop?.listenOverlay?.onState(payload => setState(payload))

    return off
  }, [])

  // Auto-scroll to the newest caption while listening, but stop the instant
  // the user scrolls up to read something earlier — a live feed that yanks
  // your scroll position back down mid-read is worse than not scrolling.
  useEffect(() => {
    const el = transcriptRef.current

    if (el && stickToBottomRef.current) {
      el.scrollTop = el.scrollHeight
    }
  })

  const onTranscriptScroll = () => {
    const el = transcriptRef.current

    if (!el) {
      return
    }

    stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
  }

  const close = () => {
    window.hermesDesktop?.listenOverlay?.close()
  }

  const start = () => {
    window.hermesDesktop?.listenOverlay?.start()
  }

  const onSelectTarget = (pid: number) => {
    window.hermesDesktop?.listenOverlay?.selectTarget?.(pid)
  }

  const onSelectDevice = (pref: 'auto' | 'gpu' | 'cpu') => {
    window.hermesDesktop?.listenOverlay?.selectDevice?.(pref)
  }

  const onSelectMode = (mode: 'subtitle' | 'dictate') => {
    window.hermesDesktop?.listenOverlay?.selectMode?.(mode)
  }

  const isListening = state.status === 'listening' || state.status === 'starting'
  const isBusy = isListening || state.status === 'stopping'
  const isIdle = state.status === 'idle'
  const caption = state.status === 'done' ? state.finalText || state.caption : state.caption

  return (
    <div
      style={{
        background: 'color-mix(in srgb, var(--color-card) 92%, transparent)',
        border: '1px solid var(--color-border)',
        borderRadius: 10,
        boxShadow: '0 12px 32px rgba(0,0,0,0.35)',
        color: 'var(--color-foreground)',
        display: 'flex',
        flexDirection: 'column',
        fontSize: 12,
        height: '100vh',
        overflow: 'hidden',
        userSelect: 'none',
        width: '100vw'
      }}
    >
      <div
        onPointerDown={onPointerDown}
        style={{
          alignItems: 'center',
          borderBottom: '1px solid var(--color-border)',
          cursor: 'grab',
          display: 'flex',
          flexShrink: 0,
          gap: 8,
          justifyContent: 'space-between',
          padding: '8px 12px'
        }}
      >
        <div style={{ alignItems: 'center', display: 'flex', gap: 8 }}>
          <span
            style={{
              background: isListening ? '#ef4444' : state.status === 'error' ? '#f59e0b' : 'var(--color-muted-foreground)',
              borderRadius: '50%',
              display: 'inline-block',
              flexShrink: 0,
              height: 8,
              width: 8,
              ...(isListening ? { animation: 'listen-overlay-pulse 1.4s ease-in-out infinite' } : {})
            }}
          />
          <span style={{ fontWeight: 600 }}>Listening{state.lang ? ` · ${state.lang}` : ''}</span>
        </div>
        <div style={{ alignItems: 'center', display: 'flex', gap: 10 }}>
          <span style={{ color: 'var(--color-muted-foreground)' }}>{STATUS_LABEL[state.status] ?? state.status}</span>
          <button
            aria-label="Close"
            data-listen-ctl
            onClick={close}
            onPointerDown={e => e.stopPropagation()}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--color-muted-foreground)',
              cursor: 'pointer',
              fontSize: 14,
              lineHeight: 1,
              padding: 2
            }}
            type="button"
          >
            ✕
          </button>
        </div>
      </div>

      <div
        style={{
          alignItems: 'center',
          borderBottom: '1px solid var(--color-border)',
          display: 'flex',
          flexShrink: 0,
          gap: 8,
          padding: '6px 12px 0 12px',
          position: 'relative'
        }}
      >
        <span style={{ color: 'var(--color-muted-foreground)', flexShrink: 0 }}>Source</span>
        <SourcePicker disabled={isBusy} onChange={onSelectTarget} targets={state.targets} value={state.selectedPid} />
        {isIdle && (
          <button
            data-listen-ctl
            onClick={start}
            onPointerDown={e => e.stopPropagation()}
            style={{
              background: 'var(--color-accent)',
              border: 'none',
              borderRadius: 6,
              color: 'var(--color-foreground)',
              cursor: 'pointer',
              flexShrink: 0,
              fontSize: 12,
              fontWeight: 600,
              padding: '4px 10px'
            }}
            type="button"
          >
            Start
          </button>
        )}
      </div>

      <div
        style={{
          alignItems: 'center',
          borderBottom: '1px solid var(--color-border)',
          display: 'flex',
          flexShrink: 0,
          gap: 8,
          padding: '6px 12px',
          position: 'relative'
        }}
      >
        <span style={{ color: 'var(--color-muted-foreground)', flexShrink: 0 }}>Device</span>
        <DevicePicker disabled={isBusy} onChange={onSelectDevice} value={state.devicePref} />
      </div>

      <div
        style={{
          alignItems: 'center',
          borderBottom: '1px solid var(--color-border)',
          display: 'flex',
          flexShrink: 0,
          gap: 8,
          padding: '6px 12px',
          position: 'relative'
        }}
      >
        <span style={{ color: 'var(--color-muted-foreground)', flexShrink: 0 }}>Mode</span>
        <ModeToggle disabled={isBusy} onChange={onSelectMode} value={state.mode} />
      </div>

      <div
        onScroll={onTranscriptScroll}
        ref={transcriptRef}
        style={{
          flex: 1,
          lineHeight: 1.5,
          minHeight: 0,
          overflowY: 'auto',
          padding: '10px 14px'
        }}
      >
        {state.error ? (
          <span style={{ color: '#f59e0b' }}>{state.error}</span>
        ) : caption ? (
          <span style={{ fontSize: 13, whiteSpace: 'pre-wrap' }}>{caption}</span>
        ) : (
          <span style={{ color: 'var(--color-muted-foreground)' }}>
            {isIdle
              ? 'Pick a source, then press Start (or ⌃⇧L again).'
              : isListening
                ? 'Listening…'
                : 'Press Ctrl+Shift+L again to stop.'}
          </span>
        )}
      </div>

      <div
        style={{
          borderTop: '1px solid var(--color-border)',
          color: 'var(--color-muted-foreground)',
          flexShrink: 0,
          fontSize: 11,
          padding: '6px 12px'
        }}
      >
        {state.t ? `${Math.round(state.t)}s` : ''}
        {state.device ? ` · ${state.device.toUpperCase()}` : ''}
        {state.deviceNote ? ` (${state.deviceNote})` : ''}
        {state.status === 'done'
          ? state.mode === 'dictate'
            ? ' · sent to Kirsin · press ⌃⇧L to listen again'
            : ' · press ⌃⇧L to listen again'
          : ''}
      </div>

      <style>{`
        @keyframes listen-overlay-pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.35; }
        }
      `}</style>
    </div>
  )
}

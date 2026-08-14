// Local-only hitch-capture surface (U4).
//
// One action either way: "Start capture" arms the renderer, main and gateway
// rings; "Stop & export" gathers all three, writes a sanitized bundle under
// userData and reports where it landed plus how the exporter classified it.
// There is no separate export step and nothing is uploaded — the user attaches
// the directory by hand or does not.
//
// Deliberately minimal: no charts, no live event feed. Anything richer would
// have to sample while a capture is running, which is exactly the cost the
// whole design exists to avoid. Copy is hardcoded English, matching
// `uninstall-section.tsx` — this is a support/triage surface, not product UI.

import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import type { DesktopDiagnosticsExport } from '@/global'
import { Activity, Bug, Loader2 } from '@/lib/icons'

import { ListRow, Pill, SectionHeading, SettingsContent } from './primitives'

/** Plain-English gloss per R3 label, so the classification is actionable
 *  without opening classification.json. */
const LABEL_HINTS: Record<string, string> = {
  'gateway-bound': 'The backend event loop stalled — work on the gateway blocked everything downstream.',
  'history-bound': 'Commit cost tracked transcript length — long sessions, not big updates.',
  'ipc-transport-bound': 'Requests to the backend failed or timed out — the UI was waiting, not busy.',
  'memory-gc-bound': 'Renderer heap grew or sat near its limit — GC pressure.',
  'renderer-bound': 'The renderer main thread was busy — long animation frames.',
  unclassified: 'Nothing crossed a threshold. Either the capture missed the hitch, or it was mild.'
}

export function DiagnosticsSettings() {
  const bridge = window.hermesDesktop?.diagnosticsCapture

  const [armed, setArmed] = useState(false)
  const [captureId, setCaptureId] = useState<null | string>(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<DesktopDiagnosticsExport | null>(null)
  const [error, setError] = useState<null | string>(null)

  useEffect(() => {
    let alive = true

    if (!bridge) {
      return
    }

    void bridge
      .status()
      .then(status => {
        if (alive) {
          setArmed(Boolean(status?.armed))
          setCaptureId(status?.captureId ?? null)
        }
      })
      .catch(() => {
        // A status probe that fails tells us nothing worth showing; the
        // buttons still work and will report their own errors.
      })

    return () => {
      alive = false
    }
  }, [bridge])

  if (!bridge) {
    return null
  }

  const start = async () => {
    setBusy(true)
    setError(null)
    setResult(null)

    try {
      const started = await bridge.start()

      setArmed(true)
      setCaptureId(started?.captureId ?? null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const stop = async () => {
    setBusy(true)
    setError(null)

    try {
      const exported = await bridge.stop()

      setArmed(false)
      setCaptureId(null)
      setResult(exported)

      if (!exported) {
        setError('No capture was running.')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <SettingsContent>
      <SectionHeading icon={Activity} title="Diagnostics capture" />

      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        Records renderer, app and gateway timing into memory while armed, then writes a sanitized bundle to this
        machine. Sizes, counts and durations only — no message text, tool output, credentials or paths. Nothing is
        uploaded.
      </p>

      <ListRow
        action={
          <div className="flex flex-wrap items-center justify-end gap-2">
            <Button disabled={busy || armed} onClick={() => void start()} size="sm" type="button" variant="outline">
              {busy && !armed && <Loader2 className="size-3 animate-spin" />}
              Start capture
            </Button>
            <Button disabled={busy || !armed} onClick={() => void stop()} size="sm" type="button" variant="outline">
              {busy && armed && <Loader2 className="size-3 animate-spin" />}
              Stop &amp; export
            </Button>
          </div>
        }
        description={
          armed
            ? 'Recording. Reproduce the hitch, then stop and export.'
            : 'Start before the hitch — the buffer holds roughly the last five minutes.'
        }
        title={
          <span className="flex items-center gap-2">
            Capture
            <Pill tone={armed ? 'primary' : 'muted'}>{armed ? 'Armed' : 'Idle'}</Pill>
          </span>
        }
      />

      {captureId && (
        <p className="mt-1 font-mono text-[0.68rem] text-muted-foreground/60">Capture ID: {captureId}</p>
      )}

      {error && <p className="mt-2 text-xs text-destructive">{error}</p>}

      {result && (
        <div className="mt-4 rounded-xl border border-border/60 bg-background/40 px-4 py-3">
          <SectionHeading icon={Bug} title="Last export" />

          <p className="text-xs text-muted-foreground">
            {result.labels.map(label => LABEL_HINTS[label] ?? label).join(' ')}
          </p>

          <div className="mt-2 flex flex-wrap gap-1.5">
            {result.labels.map(label => (
              <Pill key={label} tone={label === 'unclassified' ? 'muted' : 'warn'}>
                {label}
              </Pill>
            ))}
          </div>

          <p className="mt-2 text-xs text-muted-foreground">
            {result.streams
              .map(stream => `${stream.name}: ${stream.absent ? `absent (${stream.absent})` : `${stream.events} events`}`)
              .join(' · ')}
          </p>

          <p className="mt-2 font-mono text-[0.68rem] break-all text-muted-foreground/60">{result.directory}</p>
        </div>
      )}
    </SettingsContent>
  )
}

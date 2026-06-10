import { useState } from 'react'
import { type CSSProperties } from 'react'
import { Button } from '../components/button'
import { launchHermesDesktop } from '../store'
import { Rocket, AlertCircle } from 'lucide-react'

/*
 * Success screen. HERMES AGENT wordmark stays as the visual anchor
 * (same Collapse Bold treatment as Welcome + the desktop chat intro),
 * with a status line below.
 *
 * Launching the desktop can fail (e.g. Stage-Desktop was skipped and
 * Hermes.exe doesn't exist). We catch the Tauri error and surface it
 * inline rather than silently doing nothing — the previous version
 * had `onClick={() => void launchHermesDesktop()}` which swallowed
 * the rejection and left the user staring at an unresponsive button.
 */
export default function Success() {
  const [error, setError] = useState<string | null>(null)
  const [launching, setLaunching] = useState(false)

  async function handleLaunch() {
    setError(null)
    setLaunching(true)
    try {
      await launchHermesDesktop()
      // On success the installer exits — control never returns here.
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
      setLaunching(false)
    }
  }

  return (
    <div className="hermes-fade-in flex h-full flex-col items-center justify-center gap-8 px-12 py-10">
      <div className="w-full max-w-2xl min-w-0 text-center">
        <p
          className="fit-text mx-auto mb-4 w-full font-['Collapse'] font-bold uppercase leading-[0.9] tracking-[0.08em] text-midground mix-blend-plus-lighter dark:text-foreground/90"
          style={
            {
              '--fit-text-line-height': '0.9',
              '--fit-text-max': '5rem',
              '--fit-text-min': '2.25rem'
            } as CSSProperties
          }
        >
          <span>
            <span>Hermes is ready</span>
          </span>
          <span aria-hidden="true">Hermes is ready</span>
        </p>

        <p className="m-0 text-center text-base leading-normal tracking-tight text-muted-foreground">
          You can launch from here, or any time from your terminal with{' '}
          <code className="rounded bg-muted/60 px-1 py-0.5 font-mono text-sm">
            hermes desktop
          </code>
          .
        </p>
      </div>

      <Button
        onClick={() => void handleLaunch()}
        size="lg"
        disabled={launching}
        className="inline-flex items-center gap-2 px-6"
      >
        <Rocket size={18} />
        {launching ? 'Launching…' : 'Launch Hermes'}
      </Button>

      {error && (
        <div
          role="alert"
          className="flex max-w-2xl items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive"
        >
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <div className="min-w-0">
            <div className="font-medium">Couldn&rsquo;t launch the desktop app</div>
            <div className="mt-1 text-destructive/80">{error}</div>
          </div>
        </div>
      )}
    </div>
  )
}

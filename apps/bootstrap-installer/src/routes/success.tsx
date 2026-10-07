import { AlertCircle } from 'lucide-react'
import { useState } from 'react'
import { type CSSProperties } from 'react'

import { HackeryButton } from '../components/hackery-button'
import { launchRabbitDesktop } from '../store'

/*
 * Success screen. RABBIT AGENT wordmark stays as the visual anchor
 * (same Collapse Bold treatment as Welcome + the desktop chat intro),
 * with a status line below.
 *
 * Launching the desktop can fail (e.g. Stage-Desktop was skipped and
 * Rabbit.exe doesn't exist). We catch the Tauri error and surface it
 * inline rather than silently doing nothing — the previous version
 * had `onClick={() => void launchRabbitDesktop()}` which swallowed
 * the rejection and left the user staring at an unresponsive button.
 */
export default function Success() {
  const [error, setError] = useState<string | null>(null)
  const [launching, setLaunching] = useState(false)

  async function handleLaunch() {
    setError(null)
    setLaunching(true)

    try {
      await launchRabbitDesktop()
      // On success the installer exits — control never returns here.
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
      setLaunching(false)
    }
  }

  return (
    <div className="rabbit-fade-in flex h-full flex-col items-center justify-center gap-8 px-12 py-10">
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
            <span>Rabbit is ready</span>
          </span>
          <span aria-hidden="true">Rabbit is ready</span>
        </p>

        <p className="m-0 text-center text-base leading-normal tracking-tight text-muted-foreground">
          You can launch from here, or any time from your terminal with{' '}
          <code className="font-mono text-sm text-foreground/80">rabbit desktop</code>.
        </p>
      </div>

      <HackeryButton
        disabled={launching}
        label={launching ? 'Launching' : 'Launch'}
        loading={launching}
        onClick={() => void handleLaunch()}
      />

      {error && (
        <div className="flex max-w-2xl items-start gap-2 text-sm" role="alert">
          <AlertCircle className="mt-0.5 shrink-0 text-destructive" size={16} />
          <div className="min-w-0">
            <div className="font-medium text-destructive">Couldn&rsquo;t launch the desktop app</div>
            <div className="mt-0.5 text-muted-foreground">{error}</div>
          </div>
        </div>
      )}
    </div>
  )
}

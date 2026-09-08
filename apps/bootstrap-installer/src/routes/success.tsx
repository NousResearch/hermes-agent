import { useStore } from '@nanostores/react'
import { AlertCircle } from 'lucide-react'
import { useState } from 'react'
import { type CSSProperties } from 'react'

import { HackeryButton } from '../components/hackery-button'
import { $bootstrap, launchNorthForge } from '../store'

/*
 * Success screen. NORTH FORGE wordmark as the visual anchor, the venv /
 * data paths that were created, and a Launch button that opens a North
 * Forge terminal (Rust runs north-forge.cmd — or the drive-root
 * "Start North Forge.lnk" — in a fresh console, then the installer exits).
 */
export default function Success() {
  const bootstrap = useStore($bootstrap)
  const [error, setError] = useState<string | null>(null)
  const [launching, setLaunching] = useState(false)

  async function handleLaunch() {
    setError(null)
    setLaunching(true)

    try {
      await launchNorthForge()
      // On success the installer exits — control never returns here.
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setLaunching(false)
    }
  }

  return (
    <div className="nf-fade-in flex h-full flex-col items-center justify-center gap-7 px-12 py-10">
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
            <span>North Forge is ready</span>
          </span>
          <span aria-hidden="true">North Forge is ready</span>
        </p>

        <p className="m-0 text-center text-base leading-normal tracking-tight text-muted-foreground">
          Launch from here, or any time by double-clicking{' '}
          <code className="font-mono text-sm text-foreground/80">north-forge.cmd</code> in the checkout
          (or <code className="font-mono text-sm text-foreground/80">Start North Forge.lnk</code> at the drive root).
        </p>
      </div>

      {(bootstrap.dataDir || bootstrap.venvDir) && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
          <dt className="text-muted-foreground">venv</dt>
          <dd className="font-mono text-foreground/80">{bootstrap.venvDir ?? '—'}</dd>
          <dt className="text-muted-foreground">data (HERMES_HOME)</dt>
          <dd className="font-mono text-foreground/80">{bootstrap.dataDir ?? '—'}</dd>
        </dl>
      )}

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
            <div className="font-medium text-destructive">Couldn&rsquo;t open a North Forge terminal</div>
            <div className="mt-0.5 text-muted-foreground">{error}</div>
          </div>
        </div>
      )}
    </div>
  )
}

import { useStore } from '@nanostores/react'
import { open } from '@tauri-apps/plugin-dialog'
import { AlertTriangle, FolderOpen, HardDrive } from 'lucide-react'

import { Button } from '../components/button'
import { HackeryButton } from '../components/hackery-button'
import {
  $locationNote,
  $repo,
  $route,
  chooseLocation,
  startInstall
} from '../store'

/*
 * Location screen — shown before anything runs.
 *
 * North Forge's checkout is already on a drive; this screen just confirms
 * WHICH one and shows the sibling folders the bootstrap will create:
 *   <parent>\<leaf>-venv     the Python environment
 *   <parent>\<leaf>-data     the data folder (HERMES_HOME)
 *
 * The drive quick-picks refuse the system drive (North Forge is meant to
 * travel on a separate drive); an already-present checkout on the system
 * drive is surfaced with a warning rather than hard-blocked.
 */
export default function Location() {
  const repo = useStore($repo)
  const note = useStore($locationNote)

  const resolved = repo?.repoRoot ?? null
  const canInstall = Boolean(resolved && repo?.bootstrapScript)

  async function browse() {
    const picked = await open({ directory: true, multiple: false, title: 'Choose the North Forge checkout (or its drive)' })

    if (typeof picked === 'string') {
      await chooseLocation(picked)
    }
  }

  const sourceLabel: Record<string, string> = {
    env: 'from NORTH_FORGE_REPO_ROOT',
    exe: 'found next to Setup',
    scan: 'found by scanning your drives',
    picked: 'you chose this',
    none: ''
  }

  return (
    <div className="nf-fade-in flex h-full flex-col px-8 pt-7 pb-5">
      <div className="shrink-0">
        <h2 className="text-xl font-semibold tracking-tight">Where is North Forge?</h2>
        <p className="mt-1.5 text-sm text-muted-foreground">
          Setup runs <code className="font-mono text-xs">scripts\bootstrap-north-forge.ps1</code> against the
          checkout on your drive. Confirm the location below.
        </p>
      </div>

      <div className="mt-5 flex-1 overflow-y-auto">
        {resolved ? (
          <div className="rounded-md border border-(--stroke-nous) p-4">
            <div className="flex items-center gap-2 text-sm font-medium">
              <HardDrive className="shrink-0 text-muted-foreground" size={15} />
              <span className="truncate">{resolved}</span>
              {repo?.source && sourceLabel[repo.source] && (
                <span className="shrink-0 text-xs text-muted-foreground/70">({sourceLabel[repo.source]})</span>
              )}
            </div>

            <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
              <dt className="text-muted-foreground">venv</dt>
              <dd className="font-mono text-foreground/80">{repo?.venvDir ?? '—'}</dd>
              <dt className="text-muted-foreground">data (HERMES_HOME)</dt>
              <dd className="font-mono text-foreground/80">{repo?.dataDir ?? '—'}</dd>
            </dl>

            {repo?.bootstrapped && (
              <p className="mt-3 text-xs text-muted-foreground">
                This checkout is already bootstrapped. Running setup again re-checks the venv and
                rebuilds it if needed; your data folder is left untouched.
              </p>
            )}
            {repo?.onSystemDrive && (
              <p className="mt-3 flex items-start gap-1.5 text-xs text-(--dt-destructive,#c0473a)">
                <AlertTriangle className="mt-0.5 shrink-0" size={13} />
                This checkout is on the system drive. North Forge is meant to run from a separate
                drive that can travel between machines.
              </p>
            )}
          </div>
        ) : (
          <div className="rounded-md border border-dashed border-(--stroke-nous) p-4 text-sm text-muted-foreground">
            No North Forge checkout found automatically. Pick the drive it&rsquo;s on, or browse to the
            folder that contains <code className="font-mono text-xs">scripts\bootstrap-north-forge.ps1</code>.
          </div>
        )}

        {/* Drive quick-picks */}
        {repo?.drives && repo.drives.length > 0 && (
          <div className="mt-4">
            <div className="mb-1.5 text-xs text-muted-foreground">Drives</div>
            <div className="flex flex-wrap gap-2">
              {repo.drives.map((d) => (
                <button
                  className="inline-flex items-center gap-1.5 rounded-md border border-(--stroke-nous) px-2.5 py-1 text-xs transition-colors hover:border-primary/60 hover:bg-primary/[0.06] disabled:pointer-events-none disabled:opacity-40"
                  disabled={d.isSystem}
                  key={d.letter}
                  onClick={() => void chooseLocation(d.checkoutPath ?? d.path)}
                  title={d.isSystem ? 'System drive — not a valid target' : d.checkoutPath ?? d.path}
                  type="button"
                >
                  <HardDrive size={12} />
                  {d.letter}
                  {d.isSystem && <span className="text-muted-foreground/70">system</span>}
                  {d.hasCheckout && !d.isSystem && <span className="text-primary">✓ checkout</span>}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="mt-3">
          <Button className="gap-1.5" onClick={() => void browse()} size="sm" variant="outline">
            <FolderOpen size={13} />
            Browse…
          </Button>
        </div>

        {note && (
          <p className="mt-3 flex items-start gap-1.5 text-xs text-(--dt-destructive,#c0473a)" role="alert">
            <AlertTriangle className="mt-0.5 shrink-0" size={13} />
            {note}
          </p>
        )}
      </div>

      <div className="mt-5 flex shrink-0 items-center justify-between border-t border-(--stroke-nous) pt-4">
        <Button onClick={() => $route.set('welcome')} variant="text">
          Back
        </Button>
        <HackeryButton
          disabled={!canInstall}
          label="Install"
          onClick={() => void startInstall()}
        />
      </div>
    </div>
  )
}

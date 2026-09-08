import { useQuery } from '@hermes/plugin-sdk'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Box, CheckCircle2, Globe, PawPrint, Upload } from '@/lib/icons'
import {
  $worldEnabled,
  $worldOnboardingDismissed,
  $worldProjection,
  resetWorldProjection,
  setWorldOnboardingDismissed
} from '@/store/lunar-city'

import {
  $boardSlug,
  addComment,
  boardKey,
  createTask,
  fetchBoard,
  patchTask,
  reassignTask,
  reclaimTask
} from '../../plugins/kanban/api'

import { type DialogueSubject, DialogueTray } from './dialogue-tray'
import { DispatcherCube } from './dispatcher-cube'
import { createWorldActionRunner } from './world-actions'
import { WorldScene } from './world-scene'
import { bindWorldSources, refreshWorldProjection, storeWorldSyncSink } from './world-sync'

interface LocalWorldAsset {
  name: string
  size: number
  type: string
}

const BUILT_IN_ASSETS = ['Hermes pet companions', 'Lunar City starter scene', 'Built-in textures and props']
const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

export interface LunarCityProps {
  onNewSession?: (params: Record<string, unknown>) => Promise<unknown> | unknown
}

function assetKey(asset: LocalWorldAsset): string {
  return `${asset.name}\0${asset.size}\0${asset.type}`
}

export function LunarCity({ onNewSession }: LunarCityProps) {
  const enabled = useStore($worldEnabled)
  const onboardingDismissed = useStore($worldOnboardingDismissed)
  const projection = useStore($worldProjection)
  const boardSlug = useStore($boardSlug)

  const { data: board } = useQuery({
    enabled,
    queryFn: () => fetchBoard(false),
    queryKey: boardKey(boardSlug, false),
    refetchInterval: 60_000
  })

  const [localAssets, setLocalAssets] = useState<LocalWorldAsset[]>([])
  const [dialogueSubject, setDialogueSubject] = useState<DialogueSubject | null>(null)
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const actionRunner = useMemo(
    () =>
      createWorldActionRunner({
        kanban: { addComment, createTask, patchTask, reclaimTask, reassignTask },
        ...(onNewSession ? { createSession: onNewSession } : {})
      }),
    [onNewSession]
  )

  useEffect(() => {
    if (!enabled) {
      resetWorldProjection()

      return
    }

    const sink = storeWorldSyncSink()
    const dispose = bindWorldSources({}, sink)

    void refreshWorldProjection(async () => {
      const current = board ?? (await fetchBoard(false))

      return { tasks: current.columns.flatMap(column => column.tasks) }
    }, sink)

    return () => {
      dispose()
      resetWorldProjection()
    }
  }, [board, boardSlug, enabled])

  const addFiles = (files: FileList | File[]) => {
    const next = Array.from(files).map(file => ({ name: file.name, size: file.size, type: file.type }))

    setLocalAssets(current => {
      const seen = new Set(current.map(assetKey))

      return [...current, ...next.filter(asset => !seen.has(assetKey(asset)))]
    })
  }

  if (!enabled) {
    return (
      <section aria-label="World disabled" className="flex h-full items-center justify-center p-8" role="status">
        <div className="max-w-md rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary) p-6 text-center">
          <Globe className="mx-auto mb-3 size-8 text-muted-foreground" />
          <h1 className="text-lg font-semibold">World disabled</h1>
          <p className="mt-2 text-sm text-(--ui-text-tertiary)">
            Enable World in Settings → Plugins to make Lunar City available again.
          </p>
        </div>
      </section>
    )
  }

  return (
    <main aria-label="Lunar City world" className="h-full overflow-y-auto p-6">
      <div className="mx-auto flex max-w-4xl flex-col gap-6">
        <header>
          <div className="flex items-center gap-2 text-(--ui-text-tertiary)">
            <Globe className="size-4" />
            <span className="text-xs font-medium uppercase tracking-wide">Lunar City</span>
          </div>
          <h1 className="mt-2 text-2xl font-semibold">Your Hermes world</h1>
          <p className="mt-1 max-w-2xl text-sm text-(--ui-text-tertiary)">
            A living interface for Hermes work, agents, alerts, and celebrations. Every action stays connected to the
            real system.
          </p>
        </header>

        <WorldScene onSelectSubject={setDialogueSubject} projection={projection} />

        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,24rem)]">
          <DispatcherCube
            context={{
              actionRunner,
              conditions: projection.conditions,
              events: projection.recentEvents
            }}
            onSubjectSelected={setDialogueSubject}
          />
          {dialogueSubject ? (
            <DialogueTray
              onAction={intent => actionRunner.run(intent)}
              onClose={() => setDialogueSubject(null)}
              subject={dialogueSubject}
            />
          ) : (
            <div className="rounded-2xl border border-dashed border-(--ui-stroke-tertiary) p-4 text-sm text-(--ui-text-tertiary)">
              Select a worker, leader, task, or event to open its in-world dialogue.
            </div>
          )}
        </div>

        <section className="rounded-2xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-secondary) p-6">
          <div className="flex items-center gap-3">
            <div className="flex size-12 items-center justify-center rounded-xl bg-(--ui-bg-tertiary)">
              <PawPrint className="size-6 text-(--ui-text-secondary)" />
            </div>
            <div>
              <h2 className="font-medium">Hermes defaults are active</h2>
              <p className="text-sm text-(--ui-text-tertiary)">Built-in pets and starter assets are available.</p>
            </div>
          </div>
          <div className="mt-5 grid gap-2 sm:grid-cols-3">
            {BUILT_IN_ASSETS.map(asset => (
              <div className="flex items-center gap-2 rounded-lg bg-(--ui-bg-tertiary) px-3 py-2 text-sm" key={asset}>
                <CheckCircle2 className="size-4 text-(--ui-text-tertiary)" />
                <span>{asset}</span>
              </div>
            ))}
          </div>
        </section>

        {localAssets.length > 0 && (
          <section aria-label="Local assets" className="rounded-xl border border-(--ui-stroke-tertiary) p-4">
            <div className="flex items-center gap-2">
              <Box className="size-4" />
              <h2 className="font-medium">Local assets for review</h2>
            </div>
            <ul className="mt-3 space-y-1 text-sm text-(--ui-text-secondary)">
              {localAssets.map(asset => (
                <li key={assetKey(asset)}>{asset.name}</li>
              ))}
            </ul>
            <p className="mt-3 text-xs text-(--ui-text-tertiary)">
              Review only — files stay local to this window; no importer is connected.
            </p>
          </section>
        )}
      </div>

      <Dialog
        onOpenChange={open => {
          if (!open) {
            setWorldOnboardingDismissed()
          }
        }}
        open={!onboardingDismissed}
      >
        <DialogContent aria-describedby="lunar-city-world-setup-description" aria-label="Lunar City world setup">
          <DialogHeader>
            <DialogTitle>Lunar City world setup</DialogTitle>
            <DialogDescription id="lunar-city-world-setup-description">
              Start with Hermes built-in pets and assets, or stage local files for review.
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-3">
            <div className="rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) p-3">
              <div className="flex items-center gap-2 font-medium">
                <PawPrint className="size-4" />
                Hermes built-in pets and assets
              </div>
              <img
                alt="Hermes built-in pet"
                className="mt-3 size-16 rounded-lg object-cover"
                src={assetPath('hermes.png')}
              />
              <p className="mt-1 text-sm text-(--ui-text-tertiary)">
                The safe default is ready now; no files leave this device.
              </p>
            </div>

            <div
              aria-label="Drop local world assets"
              className="flex min-h-28 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-(--ui-stroke-tertiary) p-4 text-center outline-none transition hover:bg-(--ui-bg-tertiary) focus-visible:ring-2 focus-visible:ring-ring"
              data-testid="world-asset-drop-zone"
              onClick={() => fileInputRef.current?.click()}
              onDragEnter={event => {
                event.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDragOver={event => event.preventDefault()}
              onDrop={event => {
                event.preventDefault()
                setDragging(false)
                addFiles(event.dataTransfer.files)
              }}
              onKeyDown={event => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  fileInputRef.current?.click()
                }
              }}
              role="button"
              tabIndex={0}
            >
              <Upload className="mb-2 size-5 text-(--ui-text-tertiary)" />
              <span className="text-sm font-medium">
                {dragging ? 'Drop to stage for review' : 'Drop local world assets'}
              </span>
              <span className="mt-1 text-xs text-(--ui-text-tertiary)">GLB, GLTF, PNG, JPG, or WEBP</span>
              <input
                accept=".glb,.gltf,.png,.jpg,.jpeg,.webp"
                className="sr-only"
                multiple
                onChange={event => {
                  if (event.target.files) {
                    addFiles(event.target.files)
                  }
                }}
                ref={fileInputRef}
                type="file"
              />
            </div>

            {localAssets.length > 0 && (
              <p aria-live="polite" className="text-xs text-(--ui-text-tertiary)">
                Review only — {localAssets.length} local file{localAssets.length === 1 ? '' : 's'} staged; no importer
                is connected.
              </p>
            )}
          </div>

          <DialogFooter>
            <Button onClick={() => setWorldOnboardingDismissed()} type="button" variant="ghost">
              Skip for now
            </Button>
            <Button onClick={() => setWorldOnboardingDismissed()} type="button">
              Use Hermes defaults
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </main>
  )
}

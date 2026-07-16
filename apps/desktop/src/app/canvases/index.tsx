import type * as React from 'react'
import { useCallback, useEffect, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { getProfiles } from '@/hermes'
import { copyTextToClipboard, readDesktopDir, readDesktopFileText } from '@/lib/desktop-fs'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'

import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import { CanvasRenderer } from './canvas-renderer'
import { parseCanvasManifest } from './canvas-manifest'
import { canvasRefreshPrompt } from './canvas-policy'
import type { CanvasDefinition } from './types'

interface CanvasesViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

interface CanvasEntry {
  id: string
  label: string
  path: string
  profile: string
}

function canvasLabel(path: string): string {
  return (
    path
      .split(/[\\/]/)
      .pop()
      ?.replace(/\.canvas\.json$/i, '')
      .replace(/[-_]+/g, ' ')
      .replace(/\b\w/g, character => character.toUpperCase()) || path
  )
}

export function CanvasesView({
  className,
  setStatusbarItemGroup: _setStatusbarItemGroup,
  ...props
}: CanvasesViewProps) {
  const [canvases, setCanvases] = useState<CanvasEntry[] | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [canvas, setCanvas] = useState<CanvasDefinition | null>(null)
  const [canvasSourceId, setCanvasSourceId] = useState<string | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [refreshPromptCopied, setRefreshPromptCopied] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)

  const refreshCanvases = useCallback(async () => {
    setIsRefreshing(true)
    try {
      const { profiles } = await getProfiles()
      const results = await Promise.all(
        profiles.map(async profile => {
          const directory = await readDesktopDir(profile.path, profile.name)

          return directory.entries
            .filter(entry => !entry.isDirectory && /\.canvas\.json$/i.test(entry.name))
            .map<CanvasEntry>(entry => ({
              id: `${profile.name}:${entry.path}`,
              label: canvasLabel(entry.path),
              path: entry.path,
              profile: profile.name
            }))
        })
      )
      const nextCanvases = results.flat().sort((left, right) => left.label.localeCompare(right.label))

      setCanvases(current => {
        // A Canvas may be replaced atomically by the agent. Keep the last
        // known entry visible during the short write/rename window instead of
        // flashing back to the demo or removing a report mid-refresh.
        const retained =
          selectedId && !nextCanvases.some(canvas => canvas.id === selectedId)
            ? current?.filter(canvas => canvas.id === selectedId) || []
            : []

        return [...nextCanvases, ...retained]
      })
      setSelectedId(current =>
        current && nextCanvases.some(canvas => canvas.id === current) ? current : nextCanvases[0]?.id || null
      )
    } catch (error) {
      notifyError(error, 'No se pudieron cargar los canvases')
      setCanvases([])
    } finally {
      setIsRefreshing(false)
    }
  }, [selectedId])

  useEffect(() => {
    void refreshCanvases()
  }, [refreshCanvases])

  const selectedCanvas = canvases?.find(canvas => canvas.id === selectedId) || null

  useEffect(() => {
    let cancelled = false

    if (!selectedCanvas) {
      setCanvas(null)
      setCanvasSourceId(null)
      setLoadError(null)
      return
    }

    const isReplacingCurrentCanvas = canvasSourceId === selectedCanvas.id
    if (!isReplacingCurrentCanvas) {
      setCanvas(null)
      setLoadError(null)
    }
    void readDesktopFileText(selectedCanvas.path, selectedCanvas.profile)
      .then(file => {
        if (cancelled) return
        if (file.binary || file.truncated || !file.text) {
          throw new Error('El manifiesto del canvas no se puede leer')
        }
        setCanvas(parseCanvasManifest(file.text, selectedCanvas.profile))
        setCanvasSourceId(selectedCanvas.id)
        setLoadError(null)
      })
      .catch(error => {
        if (cancelled) return
        if (!isReplacingCurrentCanvas) {
          setLoadError(error instanceof Error ? error.message : 'No se pudo abrir este canvas')
        }
      })

    return () => {
      cancelled = true
    }
  }, [canvasSourceId, selectedCanvas])

  return (
    <section
      {...props}
      className={cn('flex h-full min-w-0 overflow-hidden bg-(--ui-chat-surface-background)', className)}
    >
      <aside className="hidden w-72 shrink-0 border-r border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background) md:block">
        <div className="flex items-center justify-between gap-2 px-4 pb-2 pt-[calc(var(--titlebar-height)+0.75rem)]">
          <div className="text-xs font-semibold tracking-[0.16em] text-(--ui-text-tertiary) uppercase">Canvases</div>
          <button
            className="rounded px-1.5 py-0.5 text-[0.6875rem] text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-(--ui-text-secondary) disabled:cursor-wait disabled:opacity-70"
            disabled={isRefreshing}
            onClick={() => void refreshCanvases()}
            type="button"
          >
            {isRefreshing ? 'Actualizando…' : 'Actualizar'}
          </button>
        </div>
        <div className="border-t border-(--ui-stroke-tertiary) px-2 py-2">
          {canvases === null ? (
            <div className="flex items-center gap-2 px-2 py-3 text-xs text-(--ui-text-tertiary)">
              <GlyphSpinner ariaLabel="Cargando canvases" />
              <span>Cargando canvases…</span>
            </div>
          ) : null}
          {canvases?.map(canvas => (
            <button
              aria-current={canvas.id === selectedId ? 'page' : undefined}
              className={cn(
                'group flex w-full items-start gap-2 rounded-md px-2 py-2 text-left outline-none transition-colors hover:bg-(--ui-control-hover-background) focus-visible:ring-1 focus-visible:ring-(--ui-focus-ring)',
                canvas.id === selectedId && 'bg-(--ui-row-active-background)'
              )}
              key={canvas.id}
              onClick={() => setSelectedId(canvas.id)}
              type="button"
            >
              <span className="mt-0.5 grid size-4 shrink-0 place-items-center text-(--ui-text-tertiary)">
                <Codicon name="graph" size="0.875rem" />
              </span>
              <span className="min-w-0">
                <span className="block truncate text-[0.8125rem] font-medium leading-5 text-(--ui-text-primary)">
                  {canvas.label}
                </span>
                <span className="block truncate text-[0.6875rem] leading-4 text-(--ui-text-tertiary)">
                  {canvas.profile || 'default'}
                </span>
              </span>
            </button>
          ))}
          {canvases?.length === 0 ? (
            <p className="px-2 py-3 text-xs leading-5 text-(--ui-text-tertiary)">
              Los canvases persistentes de los perfiles aparecerán aquí.
            </p>
          ) : null}
        </div>
      </aside>
      <main className="min-w-0 flex-1 overflow-y-auto px-5 pb-10 pt-[calc(var(--titlebar-height)+1.5rem)] sm:px-8">
        {canvases === null ? (
          <div className="grid h-full place-items-center">
            <div className="flex items-center gap-2 text-sm text-(--ui-text-secondary)">
              <GlyphSpinner ariaLabel="Cargando biblioteca de canvases" />
              <span>Cargando biblioteca de canvases…</span>
            </div>
          </div>
        ) : selectedCanvas ? (
          <div className="mx-auto max-w-6xl">
            <div className="mb-5 flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="text-xs font-semibold tracking-[0.16em] text-(--ui-text-tertiary) uppercase">
                  Canvas · {selectedCanvas.profile || 'default'}
                </div>
                <h1 className="mt-2 truncate text-2xl font-semibold tracking-tight text-(--ui-text-primary)">
                  {selectedCanvas.label}
                </h1>
              </div>
              <button
                className="rounded-md px-2 py-1 text-xs text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) disabled:cursor-wait disabled:opacity-70"
                disabled={isRefreshing}
                onClick={() => void refreshCanvases()}
                type="button"
              >
                {isRefreshing ? 'Actualizando…' : 'Actualizar'}
              </button>
            </div>
            {loadError ? (
              <div className="rounded-lg border border-(--ui-stroke-tertiary) p-4 text-sm text-(--ui-text-secondary)">
                {loadError}
              </div>
            ) : null}
            {!loadError && !canvas ? <div className="text-sm text-(--ui-text-secondary)">Abriendo canvas…</div> : null}
            {canvas ? (
              <div>
                <button
                  className="mb-4 rounded-md bg-(--ui-row-active-background) px-3 py-2 text-xs font-medium text-(--ui-text-primary) hover:bg-(--ui-control-hover-background)"
                  onClick={() => {
                    void copyTextToClipboard(canvasRefreshPrompt(canvas, selectedCanvas.path)).then(() => {
                      setRefreshPromptCopied(true)
                      window.setTimeout(() => setRefreshPromptCopied(false), 2_500)
                    })
                  }}
                  type="button"
                >
                  {refreshPromptCopied ? 'Instrucción copiada' : 'Actualizar con Hermes'}
                </button>
                <p className="mb-6 max-w-3xl text-sm leading-6 text-(--ui-text-secondary)">{canvas.summary}</p>
                <CanvasRenderer blocks={canvas.blocks} />
              </div>
            ) : null}
          </div>
        ) : (
          <div className="grid h-full place-items-center">
            <p className="max-w-sm text-center text-sm leading-6 text-(--ui-text-secondary)">
              Aún no hay canvases para los perfiles disponibles.
            </p>
          </div>
        )}
      </main>
    </section>
  )
}

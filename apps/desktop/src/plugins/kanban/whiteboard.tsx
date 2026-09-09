import '@excalidraw/excalidraw/index.css'

import { Excalidraw, serializeAsJSON } from '@excalidraw/excalidraw'
import type { ExcalidrawInitialDataState, ExcalidrawProps } from '@excalidraw/excalidraw/types'
import { host, Loader, useQuery, useTheme } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { $boardSlug, fetchWhiteboard, saveWhiteboard, type WhiteboardScene } from './api'
import { errText, useKanban } from './ui'

const SAVE_DELAY_MS = 800

type SaveState = 'clean' | 'dirty' | 'saving'

/** A board-scoped, single-user drawing surface. Excalidraw owns drawing and
 * export; Hermes owns persistence, isolation, error reporting, and lifecycle. */
export function KanbanWhiteboard() {
  const k = useKanban()
  const { renderedMode } = useTheme()
  const slug = $boardSlug.get()
  const [saveState, setSaveState] = useState<SaveState>('clean')
  const latestScene = useRef<null | WhiteboardScene>(null)
  const saveTimer = useRef<null | ReturnType<typeof setTimeout>>(null)

  const { data, error } = useQuery({
    queryFn: fetchWhiteboard,
    queryKey: ['kanban', 'whiteboard', slug],
    retry: 1,
    staleTime: Number.POSITIVE_INFINITY
  })

  const commit = async () => {
    const scene = latestScene.current

    if (!scene) {
      return
    }

    saveTimer.current = null
    setSaveState('saving')

    try {
      await saveWhiteboard(scene)

      if (latestScene.current === scene) {
        setSaveState('clean')
      } else {
        setSaveState('dirty')
        saveTimer.current = setTimeout(() => void commit(), SAVE_DELAY_MS)
      }
    } catch (saveError) {
      setSaveState('dirty')
      host.notify({ kind: 'error', message: k.whiteboardSaveError(errText(saveError)) })
    }
  }

  const onChange: NonNullable<ExcalidrawProps['onChange']> = (elements, appState, files) => {
    // Excalidraw's serializer removes transient UI state while retaining the
    // viewport, drawing, images, and other data needed for an exact restore.
    latestScene.current = JSON.parse(serializeAsJSON(elements, appState, files, 'local')) as WhiteboardScene
    setSaveState('dirty')

    if (saveTimer.current) {
      clearTimeout(saveTimer.current)
    }

    saveTimer.current = setTimeout(() => void commit(), SAVE_DELAY_MS)
  }

  useEffect(
    () => () => {
      if (saveTimer.current) {
        clearTimeout(saveTimer.current)
      }

      // Best-effort flush when switching views/boards. Normal edits take the
      // awaited debounced path above; this protects the final sub-second edit.
      if (latestScene.current) {
        void saveWhiteboard(latestScene.current)
      }
    },
    []
  )

  if (error) {
    return (
      <div className="grid flex-1 place-items-center px-4 text-center text-xs text-(--ui-text-secondary)">
        <div>
          <p className="font-medium text-foreground">{k.whiteboardLoadError}</p>
          <p className="mt-1">{errText(error)}</p>
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="grid flex-1 place-items-center">
        <Loader type="lemniscate-bloom" />
      </div>
    )
  }

  return (
    <div className="kanban-whiteboard relative min-h-0 flex-1 overflow-hidden border-t border-(--ui-stroke-secondary)">
      <Excalidraw
        autoFocus
        initialData={data.scene as ExcalidrawInitialDataState}
        onChange={onChange}
        theme={renderedMode}
        UIOptions={{
          canvasActions: {
            loadScene: false,
            saveToActiveFile: false
          }
        }}
      />
      <div
        aria-live="polite"
        className="pointer-events-none absolute right-3 bottom-3 rounded bg-(--ui-bg-primary) px-2 py-1 text-[0.625rem] text-(--ui-text-tertiary) shadow-sm"
      >
        {saveState === 'clean' ? k.whiteboardSaved : saveState === 'saving' ? k.whiteboardSaving : k.whiteboardUnsaved}
      </div>
    </div>
  )
}

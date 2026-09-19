import { useEffect, useRef, useState } from 'react'

import { CodeEditor } from '@/components/chat/code-editor'
import { Loader } from '@/components/ui/loader'
import { useI18n } from '@/i18n'
import { readDesktopFileText, writeDesktopFileText } from '@/lib/desktop-fs'
import { $workspaceChangeTick, notifyWorkspaceChanged } from '@/store/workspace-events'

import { clearIdeFileDirty, setIdeFileDirty } from './tabs'

type LoadState =
  | { kind: 'error'; message: string }
  | { kind: 'loading' }
  | { kind: 'ready'; text: string }
  | { kind: 'too-large' }

interface IdeFileEditorProps {
  path: string
}

/**
 * One open file, edited in the app's shared CodeMirror surface.
 *
 * Save is guarded: the disk is re-read first and a mismatch against the
 * baseline the user started from surfaces a conflict banner (overwrite /
 * reload) instead of silently clobbering — the same contract the file preview
 * uses, because an agent run is exactly the thing that edits files underneath
 * you. External changes also refresh clean tabs live via the workspace tick.
 */
export function IdeFileEditor({ path }: IdeFileEditorProps) {
  const { t } = useI18n()
  const [state, setState] = useState<LoadState>({ kind: 'loading' })
  const [conflict, setConflict] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<null | string>(null)
  // Remount key for the CodeMirror surface: it reads initialValue once, so new
  // content arrives by remounting rather than pushing a value in.
  const [reloadNonce, setReloadNonce] = useState(0)
  const baselineRef = useRef('')
  const draftRef = useRef('')

  // Load-buffer state lives in refs (the CodeMirror surface owns its own text;
  // these two are the disk snapshot and the live draft we diff against). Writes
  // go through this helper rather than inline in effects.
  const setBuffer = (text: string) => {
    baselineRef.current = text
    draftRef.current = text
  }

  useEffect(() => {
    let alive = true

    setState({ kind: 'loading' })
    setConflict(false)
    setSaveError(null)

    void readDesktopFileText(path)
      .then(result => {
        if (!alive) {
          return
        }

        if (result.truncated) {
          setState({ kind: 'too-large' })

          return
        }

        setBuffer(result.text)
        clearIdeFileDirty(path)
        setState({ kind: 'ready', text: result.text })
      })
      .catch((error: unknown) => {
        if (alive) {
          setState({ kind: 'error', message: error instanceof Error ? error.message : String(error) })
        }
      })

    return () => {
      alive = false
    }
  }, [path])

  // External change watcher: the workspace tick fires when a tool edits files.
  // A clean tab refreshes silently; a dirty tab surfaces the conflict banner so
  // neither editing side loses work invisibly. nanostores subscribes fire
  // immediately, so the tick captured at mount is skipped — the load effect
  // above already read the file.
  useEffect(() => {
    const mountTick = $workspaceChangeTick.get()

    const unsubscribe = $workspaceChangeTick.subscribe(tick => {
      if (tick === mountTick) {
        return
      }

      void readDesktopFileText(path)
        .then(result => {
          if (result.truncated || result.text === baselineRef.current) {
            return
          }

          if (draftRef.current !== baselineRef.current) {
            setConflict(true)

            return
          }

          setBuffer(result.text)
          clearIdeFileDirty(path)
          setState({ kind: 'ready', text: result.text })
          setReloadNonce(nonce => nonce + 1)
        })
        .catch(() => undefined)
    })

    return () => unsubscribe()
  }, [path])

  const save = async (force = false) => {
    if (saving) {
      return
    }

    setSaving(true)
    setSaveError(null)

    try {
      if (!force) {
        try {
          const current = await readDesktopFileText(path)

          if (!current.truncated && current.text !== baselineRef.current) {
            setConflict(true)
            setSaving(false)

            return
          }
        } catch {
          // Couldn't re-read for the check — fall through and attempt the write.
        }
      }

      await writeDesktopFileText(path, draftRef.current)
      baselineRef.current = draftRef.current
      clearIdeFileDirty(path)
      setConflict(false)
      notifyWorkspaceChanged(path)
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : String(error))
    } finally {
      setSaving(false)
    }
  }

  const reloadFromDisk = () => {
    setConflict(false)
    setSaveError(null)
    setReloadNonce(nonce => nonce + 1)
    // The effect above re-reads on the next mount; force it by re-running the
    // load path directly instead of waiting for a remount.
    void readDesktopFileText(path)
      .then(result => {
        if (result.truncated) {
          setState({ kind: 'too-large' })

          return
        }

        setBuffer(result.text)
        clearIdeFileDirty(path)
        setState({ kind: 'ready', text: result.text })
      })
      .catch((error: unknown) => {
        setState({ kind: 'error', message: error instanceof Error ? error.message : String(error) })
      })
  }

  if (state.kind === 'loading') {
    return (
      <div className="grid h-full place-items-center">
        <Loader />
      </div>
    )
  }

  if (state.kind === 'too-large') {
    return (
      <div className="px-4 py-6 text-xs text-muted-foreground">
        <div className="text-sm font-medium text-foreground">{t.ide.tooLargeTitle}</div>
        <div className="mt-1 leading-relaxed">{t.ide.tooLargeBody}</div>
      </div>
    )
  }

  if (state.kind === 'error') {
    return (
      <div className="px-4 py-6 text-xs text-muted-foreground">
        <div className="text-sm font-medium text-foreground">{t.ide.loadFailedTitle}</div>
        <div className="mt-1 leading-relaxed">{state.message}</div>
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      {conflict && (
        <div className="shrink-0 border-b border-amber-400/40 bg-amber-50 px-3 py-2 text-[0.7rem] text-amber-900 dark:border-amber-300/30 dark:bg-amber-300/10 dark:text-amber-100">
          <div className="font-semibold">{t.ide.diskChangedTitle}</div>
          <div className="mt-0.5 leading-relaxed">{t.ide.diskChangedBody}</div>
          <div className="mt-1.5 flex gap-3">
            <button
              className="font-bold underline underline-offset-4 transition-opacity hover:opacity-80"
              onClick={() => void save(true)}
              type="button"
            >
              {t.ide.overwrite}
            </button>
            <button
              className="font-bold underline underline-offset-4 transition-opacity hover:opacity-80"
              onClick={reloadFromDisk}
              type="button"
            >
              {t.ide.reloadFromDisk}
            </button>
          </div>
        </div>
      )}
      {saveError && (
        <div className="shrink-0 border-b border-destructive/40 bg-destructive/10 px-3 py-1.5 text-[0.7rem] text-destructive">
          {saveError}
        </div>
      )}
      <div className="min-h-0 flex-1">
        <CodeEditor
          disabled={saving}
          filePath={path}
          initialValue={state.text}
          key={`${path}:${reloadNonce}`}
          onChange={value => {
            draftRef.current = value
            setIdeFileDirty(path, value !== baselineRef.current)
          }}
          onSave={() => void save()}
        />
      </div>
    </div>
  )
}

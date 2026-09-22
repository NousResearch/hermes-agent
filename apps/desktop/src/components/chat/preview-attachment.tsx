import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { pickRevealLabel } from '@/app/right-sidebar/file-actions'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Download, ExternalLink, FolderOpen, MonitorPlay } from '@/lib/icons'
import { localPreviewTarget, normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { downloadGatewayMediaFile } from '@/lib/media'
import { previewName } from '@/lib/preview-targets'
import { $connectionsRegistry, ensureConnectionsRegistry, hasRegistryTopology } from '@/store/connection-registry-state'
import { notifyError } from '@/store/notifications'
import {
  $previewTabSources,
  closePreviewForSource,
  openPreview,
  type PreviewRecordSource,
  type PreviewTarget
} from '@/store/preview'
import { $connection, $sessionOwnerHintsRevision, $sessions } from '@/store/session'
import { $sessionRuntimeOwnerRevision, $sessionTiles, knownOwnerForSession } from '@/store/session-states'

interface ResolvedLocalFile {
  cwd: string
  path: string
  runtimeId: null | string
  storedId: null | string
  target: string
  url: string
}

function isAbsolutePath(path: string): boolean {
  return path.startsWith('/') || /^[a-z]:[\\/]/i.test(path) || path.startsWith('\\\\')
}

function localFileFromPreview(preview: PreviewTarget | null): Pick<ResolvedLocalFile, 'path' | 'url'> | null {
  if (
    preview?.kind !== 'file' ||
    !preview.path ||
    !isAbsolutePath(preview.path) ||
    !preview.url.startsWith('file://')
  ) {
    return null
  }

  return { path: preview.path, url: preview.url }
}

async function normalizeLocalActionTarget(rawTarget: string, cwd?: string): Promise<PreviewTarget | null> {
  const normalize = window.hermesDesktop?.normalizePreviewTarget

  // A present native resolver authoritatively checks that the path exists and
  // is readable. Preserve its null result; only old shells without this bridge
  // may use the renderer-side compatibility classifier.
  return normalize ? normalize(rawTarget, cwd) : localPreviewTarget(rawTarget, cwd)
}

export function PreviewAttachment({ source = 'manual', target }: { source?: PreviewRecordSource; target: string }) {
  const { t } = useI18n()
  const view = useSessionView()
  // This link lives in one session's transcript; resolve it against THAT
  // session's cwd, not the primary chat's.
  const cwd = useStore(view.$cwd)
  const runtimeId = useStore(view.$runtimeId)
  const storedId = useStore(view.$storedId)
  const connection = useStore($connection)
  useStore($connectionsRegistry)
  useStore($sessions)
  useStore($sessionTiles)
  useStore($sessionOwnerHintsRevision)
  useStore($sessionRuntimeOwnerRevision)
  const openSources = useStore($previewTabSources)
  const [resolvedLocalFile, setResolvedLocalFile] = useState<ResolvedLocalFile | null>(null)
  const [opening, setOpening] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [downloaded, setDownloaded] = useState(false)
  const cwdRef = useRef(cwd)
  const mountedRef = useRef(false)
  const requestTokenRef = useRef(0)
  const targetRef = useRef(target)
  const name = previewName(target)
  const isActive = openSources.includes(target)

  function currentSessionOwnsLocalFilesystem(): boolean {
    // Store updates precede React's next render. A retained button must not
    // dispatch a path resolved for the previous session or working directory.
    const liveRuntimeId = view.$runtimeId.get()
    const liveStoredId = view.$storedId.get()

    if (liveRuntimeId !== runtimeId || liveStoredId !== storedId || view.$cwd.get() !== cwd) {
      return false
    }

    const sessionId = liveRuntimeId ?? liveStoredId

    if (!sessionId) {
      return view.kind === 'primary' && !hasRegistryTopology() && $connection.get()?.mode === 'local'
    }

    const owner = knownOwnerForSession(sessionId)

    if (!owner) {
      return false
    }

    if (typeof owner === 'string') {
      return !hasRegistryTopology() && $connection.get()?.mode === 'local'
    }

    if (owner.mode) {
      return owner.mode === 'local'
    }

    return (
      $connectionsRegistry.get()?.connections.find(connection => connection.id === owner.connectionId)?.kind === 'local'
    )
  }

  const localFs =
    currentSessionOwnsLocalFilesystem() &&
    typeof window.hermesDesktop?.openExternal === 'function' &&
    typeof window.hermesDesktop.revealPath === 'function'

  const localFile =
    localFs &&
    resolvedLocalFile?.target === target &&
    resolvedLocalFile.cwd === cwd &&
    resolvedLocalFile.runtimeId === runtimeId &&
    resolvedLocalFile.storedId === storedId
      ? resolvedLocalFile
      : null

  const revealLabel = pickRevealLabel(t.fileMenu.revealFinder, t.fileMenu.revealExplorer, t.fileMenu.revealFileManager)

  cwdRef.current = cwd
  targetRef.current = target

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    mountedRef.current = true

    return () => {
      mountedRef.current = false
      requestTokenRef.current += 1
    }
  }, [])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    requestTokenRef.current += 1
    setOpening(false)
  }, [cwd, target])

  useEffect(() => {
    // A failed/unavailable registry read leaves mode-less owners fail-closed.
    void ensureConnectionsRegistry().catch(() => undefined)
  }, [])

  // Resolve before rendering native actions. Cleanup invalidates an in-flight
  // result on connection/owner mode, cwd, target, or unmount so a stale local
  // path can never be exposed by a later promise resolution.

  useEffect(() => {
    let cancelled = false

    setResolvedLocalFile(null)

    if (!localFs) {
      return () => {
        cancelled = true
      }
    }

    const requestCwd = cwd
    const requestTarget = target

    void normalizeLocalActionTarget(requestTarget, requestCwd || undefined)
      .then(preview => {
        const file = localFileFromPreview(preview)

        if (!cancelled && file) {
          setResolvedLocalFile({ ...file, cwd: requestCwd, runtimeId, storedId, target: requestTarget })
        }
      })
      .catch(() => {
        // Local actions fail closed. The unchanged preview action owns user-
        // visible resolver errors when the user explicitly invokes it.
      })

    return () => {
      cancelled = true
    }
  }, [cwd, localFs, runtimeId, storedId, target])

  async function togglePreview() {
    if (opening) {
      return
    }

    if (isActive) {
      closePreviewForSource(target)

      return
    }

    const requestToken = ++requestTokenRef.current
    const requestTarget = target
    const requestCwd = cwd

    setOpening(true)

    try {
      const preview = await normalizeOrLocalPreviewTarget(requestTarget, requestCwd || undefined)

      if (
        !mountedRef.current ||
        requestTokenRef.current !== requestToken ||
        targetRef.current !== requestTarget ||
        cwdRef.current !== requestCwd
      ) {
        return
      }

      if (!preview) {
        throw new Error(`Could not open preview target: ${requestTarget}`)
      }

      openPreview(preview, source)
    } catch (error) {
      if (
        !mountedRef.current ||
        requestTokenRef.current !== requestToken ||
        targetRef.current !== requestTarget ||
        cwdRef.current !== requestCwd
      ) {
        return
      }

      notifyError(error, t.preview.unavailable)
    } finally {
      if (mountedRef.current && requestTokenRef.current === requestToken) {
        setOpening(false)
      }
    }
  }

  async function downloadFile() {
    if (downloading) {
      return
    }

    setDownloading(true)

    try {
      // Works in both modes: the Electron main process fetches the bytes
      // through the session's backend connection (local gateway or remote)
      // and prompts for a save location.
      const result = await downloadGatewayMediaFile(target)

      if (mountedRef.current && result.saved) {
        setDownloaded(true)
        setTimeout(() => mountedRef.current && setDownloaded(false), 2000)
      }
    } catch (error) {
      if (mountedRef.current) {
        notifyError(error, t.fileMenu.downloadFailed)
      }
    } finally {
      if (mountedRef.current) {
        setDownloading(false)
      }
    }
  }

  async function openLocalFile() {
    const bridge = window.hermesDesktop

    if (!localFile || !currentSessionOwnsLocalFilesystem() || typeof bridge?.openExternal !== 'function') {
      return
    }

    try {
      await bridge.openExternal(localFile.url)
    } catch (error) {
      notifyError(error, t.preview.unavailable)
    }
  }

  async function revealLocalFile() {
    const bridge = window.hermesDesktop

    if (!localFile || !currentSessionOwnsLocalFilesystem() || typeof bridge?.revealPath !== 'function') {
      return
    }

    try {
      const revealed = await bridge.revealPath(localFile.path)

      if (!revealed) {
        throw new Error(`Could not reveal local file: ${localFile.path}`)
      }
    } catch (error) {
      notifyError(error, t.preview.unavailable)
    }
  }

  return (
    <div className="flex w-full max-w-160 items-center gap-2 rounded-lg border border-(--ui-stroke-tertiary) bg-card/55 px-2.5 py-1.5 text-sm">
      <span className="grid size-6 shrink-0 place-items-center rounded-md bg-muted/55 text-muted-foreground/85">
        <MonitorPlay className="size-3.5" />
      </span>
      <span className="min-w-0 flex-1 truncate text-[0.78rem] font-medium text-foreground/90" title={target}>
        {name}
      </span>
      <span className="flex min-w-0 flex-wrap items-center justify-end gap-1">
        {localFile && (
          <>
            <Button onClick={() => void openLocalFile()} size="xs" type="button" variant="outline">
              <ExternalLink />
              {t.fileMenu.openFile}
            </Button>
            <Button onClick={() => void revealLocalFile()} size="xs" type="button" variant="outline">
              <FolderOpen />
              {revealLabel}
            </Button>
          </>
        )}
        <Button
          aria-label={t.fileMenu.download}
          disabled={downloading}
          onClick={() => void downloadFile()}
          size="xs"
          type="button"
          variant="outline"
        >
          <Download />
          {downloaded ? t.fileMenu.downloadSaved : t.fileMenu.download}
        </Button>
        <Button disabled={opening} onClick={() => void togglePreview()} size="xs" type="button" variant="outline">
          {opening ? t.preview.opening : isActive ? t.preview.hide : t.preview.openPreview}
        </Button>
      </span>
    </div>
  )
}

import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useI18n } from '@/i18n'
import { isDesktopFsRemoteMode } from '@/lib/desktop-fs'
import { Download, MonitorPlay } from '@/lib/icons'
import { isLoopbackPreviewUrl, normalizeOrLocalPreviewTarget, openPreviewTargetInBrowser } from '@/lib/local-preview'
import { downloadGatewayMediaFile } from '@/lib/media'
import { previewOwnerIsAmbient } from '@/lib/preview-owner'
import { reachablePreviewUrl } from '@/lib/preview-reach'
import { previewName } from '@/lib/preview-targets'
import { $alwaysExternalLinks } from '@/store/external-links'
import { notifyError } from '@/store/notifications'
import { $previewTabSources, closePreviewForSource, openPreview } from '@/store/preview'

export function PreviewAttachment({ target }: { target: string }) {
  const { t } = useI18n()
  // This link lives in one session's transcript; resolve it against THAT
  // session's cwd, not the primary chat's.
  const view = useSessionView()
  const cwd = useStore(view.$cwd)
  const sessionId = useStore(view.$runtimeId)
  const openSources = useStore($previewTabSources)
  const alwaysExternal = useStore($alwaysExternalLinks)
  const [opening, setOpening] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [downloaded, setDownloaded] = useState(false)
  const cwdRef = useRef(cwd)
  const mountedRef = useRef(false)
  const requestTokenRef = useRef(0)
  const targetRef = useRef(target)
  const name = previewName(target)
  const isActive = openSources.includes(target)

  // A file on an SSH backend is not a file on this machine. Only web URLs
  // and HTML (staged locally by the preview bridge) can open externally.
  const browserCapable =
    previewOwnerIsAmbient(sessionId) &&
    (!isDesktopFsRemoteMode() || /^https?:\/\//i.test(target) || /\.html?(?:$|[?#])/i.test(target))

  const browserPreferred = alwaysExternal && browserCapable

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

  async function togglePreview(destination: 'default' | 'external' | 'in-app' = 'default') {
    if (opening) {
      return
    }

    const openExternally =
      browserCapable && (destination === 'external' || (destination === 'default' && browserPreferred))

    if (isActive && !openExternally) {
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

      const url = openExternally && preview.kind === 'url' ? await reachablePreviewUrl(preview.url) : preview.url

      if (
        !mountedRef.current ||
        requestTokenRef.current !== requestToken ||
        targetRef.current !== requestTarget ||
        cwdRef.current !== requestCwd
      ) {
        return
      }

      const reachable = !(isDesktopFsRemoteMode() && isLoopbackPreviewUrl(preview.url) && url === preview.url)

      if (
        openExternally &&
        reachable &&
        (preview.kind === 'url' || preview.previewKind === 'html' || !isDesktopFsRemoteMode())
      ) {
        await openPreviewTargetInBrowser(url === preview.url ? preview : { ...preview, url })
      } else {
        // Remote non-HTML files have no local browser URL; the gateway-backed
        // in-app pane is the only readable destination.
        openPreview(preview)
      }
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

  return (
    <div className="flex w-full max-w-160 items-center gap-2 rounded-lg border border-(--ui-stroke-tertiary) bg-card/55 px-2.5 py-1.5 text-sm">
      <span className="grid size-6 shrink-0 place-items-center rounded-md bg-muted/55 text-muted-foreground/85">
        <MonitorPlay className="size-3.5" />
      </span>
      <span className="min-w-0 flex-1 truncate text-[0.78rem] font-medium text-foreground/90" title={target}>
        {name}
      </span>
      <button
        aria-label={t.fileMenu.download}
        className="flex shrink-0 items-center gap-1 rounded-md border border-(--ui-stroke-tertiary) bg-background/40 px-2 py-1 text-[0.7rem] font-medium text-muted-foreground transition-colors hover:bg-accent/55 hover:text-foreground disabled:opacity-50"
        disabled={downloading}
        onClick={() => void downloadFile()}
        type="button"
      >
        <Download className="size-3" />
        {downloaded ? t.fileMenu.downloadSaved : t.fileMenu.download}
      </button>
      <button
        className="shrink-0 rounded-md border border-(--ui-stroke-tertiary) bg-background/40 px-2 py-1 text-[0.7rem] font-medium text-muted-foreground transition-colors hover:bg-accent/55 hover:text-foreground disabled:opacity-50"
        disabled={opening}
        onClick={() => void togglePreview()}
        type="button"
      >
        {opening
          ? t.preview.opening
          : browserPreferred
            ? t.preview.openInBrowser
            : isActive
              ? t.preview.hide
              : t.preview.openPreview}
      </button>
      {browserCapable && (
        <button
          className="shrink-0 rounded-md border border-(--ui-stroke-tertiary) bg-background/40 px-2 py-1 text-[0.7rem] font-medium text-muted-foreground transition-colors hover:bg-accent/55 hover:text-foreground disabled:opacity-50"
          disabled={opening}
          onClick={() => void togglePreview(browserPreferred ? 'in-app' : 'external')}
          type="button"
        >
          {browserPreferred ? (isActive ? t.preview.hide : t.preview.openPreview) : t.preview.openInBrowser}
        </button>
      )}
    </div>
  )
}

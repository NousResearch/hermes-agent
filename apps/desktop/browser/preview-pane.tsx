import type { ComponentProps } from 'react'
import { useEffect, useState } from 'react'

import { ArtifactPreview } from '../src/app/chat/right-rail/preview-artifact'
import { LocalFilePreview } from '../src/app/chat/right-rail/preview-file'
import type { PreviewPane as DesktopPreviewPane } from '../src/app/chat/right-rail/preview-pane'
import { useI18n } from '../src/i18n'

import { sharedMessage } from './messages'
import { backendFetch } from './transport'

type PreviewPaneProps = ComponentProps<typeof DesktopPreviewPane>

/** Browser guest content gets an opaque sandbox origin, never the backend's cookies. */
export function PreviewPane({ target, onClose, reloadRequest = 0 }: PreviewPaneProps) {
  if (target.kind === 'artifact') {return <ArtifactPreview onClose={onClose} target={target} />}

  if (target.kind === 'file' && target.previewKind !== 'html') {
    return <LocalFilePreview onClose={onClose} reloadKey={reloadRequest} target={target} />
  }

  return <BrowserFrame reloadRequest={reloadRequest} target={target} />
}

function BrowserFrame({ target, reloadRequest }: Pick<PreviewPaneProps, 'target' | 'reloadRequest'>) {
  const { t } = useI18n()
  const [html, setHtml] = useState<string>()
  const [error, setError] = useState<string>()
  useEffect(() => {
    const controller = new AbortController()
    setHtml(undefined)
    setError(undefined)

    if (target.kind === 'file') {
      backendFetch(`/api/fs/download?path=${encodeURIComponent(target.path || target.url)}`, { signal: controller.signal })
        .then(response => { if (!response.ok) {throw new Error(sharedMessage('preview.unavailable'));}

 return response.text() })
        .then(setHtml).catch(error => { if (!controller.signal.aborted) {setError(String(error))} })
    }

    return () => controller.abort()
  }, [target, reloadRequest])
  const url = target.kind === 'url' && /^https?:\/\//i.test(target.url) ? target.url : undefined

  return <div className="flex h-full min-h-0 flex-col">
    <div className="flex items-center gap-2 border-b px-3 py-2 text-xs">
      <span className="min-w-0 flex-1 truncate">{target.label}</span>
      {url && <button onClick={() => void window.hermesDesktop.openExternal(url)}>{t.sidebar.row.openInNewTab}</button>}
    </div>
    {error ? <div className="p-4" role="alert">{error}</div> :
      <iframe className="min-h-0 flex-1 border-0 bg-white" key={`${target.url}:${reloadRequest}`} referrerPolicy="no-referrer"
        sandbox="allow-scripts allow-forms allow-downloads" src={url} srcDoc={html} title={target.label} />}
  </div>
}

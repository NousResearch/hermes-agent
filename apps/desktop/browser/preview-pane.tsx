import type { ComponentProps } from 'react'
import { useEffect, useState } from 'react'

import { ArtifactPreview } from '../src/app/chat/right-rail/preview-artifact'
import { LocalFilePreview } from '../src/app/chat/right-rail/preview-file'
import type { PreviewPane as DesktopPreviewPane } from '../src/app/chat/right-rail/preview-pane'
import { usePaneVisible } from '../src/components/pane-shell/pane-visibility'
import { Button } from '../src/components/ui/button'
import { useI18n } from '../src/i18n'

import { sharedMessage } from './messages'
import { backendFetch } from './transport'

type PreviewPaneProps = ComponentProps<typeof DesktopPreviewPane>

/** Browser guest content gets an opaque sandbox origin, never the backend's cookies. */
export function PreviewPane({ target, onClose, reloadRequest = 0 }: PreviewPaneProps) {
  const { t } = useI18n()
  const visible = usePaneVisible()
  const [phone, setPhone] = useState(() => window.matchMedia('(max-width: 640px)').matches)
  useEffect(() => {
    const query = window.matchMedia('(max-width: 640px)')
    const update = () => setPhone(query.matches)
    query.addEventListener('change', update)

    return () => query.removeEventListener('change', update)
  }, [])
  useEffect(() => {
    if (phone && visible && document.activeElement instanceof HTMLElement) {
      document.activeElement.blur()
    }
  }, [phone, target, visible])

  const content = target.kind === 'artifact' ? <ArtifactPreview onClose={onClose} target={target} /> :
    target.kind === 'file' && target.previewKind !== 'html' ?
      <LocalFilePreview onClose={onClose} reloadKey={reloadRequest} target={target} /> :
      <BrowserFrame reloadRequest={reloadRequest} target={target} />

  return <section className="browser-preview-surface" data-overlay-surface={phone && visible ? '' : undefined}>
    {onClose && <div className="browser-preview-mobile-header">
      <span className="min-w-0 flex-1 truncate">{target.label}</span>
      <Button onClick={onClose} variant="ghost">{t.common.close}</Button>
    </div>}
    <div className="min-h-0 flex-1 overflow-auto">{content}</div>
  </section>
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

import { useEffect, useRef, useState } from 'react'

import { MarkdownPreview } from '@/app/chat/right-rail/preview-file'
import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import { PathText } from '@/components/ui/path-text'
import { PixelIcon } from '@/components/ui/pixel-icon'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Skeleton } from '@/components/ui/skeleton'
import { Tip } from '@/components/ui/tooltip'
import { readDesktopFileText } from '@/lib/desktop-fs'
import { ESCAPE_PRIORITY, isTopEscapeLayer } from '@/lib/escape-layers'
import { fmtDateTime } from '@/lib/time'

import type { ArtifactRecord } from '../artifacts/artifact-utils'
import type { IndexedChat } from '../artifacts/item-index-snapshot'

import { chatOfItem, itemOwner } from './library-actions'
import { useLibraryCopy } from './library-copy'

type MarkdownViewMode = 'rendered' | 'source'

interface MarkdownPreviewViewProps {
  item: ArtifactRecord
  chats: Record<string, IndexedChat>
  backLabel: string
  onBack: () => void
  onDownload: (item: ArtifactRecord) => void
  onOpenChat: (item: ArtifactRecord) => void
  onOpenExternal: (item: ArtifactRecord) => void
}

/** In-app Markdown preview in Library (HADESK-96): shares the chat right-rail's `MarkdownPreview`
 *  renderer with a Rendered / Source switch, matching `ImagePreview`'s header & metadata chrome. */
export function MarkdownPreviewView({
  backLabel,
  chats,
  item,
  onBack,
  onDownload,
  onOpenChat,
  onOpenExternal
}: MarkdownPreviewViewProps) {
  const copy = useLibraryCopy()
  const chat = chatOfItem(chats, item)
  const ownerConnectionId = chat?.connectionId
  const ownerProfile = chat?.profile ?? item.profile
  const titleRef = useRef<HTMLHeadingElement>(null)
  const [mode, setMode] = useState<MarkdownViewMode>('rendered')
  const [reloadKey, setReloadKey] = useState(0)
  const [state, setState] = useState<{ error: string | null; loading: boolean; text: string }>({
    error: null,
    loading: true,
    text: ''
  })

  useEffect(() => titleRef.current?.focus({ preventScroll: true }), [])

  useEffect(() => {
    let live = true
    setState({ error: null, loading: true, text: '' })
    const owner = itemOwner(chat) ?? (ownerProfile ? { profile: ownerProfile } : undefined)

    readDesktopFileText(item.value, { owner, sessionId: item.sessionId })
      .then(result => {
        if (live) {
          setState({ error: null, loading: false, text: result.text ?? '' })
        }
      })
      .catch(error => {
        if (live) {
          setState({
            error: error instanceof Error ? error.message : String(error),
            loading: false,
            text: ''
          })
        }
      })

    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on primitive owner fields so background index refreshes do not re-fetch.
  }, [item.sessionId, item.value, ownerConnectionId, ownerProfile, reloadKey])

  return (
    <div
      className="library-preview"
      data-slot="library-markdown-preview"
      onKeyDown={event => {
        if (event.key === 'Escape' && !event.defaultPrevented && isTopEscapeLayer(ESCAPE_PRIORITY.overlay)) {
          event.preventDefault()
          onBack()
        }
      }}
      tabIndex={-1}
    >
      <div className="library-preview-head">
        <Button className="library-back" onClick={onBack} size="sm" variant="secondary">
          <PixelIcon name="chevron-left" />
          {backLabel}
        </Button>
        <div className="library-preview-title">
          <h1 className="library-preview-name" ref={titleRef} tabIndex={-1}>
            {item.label}
          </h1>
          <p>Markdown</p>
        </div>
        <div className="library-preview-steps">
          <SegmentedControl
            onChange={next => setMode(next as MarkdownViewMode)}
            options={[
              { id: 'rendered', label: 'Preview' },
              { id: 'source', label: 'Source' }
            ]}
            value={mode}
          />
        </div>
      </div>

      <div className="library-markdown-stage">
        {state.loading ? (
          <div aria-busy="true" className="library-markdown-skeleton" role="status">
            <Skeleton className="h-5 w-2/5" />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-5/6" />
            <Skeleton className="h-3 w-3/4" />
          </div>
        ) : state.error ? (
          <div className="library-markdown-error" role="alert">
            <p>{state.error}</p>
            <div className="flex items-center gap-2">
              <Button onClick={() => setReloadKey(count => count + 1)} size="sm" variant="secondary">
                Retry
              </Button>
              <Button onClick={() => onOpenExternal(item)} size="sm" variant="ghost">
                Open file
              </Button>
            </div>
          </div>
        ) : mode === 'rendered' ? (
          <div className="library-markdown-body">
            <MarkdownPreview filePath={item.value} text={state.text} />
          </div>
        ) : (
          <pre className="library-markdown-source">
            <code>{state.text}</code>
          </pre>
        )}
      </div>

      <div className="library-preview-meta">
        <dl>
          <div>
            <dt>{copy.meta.chat}</dt>
            <dd>{chat?.title ?? item.sessionTitle}</dd>
          </div>
          <div>
            <dt>{copy.meta.created}</dt>
            <dd>{fmtDateTime.format(new Date(item.timestamp))}</dd>
          </div>
          <div className="library-preview-path">
            <dt>{copy.meta.path}</dt>
            <dd>
              <PathText path={item.value} />
            </dd>
          </div>
        </dl>
        <div className="library-preview-actions">
          <CopyButton
            appearance="icon"
            buttonSize="icon"
            label={copy.copyPath}
            text={item.value}
            title={copy.copyPath}
          />
          <Tip label={copy.download}>
            <Button aria-label={copy.download} onClick={() => onDownload(item)} size="icon" variant="ghost">
              <PixelIcon name="download" />
            </Button>
          </Tip>
          <Button onClick={() => onOpenExternal(item)} size="sm" variant="secondary">
            Open file
          </Button>
          <Button onClick={() => onOpenChat(item)}>{copy.openChat}</Button>
        </div>
      </div>
    </div>
  )
}

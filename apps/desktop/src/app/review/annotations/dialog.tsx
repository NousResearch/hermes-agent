import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { ESCAPE_PRIORITY, isTopEscapeLayer, pushEscapeLayer } from '@/lib/escape-layers'
import {
  $annotationDiscardIntent,
  $annotationDraft,
  $annotationEditorAnchor,
  $annotationEditorCollapsed,
  $annotationStorageError,
  annotationExcerpt,
  cancelDiscardAnnotationDraft,
  collapseAnnotationEditor,
  confirmDiscardAnnotationDraft,
  requestDiscardAnnotationDraft,
  REVIEW_ANNOTATION_LABELS,
  saveAnnotationDraft,
  toggleAnnotationDraftLabel,
  updateAnnotationDraft
} from '@/store/annotations'

import { annotationAnchorLabel } from './feedback'

export function AnnotationDialog() {
  const { t } = useI18n()
  const copy = t.desktop.annotations
  const draft = useStore($annotationDraft)
  const editorAnchor = useStore($annotationEditorAnchor)
  const collapsed = useStore($annotationEditorCollapsed)
  const discardIntent = useStore($annotationDiscardIntent)
  const storageError = useStore($annotationStorageError)
  const editorRef = useRef<HTMLElement | null>(null)

  const viewportWidth = typeof window === 'undefined' ? 336 : window.innerWidth
  const viewportHeight = typeof window === 'undefined' ? 640 : window.innerHeight
  const estimatedHeight = draft?.type === 'suggestion' ? 350 : 290
  const boundary = editorAnchor?.boundary ?? { height: viewportHeight, width: viewportWidth, x: 0, y: 0 }
  const boundaryRight = boundary.x + boundary.width
  const boundaryBottom = boundary.y + boundary.height
  const width = Math.min(320, Math.max(220, boundary.width - 16))
  const minLeft = boundary.x + 8
  const maxLeft = Math.max(minLeft, boundaryRight - width - 8)
  const left = editorAnchor ? Math.max(minLeft, Math.min(editorAnchor.x, maxLeft)) : minLeft
  const below = editorAnchor ? editorAnchor.y + editorAnchor.height + 8 : boundary.y + 8
  const above = editorAnchor ? editorAnchor.y - estimatedHeight - 8 : below
  const preferredTop = below + estimatedHeight <= boundaryBottom - 8 ? below : above
  const minTop = boundary.y + 8
  const maxTop = Math.max(minTop, boundaryBottom - estimatedHeight - 8)
  const top = Math.max(minTop, Math.min(preferredTop, maxTop))
  const maxHeight = Math.max(120, boundary.height - 16)

  const excerpt = draft ? annotationExcerpt(draft.anchor) : ''

  useEffect(() => {
    if (!draft || collapsed || discardIntent) {
      return
    }

    const popEscapeLayer = pushEscapeLayer(ESCAPE_PRIORITY.annotation)

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !event.defaultPrevented && isTopEscapeLayer(ESCAPE_PRIORITY.annotation)) {
        event.preventDefault()
        collapseAnnotationEditor()
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => {
      window.removeEventListener('keydown', onKeyDown)
      popEscapeLayer()
    }
  }, [collapsed, discardIntent, draft])

  return (
    <>
      {draft &&
        !collapsed &&
        createPortal(
          <section
            aria-label={copy.title}
            aria-modal="false"
            className="fixed z-[100] flex flex-col gap-2 overflow-y-auto rounded-md border border-(--ui-stroke-secondary) bg-(--ui-surface-background) p-2.5 shadow-xl [-webkit-app-region:no-drag]"
            data-annotation-editor=""
            ref={editorRef}
            role="dialog"
            style={{ left, maxHeight, top, width }}
          >
            <header className="flex min-w-0 items-start gap-1">
              <div className="min-w-0 flex-1">
                <h2 className="text-xs font-semibold text-(--ui-text-primary)">
                  {draft.editingId ? copy.edit : copy.title}
                </h2>
                <p className="truncate text-xs text-(--ui-text-muted)" title={annotationAnchorLabel(draft.anchor)}>
                  {annotationAnchorLabel(draft.anchor)}
                </p>
              </div>
              <Button
                aria-label={copy.preview.minimizeEditor}
                className="size-6 shrink-0 p-0"
                onClick={collapseAnnotationEditor}
                size="icon"
                title={copy.preview.minimizeEditor}
                type="button"
                variant="ghost"
              >
                <Codicon name="chrome-minimize" size="0.75rem" />
              </Button>
            </header>

            {storageError && (
              <Alert className="grid-cols-1 px-2 py-1 text-[0.68rem]" variant="warning">
                <AlertDescription className="col-start-1">{copy.storageError}</AlertDescription>
              </Alert>
            )}

            {excerpt && (
              <blockquote className="line-clamp-3 max-h-14 overflow-hidden rounded border-l-2 border-(--ui-accent) bg-(--ui-surface-secondary) px-2 py-1 font-mono text-[0.64rem] whitespace-pre-wrap text-(--ui-text-secondary)">
                {excerpt}
              </blockquote>
            )}

            <div aria-label={copy.title} className="flex gap-1" role="group">
              {(['comment', 'suggestion', 'concern'] as const).map(type => (
                <Button
                  className="h-6 px-2 text-[0.65rem]"
                  key={type}
                  onClick={() => updateAnnotationDraft({ type })}
                  size="sm"
                  type="button"
                  variant={draft.type === type ? 'default' : 'outline'}
                >
                  {copy[type]}
                </Button>
              ))}
            </div>

            <Textarea
              aria-label={copy.commentPlaceholder}
              className="min-h-20 resize-y text-xs"
              onChange={event => updateAnnotationDraft({ comment: event.currentTarget.value })}
              placeholder={copy.commentPlaceholder}
              value={draft.comment}
            />

            {draft.type === 'suggestion' && (
              <Textarea
                aria-label={copy.suggestedReplacement}
                className="min-h-16 resize-y font-mono text-xs"
                onChange={event => updateAnnotationDraft({ suggestion: event.currentTarget.value })}
                placeholder={copy.suggestionPlaceholder}
                value={draft.suggestion}
              />
            )}

            <div className="flex flex-wrap gap-1">
              {REVIEW_ANNOTATION_LABELS.map(label => (
                <Button
                  aria-pressed={draft.labels.includes(label)}
                  className="h-5 px-1.5 text-[0.6rem]"
                  key={label}
                  onClick={() => toggleAnnotationDraftLabel(label)}
                  size="sm"
                  type="button"
                  variant={draft.labels.includes(label) ? 'secondary' : 'ghost'}
                >
                  {copy.labels[label]}
                </Button>
              ))}
            </div>

            <footer className="flex justify-end gap-1">
              <Button onClick={requestDiscardAnnotationDraft} size="sm" type="button" variant="ghost">
                {t.common.cancel}
              </Button>
              <Button disabled={!draft.comment.trim()} onClick={saveAnnotationDraft} size="sm" type="button">
                {copy.save}
              </Button>
            </footer>
          </section>,
          document.body
        )}

      <ConfirmDialog
        confirmLabel={copy.discard}
        description={copy.discardDescription}
        destructive
        dismissOnConfirm
        onClose={cancelDiscardAnnotationDraft}
        onConfirm={confirmDiscardAnnotationDraft}
        open={discardIntent !== null}
        title={discardIntent === 'replace' ? copy.discardReplaceTitle : copy.discardTitle}
      />
    </>
  )
}

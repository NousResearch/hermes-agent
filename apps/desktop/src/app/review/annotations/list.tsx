import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import {
  $annotationContext,
  $annotations,
  annotationExcerpt,
  clearAnnotations,
  editAnnotation,
  removeAnnotation
} from '@/store/annotations'

import { annotationAnchorLabel, sendAnnotationsToComposer } from './feedback'
import { navigateToAnnotation } from './navigation'

export function ReviewAnnotationsList() {
  const { t } = useI18n()
  const copy = t.desktop.annotations
  const context = useStore($annotationContext)
  const items = useStore($annotations)
  const [removeId, setRemoveId] = useState<string | null>(null)
  const [clearing, setClearing] = useState(false)
  const [sendError, setSendError] = useState<string | null>(null)
  const [sending, setSending] = useState(false)

  const statusLabel = {
    orphaned: copy.statusOrphaned,
    sent: copy.statusSent,
    stale: copy.statusStale
  } as const

  if (items.length === 0) {
    return null
  }

  return (
    <>
      <section className="flex max-h-[42%] shrink-0 flex-col border-t border-(--ui-stroke-tertiary)">
        <div className="flex items-center gap-2 px-2.5 py-1.5">
          <span className="min-w-0 flex-1 text-[0.66rem] font-semibold uppercase tracking-wide text-(--ui-text-secondary)">
            {copy.listTitle} ({items.length})
          </span>
          <Tip label={copy.sendToAgent}>
            <Button
              aria-label={copy.sendToAgent}
              disabled={sending}
              onClick={() => {
                setSending(true)
                setSendError(null)
                void sendAnnotationsToComposer(context, items)
                  .catch(error => setSendError(error instanceof Error ? error.message : String(error)))
                  .finally(() => setSending(false))
              }}
              size="icon-xs"
            >
              <Codicon className={sending ? 'animate-spin' : undefined} name={sending ? 'loading' : 'send'} />
            </Button>
          </Tip>
          <Tip label={copy.clear}>
            <Button aria-label={copy.clear} onClick={() => setClearing(true)} size="icon-xs" variant="ghost">
              <Codicon name="trash" />
            </Button>
          </Tip>
        </div>

        {sendError && <p className="px-2.5 pb-1 text-[0.62rem] text-destructive">{sendError}</p>}

        <div className="min-h-0 space-y-2 overflow-y-auto px-2.5 pb-2">
          {items.map(item => (
            <div className="group grid gap-1 text-[0.66rem]" key={item.id}>
              <div className="flex items-start gap-1">
                <button
                  className="min-w-0 flex-1 truncate text-left font-mono text-(--ui-text-secondary) hover:text-foreground"
                  onClick={() => navigateToAnnotation(item)}
                  title={annotationAnchorLabel(item.anchor)}
                  type="button"
                >
                  {annotationAnchorLabel(item.anchor)}
                </button>
                <Tip label={copy.edit}>
                  <Button aria-label={copy.edit} onClick={() => editAnnotation(item.id)} size="icon-xs" variant="ghost">
                    <Codicon name="edit" />
                  </Button>
                </Tip>
                <Tip label={copy.remove}>
                  <Button aria-label={copy.remove} onClick={() => setRemoveId(item.id)} size="icon-xs" variant="ghost">
                    <Codicon name="trash" />
                  </Button>
                </Tip>
              </div>

              <div className="flex flex-wrap gap-1 text-(--ui-text-quaternary)">
                <span>{copy[item.type]}</span>
                {item.labels.map(label => (
                  <span key={label}>· {copy.labels[label]}</span>
                ))}
                {item.status !== 'active' && <span>· {statusLabel[item.status]}</span>}
              </div>

              <p className="whitespace-pre-wrap leading-relaxed text-(--ui-text-tertiary)">{item.comment}</p>
              {annotationExcerpt(item.anchor) && (
                <blockquote className="line-clamp-3 border-l-2 border-(--ui-stroke-primary) pl-2 font-mono text-[0.62rem] whitespace-pre-wrap text-(--ui-text-quaternary)">
                  {annotationExcerpt(item.anchor)}
                </blockquote>
              )}
              {item.suggestion && (
                <pre className="max-h-28 overflow-auto bg-(--ui-bg-quaternary) p-1.5 whitespace-pre-wrap">
                  {item.suggestion}
                </pre>
              )}
            </div>
          ))}
        </div>
      </section>

      <ConfirmDialog
        confirmLabel={copy.remove}
        destructive
        dismissOnConfirm
        onClose={() => setRemoveId(null)}
        onConfirm={() => {
          if (removeId) {
            removeAnnotation(removeId)
          }
        }}
        open={removeId !== null}
        title={copy.remove}
      />
      <ConfirmDialog
        confirmLabel={copy.clear}
        destructive
        dismissOnConfirm
        onClose={() => setClearing(false)}
        onConfirm={clearAnnotations}
        open={clearing}
        title={copy.clear}
      />
    </>
  )
}

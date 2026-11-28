import type { RefObject } from 'react'

import { useI18n } from '@/i18n'
import type { AnnotateSession } from '@/lib/preview-annotate'

import { placeAnnotateCard, PreviewAnnotateCard } from './preview-annotate-card'
import { PreviewTextCommentCard } from './preview-text-comment-card'
import type { usePreviewTextComments } from './use-preview-text-comments'

interface PreviewCommentCardsProps {
  annotate: AnnotateSession
  cancelAnnotateDraft: () => Promise<void>
  draftNote: string
  previewContentRef: RefObject<HTMLDivElement | null>
  saveAnnotateDraft: () => Promise<void>
  setDraftNote: (note: string) => void
  textComments: ReturnType<typeof usePreviewTextComments>
}

export function PreviewCommentCards({
  annotate,
  cancelAnnotateDraft,
  draftNote,
  previewContentRef,
  saveAnnotateDraft,
  setDraftNote,
  textComments
}: PreviewCommentCardsProps) {
  const { t } = useI18n()
  const copy = t.preview.web
  const paneHeight = previewContentRef.current?.clientHeight || 360
  const paneWidth = previewContentRef.current?.clientWidth || 360

  return (
    <>
      {annotate.draft && (
        <PreviewAnnotateCard
          {...placeAnnotateCard({ paneHeight, paneWidth, rect: annotate.draft.rect })}
          note={draftNote}
          number={annotate.stack.nextNumber}
          onCancel={() => void cancelAnnotateDraft()}
          onChange={setDraftNote}
          onSave={() => void saveAnnotateDraft()}
          placeholder={copy.commentPlaceholder}
          saveLabel={copy.saveComment}
          title={copy.commentTitle(annotate.stack.nextNumber)}
        />
      )}
      {textComments.draft && textComments.enabled && (
        <PreviewTextCommentCard
          onCancel={textComments.cancel}
          onSubmit={textComments.submit}
          owner={textComments.draft.owner}
          paneHeight={paneHeight}
          paneWidth={paneWidth}
          rect={textComments.draft.selection.rect}
          selectedText={textComments.draft.selection.text}
        />
      )}
    </>
  )
}

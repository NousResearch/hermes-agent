import { atom } from 'nanostores'

import type { ReviewAnnotation } from '@/store/annotations'

export const ANNOTATION_NAVIGATE_EVENT = 'hermes:annotation:navigate'
export const $selectedAnnotationId = atom<string | null>(null)

export function navigateToAnnotation(annotation: ReviewAnnotation): void {
  $selectedAnnotationId.set(annotation.id)
  window.dispatchEvent(
    new CustomEvent(ANNOTATION_NAVIGATE_EVENT, {
      detail: { anchor: annotation.anchor, contextId: annotation.contextId, id: annotation.id }
    })
  )
}

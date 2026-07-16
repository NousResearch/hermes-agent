import { requestComposerFocus, requestComposerInsert, requestComposerInsertRefs } from '@/app/chat/composer/focus'
import { droppedFileInlineRef } from '@/app/chat/composer/inline-refs'
import {
  annotationExcerpt,
  type ReviewAnnotation,
  type ReviewAnnotationAnchor,
  type ReviewContext
} from '@/store/annotations'
import { $currentCwd } from '@/store/session'

import { attachVisualAnnotationComposites } from './visual-export'

export function annotationAnchorLabel(anchor: ReviewAnnotationAnchor): string {
  if (anchor.kind === 'file') {
    return anchor.path
  }

  if (anchor.kind === 'source' || anchor.kind === 'diff') {
    const range = anchor.lineEnd > anchor.lineStart ? `${anchor.lineStart}-${anchor.lineEnd}` : `${anchor.lineStart}`
    const side = anchor.kind === 'diff' ? ` (${anchor.side} side)` : ''

    return `${anchor.path}:${range}${side}`
  }

  if (anchor.kind === 'pdf') {
    return `${anchor.path} (${anchor.documentKind === 'tex' ? 'rendered PDF, ' : ''}page ${anchor.page})`
  }

  if (anchor.kind === 'visual') {
    const markCount = anchor.marks.length

    return `${anchor.path} (${markCount ? `${markCount} visual mark${markCount === 1 ? '' : 's'}` : 'selected image text'})`
  }

  return `${anchor.path} (${anchor.kind})`
}

function contextHeading(context: ReviewContext): string[] {
  const lines = [`Review context: ${context.kind}`, `Workspace: ${context.cwd || '(none)'}`]

  if (context.artifactPath) {
    lines.push(`Artifact: ${context.artifactPath}`)
  }

  if (context.contentHash) {
    lines.push(`Artifact revision: ${context.contentHash}`)
  }

  if (context.reviewScope) {
    lines.push(`Diff scope: ${context.reviewScope}`)
  }

  if (context.baseRef) {
    lines.push(`Base: ${context.baseRef}`)
  }

  if (context.headSha) {
    lines.push(`Head: ${context.headSha}`)
  }

  return lines
}

export function buildAnnotationFeedback(context: ReviewContext, items: readonly ReviewAnnotation[]): string {
  const lines = ['Please review and address these annotations.', '', ...contextHeading(context)]

  items.forEach((item, index) => {
    lines.push('', `${index + 1}. ${annotationAnchorLabel(item.anchor)}`)
    lines.push(`   Type: ${item.type}`)

    if (item.status === 'stale' || item.status === 'orphaned') {
      lines.push(`   Anchor status: ${item.status}; verify the location before editing.`)
    }

    if (item.labels.length > 0) {
      lines.push(`   Labels: ${item.labels.join(', ')}`)
    }

    const excerpt = annotationExcerpt(item.anchor)

    if (excerpt) {
      lines.push(`   Selected text: ${JSON.stringify(excerpt)}`)
    }

    lines.push(`   Comment: ${item.comment}`)

    if (item.suggestion) {
      lines.push('   Suggested replacement:', ...item.suggestion.split('\n').map(line => `     ${line}`))
    }
  })

  return lines.join('\n')
}

export function annotationInlineRefs(items: readonly ReviewAnnotation[]): string[] {
  const cwd = $currentCwd.get()

  const refs = items
    .filter(item => item.status !== 'orphaned' && item.status !== 'stale')
    .map(item => {
      const anchor = item.anchor

      // Old-side line numbers belong to a historical file revision and must
      // never be projected onto the current working-tree file.
      if (anchor.kind === 'diff' && anchor.side === 'old') {
        return null
      }

      if (anchor.kind !== 'source' && anchor.kind !== 'diff') {
        return null
      }

      return droppedFileInlineRef({ line: anchor.lineStart, lineEnd: anchor.lineEnd, path: anchor.path }, cwd)
    })
    .filter((ref): ref is string => Boolean(ref))

  return [...new Set(refs)]
}

export async function sendAnnotationsToComposer(
  context: ReviewContext,
  items: readonly ReviewAnnotation[]
): Promise<void> {
  if (items.length === 0) {
    return
  }

  await attachVisualAnnotationComposites(items)

  const refs = annotationInlineRefs(items)

  if (refs.length > 0) {
    requestComposerInsertRefs(refs, { target: 'main' })
  }

  requestComposerInsert(buildAnnotationFeedback(context, items), { mode: 'block', target: 'main' })
  requestComposerFocus('main')
}

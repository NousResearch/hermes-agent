import type { FC, ReactNode } from 'react'
import { useMemo } from 'react'

import { type Contribution, useContributions } from '@/contrib'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import {
  parseTranscriptDirective,
  TRANSCRIPT_DIRECTIVE_AREA,
  type TranscriptDirectiveContribution
} from '@/lib/transcript-directives'

/**
 * The transcript's directive slot. Given a paragraph's raw text, renders the
 * registered plugin component when the whole paragraph is a claimed
 * `::name{...}` directive; returns null otherwise so the caller keeps its
 * plain `<p>` — an unclaimed directive is just prose.
 *
 * Resolution is registry-backed (`transcript.directives`), so hot-loading a
 * plugin upgrades already-rendered paragraphs in place, exactly like every
 * other contribution area.
 */

/** Extract the paragraph's text when it is text-only — directives never carry
 *  inline markup, so any non-string child disqualifies the paragraph. */
export function paragraphPlainText(children: ReactNode): string | null {
  if (typeof children === 'string') {
    return children
  }

  if (Array.isArray(children) && children.length > 0 && children.every(child => typeof child === 'string')) {
    return children.join('')
  }

  return null
}

/** The contribution claiming `name`, if any. First registration wins. */
function claimFor(contributions: readonly Contribution[], name: string) {
  return contributions.find(c => (c.data as TranscriptDirectiveContribution | undefined)?.name === name)
}

export const TranscriptDirectiveLeaf: FC<{ text: string; streaming?: boolean }> = ({ text, streaming }) => {
  const contributions = useContributions(TRANSCRIPT_DIRECTIVE_AREA)
  const parsed = useMemo(() => parseTranscriptDirective(text), [text])
  const match = parsed ? claimFor(contributions, parsed.name) : undefined
  const contribution = match?.data as TranscriptDirectiveContribution | undefined
  const render = contribution?.render

  // Stable component identity for ContribRender (which mounts this AS a
  // component): a fresh closure per render would remount the widget on
  // every parent render.
  const renderLeaf = useMemo(
    () =>
      render && parsed
        ? () => render({ attrs: parsed.attrs, source: parsed.source, streaming: streaming ?? false })
        : null,
    [render, parsed, streaming]
  )

  if (!match || !renderLeaf) {
    return null
  }

  return (
    <ContribBoundary id={match.id} variant="chip">
      <ContribRender render={renderLeaf} />
    </ContribBoundary>
  )
}

/** True when the paragraph text will resolve to a registered directive —
 *  callers that must decide `<p>` vs slot before rendering use this with the
 *  same registry snapshot the leaf reads. */
export function useIsClaimedDirective(text: string | null): boolean {
  const contributions = useContributions(TRANSCRIPT_DIRECTIVE_AREA)
  const parsed = text === null ? null : parseTranscriptDirective(text)

  return parsed !== null && claimFor(contributions, parsed.name) !== undefined
}

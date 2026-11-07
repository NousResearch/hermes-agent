import { useStore } from '@nanostores/react'
import type { FC, ReactNode } from 'react'
import { Children, isValidElement, useMemo } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { type Contribution, useContributions } from '@/contrib'
import { ContribBoundary } from '@/contrib/react/boundary'
import {
  type ParsedTranscriptDirective,
  segmentTranscriptDirectives,
  TRANSCRIPT_DIRECTIVE_AREA,
  type TranscriptDirectiveContribution,
  type TranscriptParagraphSegment
} from '@/lib/transcript-directives'

/**
 * The transcript's directive slot. Given text, renders the plugin component
 * for every claimed `::name{...}` in it, in order — several, because models
 * merge lines under formatting pressure. Nothing renders for a name no plugin
 * claimed; that text stays the prose it always was.
 *
 * Resolution is registry-backed (`transcript.directives`), so hot-loading a
 * plugin upgrades already-rendered paragraphs in place, exactly like every
 * other contribution area.
 */

/** The contribution claiming `name`, if any. First registration wins. */
function claimFor(contributions: readonly Contribution[], name: string) {
  return contributions.find(c => (c.data as TranscriptDirectiveContribution | undefined)?.name === name)
}

const DirectiveEntry: FC<{
  contribution: Contribution
  parsed: ParsedTranscriptDirective
  streaming: boolean
}> = ({ contribution, parsed, streaming }) => {
  const render = (contribution.data as TranscriptDirectiveContribution).render
  // The surface this transcript renders in (primary pane or a tile) names the
  // chat that owns the card, so its actions can address that chat's composer.
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const runtimeId = useStore(view.$runtimeId)
  const sessionId = storedId || runtimeId || null

  // Stable component IDENTITY per (render, parsed) — a fresh type per parent
  // render would remount the widget (card-sized jump). Streaming and the owning
  // session arrive as real props on that stable type, so the settle flip
  // re-renders the same mount instead of being memo-skipped (ref-only reads
  // were exactly that).
  const Leaf = useMemo(
    () =>
      function DirectiveLeafHost({
        sessionId: owner,
        streaming: live
      }: {
        sessionId: null | string
        streaming: boolean
      }) {
        return <>{render({ attrs: parsed.attrs, sessionId: owner, source: parsed.source, streaming: live })}</>
      },
    [render, parsed]
  )

  return (
    <ContribBoundary id={contribution.id} variant="chip">
      <Leaf sessionId={sessionId} streaming={streaming} />
    </ContribBoundary>
  )
}

export const TranscriptDirectiveLeaf: FC<{ text: string; streaming?: boolean }> = ({ text, streaming }) => {
  const contributions = useContributions(TRANSCRIPT_DIRECTIVE_AREA)

  const segments = useMemo<TranscriptParagraphSegment[] | null>(() => segmentTranscriptDirectives(text), [text])

  const entries = useMemo(
    () =>
      (segments ?? []).flatMap(segment => {
        if (segment.kind !== 'directive') {
          return []
        }

        const match = claimFor(contributions, segment.directive.name)

        return match ? [{ key: `${match.id}:${segment.directive.source}`, match, parsed: segment.directive }] : []
      }),
    [contributions, segments]
  )

  if (entries.length === 0) {
    return null
  }

  return (
    <>
      {entries.map(entry => (
        <DirectiveEntry
          contribution={entry.match}
          key={entry.key}
          parsed={entry.parsed}
          streaming={streaming ?? false}
        />
      ))}
    </>
  )
}

/**
 * What `<hermes-directive>` renders as on its own: the directive exactly as
 * written. `remarkTranscriptDirectives` lifts every directive into that
 * element; a paragraph draws the claimed ones as cards
 * (`useResolvedParagraph`), so this is what shows everywhere else — an
 * unclaimed name, a heading, a table cell, a preview-only surface.
 */
export function TranscriptDirectiveText({ source }: Record<string, unknown>) {
  return <>{typeof source === 'string' ? source : ''}</>
}

/** The source of a lifted directive element, or null for any other child. */
function directiveSource(child: ReactNode): string | null {
  if (!isValidElement<{ source?: unknown }>(child) || child.type !== TranscriptDirectiveText) {
    return null
  }

  return typeof child.props.source === 'string' ? child.props.source : null
}

function isBlank(child: ReactNode): boolean {
  return (typeof child === 'string' && child.trim() === '') || (isValidElement(child) && child.type === 'br')
}

/** Trim a prose run at the seams a lifted card leaves: whitespace and line
 *  breaks that only separated it from the card. */
function trimRun(run: ReactNode[]): ReactNode[] {
  let start = 0
  let end = run.length

  while (start < end && isBlank(run[start])) {
    start += 1
  }

  while (end > start && isBlank(run[end - 1])) {
    end -= 1
  }

  const out = run.slice(start, end)

  if (typeof out[0] === 'string') {
    out[0] = out[0].trimStart()
  }

  if (typeof out.at(-1) === 'string') {
    out[out.length - 1] = (out.at(-1) as string).trimEnd()
  }

  return out
}

/** A paragraph resolved against the registry: the prose to keep as prose, and
 *  the claimed directives to render as cards, in the order they were written. */
export type ResolvedParagraphSegment = { kind: 'prose'; children: ReactNode[] } | { kind: 'directive'; source: string }

/**
 * How a paragraph should render. Null means "as the plain `<p>` it always
 * was" — no directive in it, or none that anyone registered.
 *
 * Works on the paragraph's rendered children, where every directive is
 * already its own element, so prose markup around it (line breaks, bold,
 * links, mentions) stays exactly as rendered. A directive nobody claimed
 * stays in the prose run as its text, which is what keeps this from taking
 * text away from the reader: the only thing that can be lifted out of a
 * sentence is markup a plugin is standing by to draw.
 */
export function useResolvedParagraph(children: ReactNode, enabled = true): ResolvedParagraphSegment[] | null {
  const contributions = useContributions(TRANSCRIPT_DIRECTIVE_AREA)

  return useMemo(() => {
    if (!enabled) {
      return null
    }

    const out: ResolvedParagraphSegment[] = []
    let claimed = false

    for (const child of Children.toArray(children)) {
      const source = directiveSource(child)
      const parsed = source === null ? undefined : segmentTranscriptDirectives(source)?.[0]

      if (source !== null && parsed?.kind === 'directive' && claimFor(contributions, parsed.directive.name)) {
        claimed = true
        out.push({ kind: 'directive', source })

        continue
      }

      const previous = out.at(-1)

      if (previous?.kind === 'prose') {
        previous.children.push(child)
      } else {
        out.push({ kind: 'prose', children: [child] })
      }
    }

    if (!claimed) {
      return null
    }

    return out.flatMap((segment): ResolvedParagraphSegment[] => {
      if (segment.kind === 'directive') {
        return [segment]
      }

      const trimmed = trimRun(segment.children)

      return trimmed.length > 0 ? [{ kind: 'prose', children: trimmed }] : []
    })
  }, [children, contributions, enabled])
}

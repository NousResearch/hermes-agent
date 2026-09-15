import type { FC, ReactNode } from 'react'
import { useEffect, useMemo } from 'react'

import { type Contribution, useContributions } from '@/contrib'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import {
  describeDirectiveDrop,
  warnDirectiveRenderFailed,
  warnUnclaimedDirective,
  warnUnparsedDirective
} from '@/lib/directive-diagnostics'
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import {
  type ParsedTranscriptDirective,
  parseTranscriptDirective,
  segmentTranscriptDirectives,
  TRANSCRIPT_DIRECTIVE_AREA,
  type TranscriptDirectiveContribution,
  type TranscriptParagraphSegment
} from '@/lib/transcript-directives'

// B4 reworks the parser; until then the prototype's parser is an onboarding feature.
const onboardingEnabled = isOnboardingEnabled()

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

const DirectiveEntry: FC<{
  contribution: Contribution
  parsed: ParsedTranscriptDirective
  streaming: boolean
}> = ({ contribution, parsed, streaming }) => {
  const render = (contribution.data as TranscriptDirectiveContribution).render

  // Stable component IDENTITY per (render, parsed) — a fresh type per parent
  // render would remount the widget (card-sized jump). Streaming arrives as a
  // real prop on that stable type, so the settle flip re-renders the same
  // mount instead of being memo-skipped (ref-only reads were exactly that).
  const Leaf = useMemo(
    () =>
      function DirectiveLeafHost({ streaming: live }: { streaming: boolean }) {
        return <>{render({ attrs: parsed.attrs, source: parsed.source, streaming: live })}</>
      },
    [render, parsed]
  )

  // The boundary only knows the contribution id; a dropped panel is far easier
  // to chase when the log also names the directive the model addressed.
  const onRenderError = useMemo(
    () => (error: Error) => warnDirectiveRenderFailed(parsed.name, contribution.id, error),
    [contribution.id, parsed.name]
  )

  return (
    <ContribBoundary id={contribution.id} onError={onRenderError} variant="chip">
      <Leaf streaming={streaming} />
    </ContribBoundary>
  )
}

export const TranscriptDirectiveLeaf: FC<{ text: string; streaming?: boolean }> = ({ text, streaming }) => {
  const contributions = useContributions(TRANSCRIPT_DIRECTIVE_AREA)

  const segments = useMemo<TranscriptParagraphSegment[] | null>(() => {
    if (onboardingEnabled) {
      return segmentTranscriptDirectives(text)
    }

    const parsed = parseTranscriptDirective(text)

    return parsed ? [{ kind: 'directive', directive: parsed }] : null
  }, [text])

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

  const match = entries[0]?.match
  const parsed = entries[0]?.parsed
  // SAFETY: claimFor resolved this entry from the directive area by its registered name.
  const render = (match?.data as TranscriptDirectiveContribution | undefined)?.render

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

  // Name the directive, not just the contribution id, when a widget throws:
  // the boundary's chip is easy to miss in a long transcript.
  const onRenderError = useMemo(
    () => (match && parsed ? (error: Error) => warnDirectiveRenderFailed(parsed.name, match.id, error) : undefined),
    [match, parsed]
  )

  if (!onboardingEnabled) {
    if (!match || !renderLeaf) {
      return null
    }

    return (
      <ContribBoundary id={match.id} onError={onRenderError} variant="chip">
        <ContribRender render={renderLeaf} />
      </ContribBoundary>
    )
  }

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

/** A paragraph resolved against the registry: the prose to keep as prose, and
 *  the claimed directives to render as cards, in the order they were written. */
export type ResolvedParagraphSegment = { kind: 'prose'; text: string } | { kind: 'directive'; source: string }

/**
 * How a paragraph should render. Null means "as the plain `<p>` it always
 * was" — no directive in it, or none that anyone registered.
 *
 * A directive nobody claimed is folded back into the prose around it, which is
 * what keeps this from taking text away from the reader: the only thing that
 * can be lifted out of a sentence is markup a plugin is standing by to draw.
 */
export function useResolvedParagraph(text: string | null): ResolvedParagraphSegment[] | null {
  const contributions = useContributions(TRANSCRIPT_DIRECTIVE_AREA)

  return useMemo(() => {
    if (!onboardingEnabled) {
      const parsed = text === null ? null : parseTranscriptDirective(text)

      return parsed && claimFor(contributions, parsed.name) ? [{ kind: 'directive', source: parsed.source }] : null
    }

    const segments = text === null ? null : segmentTranscriptDirectives(text)

    if (!segments) {
      return null
    }

    const out: ResolvedParagraphSegment[] = []
    let claimed = false

    for (const segment of segments) {
      const isCard = segment.kind === 'directive' && claimFor(contributions, segment.directive.name) !== undefined

      if (isCard) {
        claimed = true
        out.push({ kind: 'directive', source: segment.directive.source })

        continue
      }

      // Prose, or an unclaimed directive that is only ever text. Merge into the
      // run before it so a fold never splits one sentence across two <p>s.
      const raw = segment.kind === 'prose' ? segment.text : segment.directive.source
      const previous = out.at(-1)

      if (previous?.kind === 'prose') {
        previous.text += raw
      } else {
        out.push({ kind: 'prose', text: raw })
      }
    }

    if (!claimed) {
      return null
    }

    return out.filter(segment => segment.kind === 'directive' || segment.text.trim() !== '')
  }, [contributions, text])
}

/** True when the paragraph text will resolve to at least one registered
 *  directive — callers that must decide `<p>` vs slot before rendering use
 *  this against the same registry snapshot the leaf reads. */
export function useIsClaimedDirective(text: string | null): boolean {
  return useResolvedParagraph(text) !== null
}

/**
 * Report a directive-looking paragraph that will NOT become a widget, and say
 * why in one user-facing line.
 *
 * This has to live with the caller that keeps the plain `<p>`, not inside the
 * leaf: when nothing claims the name the leaf never mounts, so a drop there is
 * unobservable. Streaming is never a drop — every prefix of an arriving
 * directive is malformed, and a badge would flicker through the emission.
 */
export function useDirectiveDropWarning(text: string | null, claimed: boolean, streaming: boolean): string | null {
  const contributions = useContributions(TRANSCRIPT_DIRECTIVE_AREA)

  const registeredNames = useMemo(
    () => contributions.map(c => (c.data as TranscriptDirectiveContribution | undefined)?.name ?? '?'),
    [contributions]
  )

  useEffect(() => {
    if (text === null || claimed) {
      return
    }

    const parsed = parseTranscriptDirective(text)

    if (!parsed) {
      warnUnparsedDirective(text, streaming)

      return
    }

    warnUnclaimedDirective(parsed.name, registeredNames, streaming)
  }, [text, claimed, registeredNames, streaming])

  return useMemo(
    () => (text === null || claimed ? null : describeDirectiveDrop(text, registeredNames, streaming)),
    [text, claimed, registeredNames, streaming]
  )
}

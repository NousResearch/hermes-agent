import { createElement, type FC, type ReactNode } from 'react'

import { useContributions } from '@/contrib'
import { ContribBoundary } from '@/contrib/react/boundary'
import {
  TRANSCRIPT_MESSAGE_AREA,
  type TranscriptMessageContribution,
  type TranscriptMessageProps
} from '@/lib/transcript-message'

interface TranscriptMessageSlotProps extends TranscriptMessageProps {
  fallback?: ReactNode
}

function contributionIdentity(contribution: TranscriptMessageContribution, props: TranscriptMessageProps) {
  return [contribution.source ?? 'core', contribution.id, props.sessionId ?? 'draft', props.messageId, props.kind].join(
    ':'
  )
}

/** Shared selection and failure boundary for transcript contribution placements. */
export const TranscriptMessageSlot: FC<TranscriptMessageSlotProps> = ({ fallback, ...props }) => {
  const contributions = useContributions(TRANSCRIPT_MESSAGE_AREA) as readonly TranscriptMessageContribution[]
  const matches: TranscriptMessageContribution[] = []

  for (const contribution of contributions) {
    const identity = contributionIdentity(contribution, props)

    try {
      if (!contribution.data.match(props)) {
        continue
      }
    } catch (error) {
      console.error(`[transcript-message:${identity}] match failed`, error)

      continue
    }

    matches.push(contribution)

    if (props.kind === 'slash-result') {
      break
    }
  }

  if (matches.length === 0) {
    return fallback ?? null
  }

  return (
    <>
      {matches.map(contribution => {
        const identity = contributionIdentity(contribution, props)

        return (
          <ContribBoundary fallback={props.kind === 'slash-result' ? fallback : null} id={identity} key={identity}>
            {createElement(contribution.render, props)}
          </ContribBoundary>
        )
      })}
    </>
  )
}

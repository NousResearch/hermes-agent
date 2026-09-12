import type { FC, ReactNode } from 'react'
import { useMemo } from 'react'

import { ErrorBoundary } from '@/components/error-boundary'
import { useContributions } from '@/contrib'
import { ContribRender } from '@/contrib/react/boundary'
import {
  CHAT_WELCOME_AREA,
  type ChatWelcomeContribution,
  type ChatWelcomeProps
} from '@/lib/chat-welcome'

interface ChatWelcomeSlotProps extends ChatWelcomeProps {
  fallback: ReactNode
}

/** Renders the first valid draft welcome, falling back to core's Intro. */
export const ChatWelcomeSlot: FC<ChatWelcomeSlotProps> = ({ cwd, fallback, profile }) => {
  const contributions = useContributions(CHAT_WELCOME_AREA)
  const contribution = contributions.find(
    candidate => typeof (candidate.data as ChatWelcomeContribution | undefined)?.render === 'function'
  )
  const render = (contribution?.data as ChatWelcomeContribution | undefined)?.render
  const renderWelcome = useMemo(() => (render ? () => render({ cwd, profile }) : null), [cwd, profile, render])

  if (!contribution || !renderWelcome) {
    return fallback
  }

  return (
    <ErrorBoundary fallback={() => fallback} key={contribution.id} label={`contrib:${contribution.id}`}>
      <ContribRender render={renderWelcome} />
    </ErrorBoundary>
  )
}

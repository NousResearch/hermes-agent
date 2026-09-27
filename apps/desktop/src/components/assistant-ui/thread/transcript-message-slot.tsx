import { MessagePrimitive, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type ReactNode, useMemo } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { ErrorBoundary } from '@/components/error-boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import { type Contribution } from '@/contrib/types'
import {
  TRANSCRIPT_MESSAGE_AREA,
  type TranscriptMessageContribution,
  type TranscriptMessageProps
} from '@/lib/transcript-message'

// A replacement registration gets a fresh boundary even if it keeps its id.
// In particular, a crashed renderer must not hide its hot-reloaded successor.
const identities = new WeakMap<Contribution, number>()
let nextIdentity = 0

function identity(contribution: Contribution): number {
  let value = identities.get(contribution)

  if (!value) {
    value = ++nextIdentity
    identities.set(contribution, value)
  }

  return value
}

interface SlotProps {
  kind: TranscriptMessageProps['kind']
  command?: string
  output?: string
  fallback?: ReactNode
}

export function TranscriptMessageSlot({ kind, command, output, fallback = null }: SlotProps) {
  const messageId = useAuiState(s => s.message.id)
  const isLast = useAuiState(s => s.thread.messages.at(-1)?.id === s.message.id)
  const sessionId = useStore(useSessionView().$runtimeId)
  const contributions = useContributions(TRANSCRIPT_MESSAGE_AREA)

  if (!sessionId) {
    return <>{fallback}</>
  }

  const props: TranscriptMessageProps =
    kind === 'slash-result'
      ? { kind, sessionId, messageId, isLast, command: command!, output: output! }
      : { kind, sessionId, messageId, isLast }

  for (const contribution of contributions) {
    const data = contribution.data as TranscriptMessageContribution | undefined

    if (typeof data?.match !== 'function' || typeof data.render !== 'function') {
      continue
    }

    try {
      if (!data.match(props)) {
        continue
      }
    } catch (error) {
      console.error(`[transcript-message:${contribution.id}] match failed`, error)

      continue
    }

    return (
      <TranscriptMessageEntry
        contribution={contribution}
        fallback={fallback}
        key={`${sessionId}:${messageId}:${kind}:${identity(contribution)}:${command ?? ''}:${output ?? ''}`}
        props={props}
      />
    )
  }

  return <>{fallback}</>
}

function TranscriptMessageEntry({
  contribution,
  fallback,
  props
}: {
  contribution: Contribution
  fallback: ReactNode
  props: TranscriptMessageProps
}) {
  const render = (contribution.data as TranscriptMessageContribution).render
  // Render as a component, not a function call: plugin hooks then belong to
  // this mount, and changes to isLast update props without remounting it.
  const Leaf = useMemo(() => (live: TranscriptMessageProps) => <>{render(live)}</>, [render])
  const content = <Leaf {...props} />

  return (
    <ErrorBoundary fallback={() => fallback} label={`contrib:${contribution.id}`}>
      {props.kind === 'slash-result' ? (
        <MessagePrimitive.Root
          className="w-full min-w-0 self-start"
          data-role="system"
          data-slot="aui_system-message-root"
        >
          {content}
        </MessagePrimitive.Root>
      ) : (
        content
      )}
    </ErrorBoundary>
  )
}

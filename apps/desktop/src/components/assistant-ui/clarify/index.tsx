'use client'

import { type ToolCallMessagePartProps, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { sessionClarifyRequest } from '@/store/clarify'

import { selectMessageRunning } from '../tool/fallback-model'

import { readClarifyArgs } from './parse'
import { ClarifyToolPending } from './pending'
import { ClarifyToolSettled } from './settled'
import { useUndeliveredClarify } from './use-undelivered'

type ClarifyToolProps = ToolCallMessagePartProps & { interrupted?: boolean }

export const ClarifyTool = (props: ClarifyToolProps) => {
  // Answered → settled Q&A (ToolFallback collapsed the answer away).
  if (props.result !== undefined) {
    return <ClarifyToolSettled {...props} />
  }

  return <ClarifyToolLive {...props} />
}

function ClarifyToolLive(props: ClarifyToolProps) {
  // The tool row is in whichever session's transcript rendered it — read THAT
  // session's clarify (primary or tile), not the globally-active one.
  const sessionId = useStore(useSessionView().$runtimeId)
  const $request = useMemo(() => sessionClarifyRequest(sessionId), [sessionId])
  const request = useStore($request)
  const fromArgs = useMemo(() => readClarifyArgs(props.args), [props.args])
  const messageRunning = useAuiState(selectMessageRunning)
  // Answering clears the request a beat before tool.complete supplies the result.
  const [answered, setAnswered] = useState(false)
  const undelivered = useUndeliveredClarify(sessionId, messageRunning && !request && !answered)

  // The containing message may settle before the request hydrates. Args are
  // display-only until real request IDs arrive; explicit Stop still demotes.
  const hasQuestionPreview = Boolean(fromArgs.questions?.some(entry => entry.question.trim()))
  const endedWithoutRequest = props.interrupted || props.status.type === 'incomplete'

  if (!messageRunning && !request && !answered && (endedWithoutRequest || !hasQuestionPreview)) {
    return <ToolFallback {...props} />
  }

  return (
    <ClarifyToolPending
      fromArgs={fromArgs}
      onAnswered={() => setAnswered(true)}
      request={request}
      undelivered={undelivered}
    />
  )
}

'use client'

import { type ToolCallMessagePartProps, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { $settledClarifyResults, normalizeSetupChoose, sessionClarifyRequest } from '@/store/clarify'

import { selectMessageRunning } from '../tool/fallback-model'
import { parseMaybeObject } from '../tool/fallback-model/format'

import { readClarifyArgs } from './parse'
import { ClarifyToolPending } from './pending'
import { ClarifyToolSettled } from './settled'
import { SetupChoosePending } from './setup-pending'
import { SetupChooseSettled } from './setup-settled'
import { useUndeliveredClarify } from './use-undelivered'

export const ClarifyTool = (props: ToolCallMessagePartProps) => {
  // Answered → settled Q&A (ToolFallback collapsed the answer away).
  if (props.result !== undefined) {
    return props.toolName === 'setup_choose' ? <SetupChooseSettled {...props} /> : <ClarifyToolSettled {...props} />
  }

  return <ClarifyToolLive {...props} />
}

function ClarifyToolLive(props: ToolCallMessagePartProps) {
  // The tool row is in whichever session's transcript rendered it — read THAT
  // session's clarify (primary or tile), not the globally-active one.
  const sessionId = useStore(useSessionView().$runtimeId)
  const $request = useMemo(() => sessionClarifyRequest(sessionId), [sessionId])
  const setupCard = props.toolName === 'setup_choose'
  const sessionRequest = useStore($request)
  const request = sessionRequest && Boolean(sessionRequest.setup) === setupCard ? sessionRequest : null
  const fromArgs = useMemo(() => readClarifyArgs(props.args), [props.args])
  const setupArgs = useMemo(() => normalizeSetupChoose(parseMaybeObject(props.args)), [props.args])
  const messageRunning = useAuiState(selectMessageRunning)
  // Answering clears the request a beat before `tool.complete` swaps in the
  // settled card. Latch submit so that gap doesn't demote; Stop also clears
  // the request and must still collapse an unanswered card.
  const [answered, setAnswered] = useState(false)
  // The request this row asked, remembered past its clearing: a skip or a
  // typed answer settles it in the store before (or without) `tool.complete`.
  const [requestId, setRequestId] = useState<null | string>(null)
  const settledResults = useStore($settledClarifyResults)
  const settledResult = requestId && request?.requestId !== requestId ? settledResults[requestId] : undefined

  if (request && !settledResult && request.requestId !== requestId) {
    setRequestId(request.requestId)
  }

  const undelivered = useUndeliveredClarify(sessionId, messageRunning && !request && !answered && !settledResult)

  if (settledResult) {
    return setupCard ? (
      <SetupChooseSettled {...props} result={settledResult} />
    ) : (
      <ClarifyToolSettled {...props} result={settledResult} />
    )
  }

  // Stopped mid-prompt with no result — don't leave a dead interactive panel.
  // `session.info` reports running=false while clarify is blocking, so the
  // running flag alone would remount the question as a tool row. Keep the
  // card while a request is open or this instance already submitted.
  if (!messageRunning && !request && !answered) {
    return <ToolFallback {...props} />
  }

  return setupCard ? (
    <SetupChoosePending
      fromArgs={setupArgs}
      onAnswered={() => setAnswered(true)}
      request={request}
      undelivered={undelivered}
    />
  ) : (
    <ClarifyToolPending
      fromArgs={fromArgs}
      onAnswered={() => setAnswered(true)}
      request={request}
      undelivered={undelivered}
    />
  )
}

'use client'

import { type ToolCallMessagePartProps, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import type { TimelinePartMetadata } from '@/lib/chat-messages'
import { $settledClarifyResults, type ClarifyRequest, sessionClarifyRequest } from '@/store/clarify'

import { selectMessageRunning } from '../tool/fallback-model'

import { type ClarifyArgs, readClarifyArgs } from './parse'
import { ClarifyToolPending } from './pending'
import { ClarifyToolSettled } from './settled'
import { useUndeliveredClarify } from './use-undelivered'

type ClarifyToolProps = ToolCallMessagePartProps & Pick<TimelinePartMetadata, 'clarifyRequestId'>

export const ClarifyTool = (props: ClarifyToolProps) => {
  // Answered → settled Q&A (ToolFallback collapsed the answer away).
  if (props.result !== undefined) {
    return <ClarifyToolSettled {...props} />
  }

  return <ClarifyToolLive {...props} />
}

// The request each tool row asked, remembered past its clearing and past a
// remount (a session switch, a stopped turn): a skip or a typed answer settles
// it in the store before (or without) `tool.complete`. Keyed by session,
// message and call: some providers name every call "call_0", and a bare call id
// would show a later card the answer of an earlier one.
const requestIdByToolCall = new Map<string, string>()

// Whether this row renders the session's live request. The transcript re-arm
// ties each request to one row (`clarifyRequestId`): that row follows it, even
// when the gateway asks a batch one card per question on the same row. Every
// other row it saw is history, including an earlier call or a duplicated copy
// that asks the same question. A row no re-arm has seen follows a request its
// own args cover.
function ownsRequest(request: ClarifyRequest | null, row: ClarifyToolProps, args: ClarifyArgs): boolean {
  if (!request) {
    return false
  }

  if (row.clarifyRequestId !== undefined) {
    return row.clarifyRequestId === request.requestId
  }

  const asked = new Set((args.questions ?? []).map(entry => entry.question.trim()))

  return asked.size > 0 && request.questions.every(question => asked.has(question.question.trim()))
}

function ClarifyToolLive(props: ClarifyToolProps) {
  // The tool row is in whichever session's transcript rendered it — read THAT
  // session's clarify (primary or tile), not the globally-active one.
  const sessionId = useStore(useSessionView().$runtimeId)
  const $request = useMemo(() => sessionClarifyRequest(sessionId), [sessionId])
  const request = useStore($request)
  const fromArgs = useMemo(() => readClarifyArgs(props.args), [props.args])
  const messageRunning = useAuiState(selectMessageRunning)
  const messageId = useAuiState(s => s.message.id)
  const rowKey = `${sessionId ?? ''}:${messageId}:${props.toolCallId}`
  // Answering clears the request a beat before `tool.complete` swaps in the
  // settled card. Latch submit so that gap doesn't demote; Stop also clears
  // the request and must still collapse an unanswered card.
  const [answered, setAnswered] = useState(false)
  const settledResults = useStore($settledClarifyResults)
  const owner = ownsRequest(request, props, fromArgs)

  if (request && owner) {
    requestIdByToolCall.set(rowKey, request.requestId)
  }

  // The chat bar and Skip store an answer before `tool.complete`. It shows
  // until this row's next live request: the gateway asks a batch one card per
  // question, and a stored answer that outlived its card would hide the next
  // question while the chat bar answered it.
  const requestId = requestIdByToolCall.get(rowKey)
  const settledResult = !owner && requestId ? settledResults[requestId] : undefined

  const undelivered = useUndeliveredClarify(sessionId, messageRunning && !request && !answered && !settledResult)

  if (settledResult) {
    return <ClarifyToolSettled {...props} result={settledResult} />
  }

  // A row that does not own the live question (an earlier call, a duplicated
  // copy) is history: it renders its own args as a plain tool row, never as
  // the session's one interactive card.
  if (request && !owner && !answered) {
    return <ToolFallback {...props} />
  }

  // Stopped mid-prompt with no result — don't leave a dead interactive panel.
  // `session.info` reports running=false while clarify is blocking, so the
  // running flag alone would remount the question as a tool row. Keep the
  // card while a request is open or this instance already submitted.
  if (!messageRunning && !request && !answered) {
    return <ToolFallback {...props} />
  }

  // A row answered from its own card waits for `tool.complete` with its own
  // args. A live request it does not own belongs to another row.
  return (
    <ClarifyToolPending
      fromArgs={fromArgs}
      onAnswered={() => setAnswered(true)}
      request={owner ? request : null}
      undelivered={undelivered}
    />
  )
}

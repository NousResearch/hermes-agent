import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  type InboxRequestClarification,
  type InboxRequest,
  type ClarifyAnswerResult
} from '@/store/inbox'
import { $gateway } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'

interface ClarifyCardProps {
  clarification: InboxRequestClarification
  onResolved?: () => void
}

const OTHER_LABEL = 'Other…'

function resolveClarifyStatus(result: { status: string }, onResolved?: () => void): 'answered' | 'expired' | 'error' {
  if (result.status === 'ok') {
    onResolved?.()

    return 'answered'
  }

  if (result.status === 'expired') {
    return 'expired'
  }

  return 'error'
}

function SingleClarifyCard({
  clarification,
  onResolved
}: {
  clarification: InboxRequestClarification
  onResolved?: () => void
}) {
  const [selectedChoices, setSelectedChoices] = useState<string[]>([])
  const [freeText, setFreeText] = useState('')
  const [otherText, setOtherText] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [resolved, setResolved] = useState<'answered' | 'expired' | 'error' | null>(null)
  const pinnedGateway = useRef($gateway.get())
  const pinnedProfile = useRef($activeGatewayProfile.get() ?? '')
  const submitLock = useRef(false)
  const mountedRef = useRef(true)

  const { params } = clarification
  const hasChoices = params.choices && params.choices.length > 0
  const isMultiSelect = params.multi_select === true
  const effectiveChoices = useMemo(() => {
    if (!hasChoices) {return []}

    return isMultiSelect ? [...(params.choices ?? []), OTHER_LABEL] : (params.choices ?? [])
  }, [hasChoices, isMultiSelect, params.choices])

  const selectedAnswer = isMultiSelect
    ? selectedChoices.length > 0
      ? JSON.stringify(selectedChoices.filter(c => c !== OTHER_LABEL))
      : null
    : (selectedChoices[0] ?? null)

  const pendingAnswer = selectedAnswer ?? (isMultiSelect && otherText ? otherText.trim() : null) ?? (freeText || null)

  const isScopeValid = useCallback(() => {
    return $gateway.get() === pinnedGateway.current
      && ($activeGatewayProfile.get() ?? '') === pinnedProfile.current
  }, [])

  const submit = useCallback(async (answer: string) => {
    if (submitting || resolved || !answer) {return}
    if (submitLock.current) {return}
    submitLock.current = true
    setSubmitting(true)
    setError(null)

    try {
      if (!isScopeValid()) {
        setError('Gateway changed — re-open to act')
        setSubmitting(false)
        return
      }

      const currentProfile = pinnedProfile.current

      const rpc: InboxRequest = async (method, params) => {
        if (!isScopeValid()) {
          throw new Error('Gateway changed — re-open to act')
        }

        const gw = pinnedGateway.current

        if (!gw) {
          throw new Error('Gateway is unavailable')
        }

        return gw.request(method, params ?? {})
      }

      const result = await rpc('request.answer', {
        id: clarification.request_id,
        result: { answer },
        ...(currentProfile ? { profile: currentProfile } : {})
      }) as ClarifyAnswerResult

      if (!mountedRef.current) {return}
      if (!isScopeValid()) {
        setError('Gateway changed — re-open to act')

        return
      }

      const status = resolveClarifyStatus(result, onResolved)
      setResolved(status)

      if (status === 'error') {
        setError('Unexpected response')
      }
    } catch (err) {
      if (!mountedRef.current) {return}

      setError(err instanceof Error ? err.message : 'Failed to answer')
    } finally {
      submitLock.current = false

      if (mountedRef.current) {
        setSubmitting(false)
      }
    }
  }, [clarification.request_id, isScopeValid, onResolved, resolved, submitting])

  useEffect(() => {
    return () => { mountedRef.current = false }
  }, [])

  if (resolved === 'answered' || resolved === 'expired') {
    return (
      <div className="rounded-md border border-(--ui-stroke-tertiary) bg-foreground/5 px-3 py-2 text-xs text-muted-foreground/70">
        {resolved === 'expired' ? 'Expired' : 'Answered'}
      </div>
    )
  }

  const question = params.question ?? 'The agent has a question'

  const handleSelectChoice = (choice: string) => {
    setFreeText('')
    if (isMultiSelect) {
      setSelectedChoices(prev =>
        prev.includes(choice) ? prev.filter(c => c !== choice) : [...prev, choice]
      )
    } else {
      setSelectedChoices([choice])
    }
  }

  return (
    <div className="rounded-md border border-(--ui-stroke-tertiary) bg-foreground/5">
      <div className="px-3 py-2">
        <div className="flex items-center gap-2 text-xs text-(--ui-text-secondary)">
          <Codicon name="question" size="0.8rem" />
          <span className="font-medium">Question</span>
        </div>
        <p className="mt-1 text-[0.68rem] text-foreground/80">{question}</p>
      </div>
      {hasChoices && (
        <div className="flex flex-col gap-px px-3 pb-2">
          {effectiveChoices.map(choice => (
            <button
              className={`rounded px-2 py-1 text-left text-[0.68rem] transition-colors ${
                selectedChoices.includes(choice)
                  ? 'bg-accent/55 text-foreground'
                  : 'text-muted-foreground/70 hover:bg-(--chrome-action-hover) hover:text-foreground'
              }`}
              key={choice}
              onClick={() => handleSelectChoice(choice)}
              type="button"
            >
              {choice}
            </button>
          ))}
        </div>
      )}
      {hasChoices && isMultiSelect && selectedChoices.includes(OTHER_LABEL) && (
        <div className="px-3 pb-2">
          <input
            className="w-full rounded border border-(--ui-stroke-tertiary) bg-transparent px-2 py-1 text-[0.68rem] text-foreground outline-none focus:border-ring"
            onChange={e => setOtherText(e.target.value)}
            placeholder="Type your answer…"
            value={otherText}
          />
        </div>
      )}
      {!hasChoices && (
        <div className="px-3 pb-2">
          <input
            className="w-full rounded border border-(--ui-stroke-tertiary) bg-transparent px-2 py-1 text-[0.68rem] text-foreground outline-none focus:border-ring"
            onChange={e => setFreeText(e.target.value)}
            placeholder="Type your answer…"
            value={freeText}
          />
        </div>
      )}
      {error && (
        <div className="px-3 pb-1 text-[0.62rem] text-destructive">{error}</div>
      )}
      <div className="flex items-center gap-1.5 px-3 pb-2 pt-1">
        <Button
          disabled={submitting || !pendingAnswer}
          onClick={() => {
            const answer = pendingAnswer
            if (answer) {void submit(answer)}
          }}
          size="xs"
        >
          {submitting ? '\u2026' : 'Submit'}
        </Button>
      </div>
    </div>
  )
}

function BatchClarifyCard({
  clarification,
  onResolved
}: {
  clarification: InboxRequestClarification
  onResolved?: () => void
}) {
  const { params } = clarification
  const questions = useMemo(() => params.questions ?? [], [params.questions])
  const [stagedAnswers, setStagedAnswers] = useState<Record<string, string[]>>({})
  const [stagedDrafts, setStagedDrafts] = useState<Record<string, string>>({})
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [resolved, setResolved] = useState<'answered' | 'expired' | 'error' | null>(null)
  const [lockedQids, setLockedQids] = useState<Set<string>>(new Set())
  const pinnedGateway = useRef($gateway.get())
  const pinnedProfile = useRef($activeGatewayProfile.get() ?? '')
  const submitLock = useRef(false)
  const mountedRef = useRef(true)

  const stagedAnswer = useCallback((qid: string, question: { multi_select: boolean; choices: string[] | null }): string | null => {
    const selected = stagedAnswers[qid] ?? []
    if (selected.length > 0) {
      const filtered = selected.filter(c => c !== OTHER_LABEL)

      return question.multi_select ? JSON.stringify(filtered) : (filtered[0] ?? null)
    }

    const draft = stagedDrafts[qid]?.trim()

    return draft || null
  }, [stagedAnswers, stagedDrafts])

  const answeredCount = questions.filter(q => stagedAnswer(q.qid, q) !== null).length
  const allStaged = answeredCount === questions.length

  const isScopeValid = useCallback(() => {
    return $gateway.get() === pinnedGateway.current
      && ($activeGatewayProfile.get() ?? '') === pinnedProfile.current
  }, [])

  const submit = useCallback(async () => {
    if (submitting || resolved) {return}
    if (submitLock.current) {return}
    submitLock.current = true
    setSubmitting(true)
    setError(null)

    try {
      if (!isScopeValid()) {
        setError('Gateway changed — re-open to act')
        setSubmitting(false)
        return
      }

      const currentProfile = pinnedProfile.current

      // Build answers map
      const answers: Record<string, string> = {}
      for (const q of questions) {
        const answer = stagedAnswer(q.qid, q)

        if (answer !== null) {
          answers[q.qid] = answer
        }
      }

      if (Object.keys(answers).length === 0) {
        setSubmitting(false)
        submitLock.current = false

        return
      }

      const rpc: InboxRequest = async (method, reqParams) => {
        if (!isScopeValid()) {
          throw new Error('Gateway changed — re-open to act')
        }

        const gw = pinnedGateway.current

        if (!gw) {
          throw new Error('Gateway is unavailable')
        }

        return gw.request(method, reqParams ?? {})
      }

      let lastResult: ClarifyAnswerResult = { status: 'ok' }

      for (const q of questions) {
        if (!mountedRef.current || !isScopeValid()) {break}

        const answer = answers[q.qid] ?? ''
        lastResult = await rpc('clarify.lock', {
          request_id: clarification.request_id,
          question_id: q.qid,
          answer,
          ...(currentProfile ? { profile: currentProfile } : {})
        }) as ClarifyAnswerResult

        if (lastResult.status === 'expired') {break}
      }

      if (!mountedRef.current) {return}
      if (!isScopeValid()) {
        setError('Gateway changed — re-open to act')

        return
      }

      const status = resolveClarifyStatus(lastResult, onResolved)
      setResolved(status)

      if (status === 'answered') {
        setLockedQids(new Set(questions.map(q => q.qid)))
      } else if (status === 'error') {
        setError('Unexpected response')
      }
    } catch (err) {
      if (!mountedRef.current) {return}

      setError(err instanceof Error ? err.message : 'Failed to answer')
    } finally {
      submitLock.current = false

      if (mountedRef.current) {
        setSubmitting(false)
      }
    }
  }, [clarification.request_id, isScopeValid, onResolved, questions, resolved, stagedAnswer, submitting])

  useEffect(() => {
    return () => { mountedRef.current = false }
  }, [])

  if (resolved === 'answered' || resolved === 'expired') {
    return (
      <div className="rounded-md border border-(--ui-stroke-tertiary) bg-foreground/5 px-3 py-2 text-xs text-muted-foreground/70">
        {resolved === 'expired' ? 'Expired' : 'Answered'}
      </div>
    )
  }

  return (
    <div className="rounded-md border border-(--ui-stroke-tertiary) bg-foreground/5">
      <div className="px-3 py-2">
        <div className="flex items-center gap-2 text-xs text-(--ui-text-secondary)">
          <Codicon name="list-flat" size="0.8rem" />
          <span className="font-medium">{questions.length} questions</span>
        </div>
      </div>
      <div className="flex flex-col gap-2 px-3 pb-2">
        {questions.map(q => {
          const isLocked = lockedQids.has(q.qid)
          const hasChoices = q.choices && q.choices.length > 0
          const isMultiSelect = q.multi_select === true
          const effectiveChoices = isMultiSelect && hasChoices ? [...(q.choices ?? []), OTHER_LABEL] : q.choices
          const selected = stagedAnswers[q.qid] ?? []

          return (
            <div className="rounded bg-foreground/3 px-2 py-1.5" key={q.qid}>
              <p className="text-[0.68rem] text-foreground/80">{q.question}</p>
              {hasChoices && (
                <div className="mt-1 flex flex-wrap gap-1">
                  {effectiveChoices!.map(choice => (
                    <button
                      className={`rounded px-1.5 py-0.5 text-[0.62rem] transition-colors ${
                        selected.includes(choice)
                          ? 'bg-accent/55 text-foreground'
                          : 'text-muted-foreground/60 hover:bg-(--chrome-action-hover) hover:text-foreground'
                      }`}
                      disabled={isLocked}
                      key={choice}
                      onClick={() => {
                        if (isLocked) {return}

                        if (isMultiSelect) {
                          setStagedAnswers(prev => {
                            const current = prev[q.qid] ?? []
                            const next = current.includes(choice)
                              ? current.filter(c => c !== choice)
                              : [...current, choice]

                            return { ...prev, [q.qid]: next }
                          })
                        } else {
                          setStagedAnswers(prev => ({ ...prev, [q.qid]: [choice] }))
                        }
                      }}
                      type="button"
                    >
                      {choice}
                    </button>
                  ))}
                </div>
              )}
              {isMultiSelect && hasChoices && selected.includes(OTHER_LABEL) && (
                <input
                  className="mt-1 w-full rounded border border-(--ui-stroke-tertiary) bg-transparent px-1.5 py-0.5 text-[0.62rem] text-foreground outline-none focus:border-ring"
                  disabled={isLocked}
                  onChange={e => setStagedDrafts(prev => ({ ...prev, [q.qid]: e.target.value }))}
                  placeholder="Type your answer…"
                  value={stagedDrafts[q.qid] ?? ''}
                />
              )}
              {!hasChoices && (
                <input
                  className="mt-1 w-full rounded border border-(--ui-stroke-tertiary) bg-transparent px-1.5 py-0.5 text-[0.62rem] text-foreground outline-none focus:border-ring"
                  disabled={isLocked}
                  onChange={e => {
                    const val = e.target.value

                    if (val) {
                      setStagedDrafts(prev => ({ ...prev, [q.qid]: val }))
                    } else {
                      setStagedDrafts(prev => {
                        const next = { ...prev }
                        delete next[q.qid]

                        return next
                      })
                    }
                  }}
                  placeholder="Answer…"
                  value={stagedDrafts[q.qid] ?? ''}
                />
              )}
            </div>
          )
        })}
      </div>
      {error && (
        <div className="px-3 pb-1 text-[0.62rem] text-destructive">{error}</div>
      )}
      <div className="flex items-center gap-1.5 px-3 pb-2 pt-1">
        <Button
          disabled={submitting || answeredCount === 0}
          onClick={() => void submit()}
          size="xs"
        >
          {submitting ? '\u2026' : 'Submit answers'}
        </Button>
      </div>
    </div>
  )
}

export function ClarifyCard(props: ClarifyCardProps) {
  const { clarification } = props

  if (clarification.kind === 'batch') {
    return <BatchClarifyCard clarification={clarification} onResolved={props.onResolved} />
  }

  return <SingleClarifyCard clarification={clarification} onResolved={props.onResolved} />
}

'use client'

import { useStore } from '@nanostores/react'
import { type FormEvent, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { translateNow, useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2 } from '@/lib/icons'
import { $gateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import {
  respondUserInput,
  sessionUserInputRequests
} from '@/store/user-input'
import {
  createUserInputDraftStore,
  type UserInputDraftRequest
} from '@/store/user-input-drafts'
import {
  classifyUserInputResult,
  createUserInputSubmitLock
} from '@/store/user-input-result'

type FocusTarget = {
  kind: 'radio' | 'text'
  questionId: string
  requestId: string
  selectionEnd?: number
  selectionStart?: number
  sessionId: string
}

const userInputDraftStore = createUserInputDraftStore()
const userInputSubmitLock = createUserInputSubmitLock()

/** Non-blocking, durable user-input card for one runtime session. */
export function UserInputCard({ sessionId }: { sessionId: string | null }) {
  const { t } = useI18n()
  const $requests = useMemo(() => sessionUserInputRequests(sessionId), [sessionId])
  const requests = useStore($requests)
  const gateway = useStore($gateway)
  const request = requests[0] ?? null
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [submitting, setSubmitting] = useState(false)
  const cardRef = useRef<HTMLElement | null>(null)
  const focusTarget = useRef<FocusTarget | null>(null)

  useEffect(() => {
    setAnswers(request ? userInputDraftStore.get(request as UserInputDraftRequest) : {})
    setSubmitting(false)
  }, [request])

  useEffect(() => {
    if (!request) {return}
    const target = focusTarget.current

    if (!target || target.sessionId !== request.sessionId || target.requestId !== request.requestId) {return}

    const input = Array.from(cardRef.current?.querySelectorAll<HTMLInputElement>('input[data-user-input-question]') ?? [])
      .find(candidate => candidate.dataset.userInputQuestion === target.questionId && candidate.dataset.userInputKind === target.kind)

    if (!input) {return}
    input.focus()

    if (target.kind === 'text' && typeof target.selectionStart === 'number' && typeof target.selectionEnd === 'number') {
      input.setSelectionRange(target.selectionStart, target.selectionEnd)
    }
  }, [request])

  if (!request) {return null}

  const rememberFocus = (questionId: string, kind: FocusTarget['kind'], input: HTMLInputElement) => {
    focusTarget.current = {
      kind,
      questionId,
      requestId: request.requestId,
      ...(kind === 'text' && typeof input.selectionStart === 'number' && typeof input.selectionEnd === 'number'
        ? { selectionEnd: input.selectionEnd, selectionStart: input.selectionStart }
        : {}),
      sessionId: request.sessionId
    }
  }

  const updateAnswer = (questionId: string, value: string) => {
    setAnswers(current => {
      const next = { ...current, [questionId]: value }
      userInputDraftStore.set(request as UserInputDraftRequest, next)

      return next
    })
  }

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const missing = request.questions.find(question => !String(answers[question.id] ?? '').trim())

    if (missing) {
      notifyError(new Error(`Answer required: ${missing.text}`), 'Hermes input is incomplete')

      return
    }

    if (!gateway) {
      notifyError(new Error('Hermes gateway is disconnected.'), 'Could not send Hermes input')

      return
    }

    if (!userInputSubmitLock.acquire(request.sessionId, request.requestId)) {return}

    setSubmitting(true)
    void respondUserInput(gateway, request, answers)
      .then(acknowledgement => {
        if (acknowledgement.accepted) {
          userInputDraftStore.delete(request)

          if (acknowledgement.delivery === 'deferred') {
            notify({
              kind: 'info',
              message: 'Answer recorded. Hermes will use it on the next eligible turn; automatic resume was not confirmed.'
            })
          }

          triggerHaptic('submit')

          return
        }

        if (acknowledgement.clear) {
          userInputDraftStore.delete(request)

          return
        }

        notifyError(
          new Error(acknowledgement.message || 'Hermes did not accept these answers.'),
          'Could not send Hermes input'
        )
      })
      .catch(error => {
        const acknowledgement = classifyUserInputResult(null, 0, error)

        if (!acknowledgement.clear) {
          notifyError(error, 'Could not send Hermes input')
        }
      })
      .finally(() => {
        userInputSubmitLock.release(request.sessionId, request.requestId)
        setSubmitting(false)
      })
  }

  return (
    <section
      aria-labelledby={`user-input-title-${request.requestId}`}
      aria-live="polite"
      className="pointer-events-auto max-h-[80vh] w-[min(32rem,100%)] overflow-y-auto rounded-xl border bg-background p-4 shadow-2xl"
      data-session-id={request.sessionId}
      data-user-input-request={request.requestId}
      ref={cardRef}
      role="region"
    >
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold" id={`user-input-title-${request.requestId}`}>
            {translateNow('notifications.native.inputTitle')}
          </h2>
          {request.context ? <p className="mt-1 text-xs text-muted-foreground">{request.context}</p> : null}
        </div>
        {requests.length > 1 ? <span className="text-xs text-muted-foreground">{requests.length} pending</span> : null}
      </div>

      <form className="grid gap-4" onSubmit={onSubmit}>
        {request.questions.map(question => {
          const hasOptions = question.options.length > 0
          const useTextInput = question.allowFreeText || !hasOptions

          return (
            <fieldset className="grid gap-2" key={question.id}>
              <legend className="text-sm font-medium">{question.text}</legend>
              {hasOptions ? (
                <div aria-label={question.text} className="grid gap-2" role="radiogroup">
                  {question.options.map(option => (
                    <label className="flex cursor-pointer items-center gap-2 rounded-md border px-3 py-2 text-sm hover:bg-muted" key={option}>
                      <input
                        checked={answers[question.id] === option}
                        data-user-input-kind="radio"
                        data-user-input-question={question.id}
                        disabled={submitting}
                        name={`user-input-${request.requestId}-${question.id}`}
                        onChange={() => updateAnswer(question.id, option)}
                        onFocus={event => rememberFocus(question.id, 'radio', event.currentTarget)}
                        type="radio"
                        value={option}
                      />
                      <span>{option}</span>
                    </label>
                  ))}
                </div>
              ) : null}
              {useTextInput ? (
                  <Input
                  aria-label={`${question.text} free text`}
                  data-user-input-kind="text"
                  data-user-input-question={question.id}
                  disabled={submitting}
                  onChange={event => {
                    rememberFocus(question.id, 'text', event.currentTarget)
                    updateAnswer(question.id, event.target.value)
                  }}
                  onFocus={event => rememberFocus(question.id, 'text', event.currentTarget)}
                  onSelect={event => rememberFocus(question.id, 'text', event.currentTarget)}
                  placeholder={hasOptions ? 'Or enter another answer' : 'Your answer'}
                  value={answers[question.id] ?? ''}
                />
              ) : null}
            </fieldset>
          )
        })}
        <div className="flex justify-end">
          <Button disabled={submitting} type="submit">
            {submitting ? <Loader2 className="size-3.5 animate-spin" /> : t.common.send}
          </Button>
        </div>
      </form>
    </section>
  )
}

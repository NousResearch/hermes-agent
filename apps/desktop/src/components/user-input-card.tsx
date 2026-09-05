'use client'

import { useStore } from '@nanostores/react'
import { type FormEvent, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { translateNow, useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2 } from '@/lib/icons'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import {
  respondUserInput,
  sessionUserInputRequests,
  type UserInputRequest
} from '@/store/user-input'

function initialAnswers(request: UserInputRequest | null): Record<string, string> {
  if (!request) {return {}}

  return Object.fromEntries(
    request.questions.flatMap(question => {
      const value = question.defaultValue

      if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
        return [[question.id, String(value)]]
      }

      return []
    })
  )
}

/** Non-blocking, durable user-input card for one runtime session. */
export function UserInputCard({ sessionId }: { sessionId: string | null }) {
  const { t } = useI18n()
  const $requests = useMemo(() => sessionUserInputRequests(sessionId), [sessionId])
  const requests = useStore($requests)
  const gateway = useStore($gateway)
  const request = requests[0] ?? null
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    setAnswers(initialAnswers(request))
    setSubmitting(false)
  }, [request])

  if (!request) {return null}

  const updateAnswer = (questionId: string, value: string) => {
    setAnswers(current => ({ ...current, [questionId]: value }))
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

    setSubmitting(true)
    void respondUserInput(gateway, request, answers)
      .then(() => triggerHaptic('submit'))
      .catch(error => {
        notifyError(error, 'Could not send Hermes input')
        setSubmitting(false)
      })
  }

  return (
    <section
      aria-labelledby={`user-input-title-${request.requestId}`}
      aria-live="polite"
      className="pointer-events-auto fixed bottom-4 right-4 z-50 max-h-[80vh] w-[min(32rem,calc(100vw-2rem))] overflow-y-auto rounded-xl border bg-background p-4 shadow-2xl"
      data-session-id={request.sessionId}
      data-user-input-request={request.requestId}
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
                        disabled={submitting}
                        name={`user-input-${request.requestId}-${question.id}`}
                        onChange={() => updateAnswer(question.id, option)}
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
                  disabled={submitting}
                  onChange={event => updateAnswer(question.id, event.target.value)}
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

import './ask-choice.css'

import { useEffect, useState } from 'react'

import type { AskChoiceRequest } from '@/lib/ask-choice'

export function AskChoiceApp() {
  const [request, setRequest] = useState<AskChoiceRequest | null>(null)

  useEffect(() => {
    const api = window.hermesDesktop?.askChoice

    if (!api) {
      return
    }

    let mounted = true

    const unsubscribe = api.onRequest(next => {
      if (mounted) {
        setRequest(next)
      }
    })

    // Esc dismisses (cancel); 1–N picks the Nth option — matches the hint line.
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && request) {
        event.preventDefault()
        api.cancel({ request_id: request.request_id })
        setRequest(null)

        return
      }

      if (request && /^[1-9]$/.test(event.key)) {
        const idx = Number(event.key) - 1

        if (idx < request.options.length) {
          event.preventDefault()
          api.respond({ request_id: request.request_id, choice: request.options[idx] })
          setRequest(null)
        }
      }
    }

    window.addEventListener('keydown', onKey)

    return () => {
      mounted = false
      unsubscribe?.()
      window.removeEventListener('keydown', onKey)
    }
  }, [request])

  if (!request) {
    // No active request — the window is hidden by main anyway; render nothing.
    return null
  }

  const respond = (choice: string) => {
    const api = window.hermesDesktop?.askChoice
    api?.respond({ request_id: request.request_id, choice })
    setRequest(null)
  }

  return (
    <div className="ask-choice-surface" data-state="active">
      <div aria-label={request.question} className="ask-choice-card" role="dialog">
        <div className="ask-choice-header">
          <span aria-hidden className="ask-choice-glyph">
            ?
          </span>
          <span className="ask-choice-eyebrow">Hermes</span>
        </div>
        <div className="ask-choice-question">{request.question}</div>
        <div className="ask-choice-options">
          {request.options.map((option, index) => (
            <button
              className="ask-choice-option"
              key={`${index}-${option}`}
              onClick={() => respond(option)}
              type="button"
            >
              <span className="ask-choice-option-key">{index + 1}</span>
              <span className="ask-choice-option-label">{option}</span>
            </button>
          ))}
        </div>
        <div className="ask-choice-foot">Press 1–{request.options.length} or Esc to dismiss</div>
      </div>
    </div>
  )
}

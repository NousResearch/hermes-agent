import { type RefObject, useCallback, useEffect, useState } from 'react'

import { visibleClarifyCard } from '@/lib/keybinds/composer-focus-keys'
import type { ClarifyQuestion } from '@/store/clarify'

interface ClarifyKeysOptions {
  enabled: boolean
  formRef: RefObject<HTMLFormElement | null>
  isStaged: (question: ClarifyQuestion) => boolean
  onClear: (question: ClarifyQuestion) => void
  onConfirm: () => void
  onToggle: (question: ClarifyQuestion, choice: string) => void
  other?: boolean
  questions: ClarifyQuestion[]
}

export function useClarifyKeys({
  enabled,
  formRef,
  isStaged,
  onClear,
  onConfirm,
  onToggle,
  other = true,
  questions
}: ClarifyKeysOptions) {
  const [cursor, setCursor] = useState({ question: 0, row: 0 })
  const questionIndex = Math.min(cursor.question, Math.max(questions.length - 1, 0))
  const active = questions[questionIndex]
  const choices = active?.choices ?? []
  const otherRows = other ? 1 : 0
  const row = Math.min(cursor.row, choices.length - 1 + otherRows)

  const focusQuestion = useCallback(
    (index: number) => setCursor(current => (current.question === index ? current : { question: index, row: 0 })),
    []
  )

  const onOtherFocus = useCallback(
    (index: number) => setCursor({ question: index, row: questions[index]?.choices?.length ?? 0 }),
    [questions]
  )

  const focusOther = useCallback(
    (index: number) => {
      onOtherFocus(index)

      const block = formRef.current?.querySelectorAll<HTMLElement>('[data-clarify-batch-question]')[index]

      block?.querySelector<HTMLTextAreaElement>('textarea')?.focus()
    },
    [formRef, onOtherFocus]
  )

  const nextUnstaged = useCallback(
    (index: number): null | number => {
      for (let offset = 1; offset < questions.length; offset += 1) {
        const candidate = (index + offset) % questions.length

        if (!isStaged(questions[candidate])) {
          return candidate
        }
      }

      return null
    },
    [isStaged, questions]
  )

  const pick = useCallback(
    (index: number, choiceIndex: number) => {
      const question = questions[index]
      const choice = question?.choices?.[choiceIndex]

      if (!question || choice === undefined) {
        return
      }

      onToggle(question, choice)

      const next = question.multiSelect ? null : nextUnstaged(index)

      if (next !== null && !questions[next]?.choices?.length) {
        focusOther(next)

        return
      }

      setCursor(next === null ? { question: index, row: choiceIndex } : { question: next, row: 0 })
    },
    [focusOther, nextUnstaged, onToggle, questions]
  )

  const move = useCallback(
    (delta: number) => {
      if (!active) {
        return
      }

      if (!active.multiSelect) {
        onClear(active)
      }

      const itemCount = choices.length + otherRows

      setCursor({ question: questionIndex, row: (row + delta + itemCount) % itemCount })
    },
    [active, choices.length, onClear, otherRows, questionIndex, row]
  )

  const activate = useCallback(() => {
    if (!active) {
      return
    }

    const choice = active.choices?.[row]

    if (active.multiSelect && choice !== undefined) {
      pick(questionIndex, row)

      return
    }

    if (isStaged(active)) {
      onConfirm()

      return
    }

    if (choice !== undefined) {
      pick(questionIndex, row)

      return
    }

    focusOther(questionIndex)
  }, [active, focusOther, isStaged, onConfirm, pick, questionIndex, row])

  useEffect(() => {
    if (!enabled || !active) {
      return
    }

    const pickByIndex = (event: globalThis.KeyboardEvent, index: number) => {
      if (index < choices.length) {
        event.preventDefault()
        pick(questionIndex, index)
      } else if (other && index === choices.length) {
        event.preventDefault()
        focusOther(questionIndex)
      }
    }

    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || event.defaultPrevented) {
        return
      }

      // Not the visible card ⇒ not our keystroke. Inactive tabs stay MOUNTED,
      // so every parked clarify keeps a live `window` listener; without this the
      // card that acts is whichever mounted first, and answering the question in
      // front of you silently answers a background session's question instead —
      // resuming an agent turn the user never saw. Same resolver the composer's
      // `clarifyCardOwnsKey` yields to, so the two cannot disagree about which
      // card is live.
      if (visibleClarifyCard() !== formRef.current) {
        return
      }

      const focused = document.activeElement as HTMLElement | null

      if (
        focused &&
        (focused.isContentEditable ||
          (focused.matches('a[href], button, input, select, textarea, [role="button"]') &&
            !focused.matches('button[data-choice]')))
      ) {
        return
      }

      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        if (choices.length > 0) {
          event.preventDefault()
          move(event.key === 'ArrowDown' ? 1 : -1)
        }

        return
      }

      if (/^[1-9]$/.test(event.key)) {
        pickByIndex(event, Number(event.key) - 1)

        return
      }

      const key = event.key.toLowerCase()

      // Only the letters this card actually renders a row for. Anything past
      // the last row belongs to the composer — the user is typing a message
      // instead of picking an option, and swallowing the keystroke here would
      // make the first letter of it vanish.
      if (key.length === 1 && key >= 'a' && key <= 'z') {
        pickByIndex(event, key.charCodeAt(0) - 97)

        return
      }

      if (event.key === 'Enter') {
        event.preventDefault()
        activate()
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [activate, active, choices.length, enabled, focusOther, formRef, move, other, pick, questionIndex])

  return { activeQuestion: questionIndex, cursorRow: row, focusQuestion, onOtherFocus, pick }
}

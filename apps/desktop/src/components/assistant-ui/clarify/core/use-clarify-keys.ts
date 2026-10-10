import { type RefObject, useCallback, useEffect, useState } from 'react'

import type { ClarifyQuestion } from '@/store/clarify'

import { arrowMove, isForeignKeystroke, shortcutIndex } from './use-clarify-keys-handlers'

interface ClarifyKeysOptions {
  columns?: number
  enabled: boolean
  formRef: RefObject<HTMLFormElement | null>
  initialRow?: number
  isStaged: (question: ClarifyQuestion, row?: number) => boolean
  onClear?: (question: ClarifyQuestion) => void
  onConfirm: () => void
  onToggle: (question: ClarifyQuestion, choice: string) => void
  other?: boolean
  questions: ClarifyQuestion[]
  shortcuts?: boolean
}

export function useClarifyKeys({
  columns,
  enabled,
  formRef,
  initialRow = 0,
  isStaged,
  onClear,
  onConfirm,
  onToggle,
  other = true,
  questions,
  shortcuts = true
}: ClarifyKeysOptions) {
  const [cursor, setCursor] = useState({ question: 0, row: initialRow })
  const questionIndex = Math.min(cursor.question, Math.max(questions.length - 1, 0))
  const active = questions[questionIndex]
  const choices = active?.choices ?? []
  const otherRows = other ? 1 : 0
  const row = Math.min(cursor.row, choices.length - 1 + otherRows)

  const focusQuestion = useCallback(
    (index: number) => setCursor(current => (current.question === index ? current : { question: index, row: 0 })),
    []
  )

  const focusRow = useCallback(
    (index: number, choiceIndex: number) =>
      setCursor(current =>
        current.question === index && current.row === choiceIndex ? current : { question: index, row: choiceIndex }
      ),
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
    (delta: number, wrap = true) => {
      const itemCount = choices.length + otherRows

      if (!active || (!wrap && (row + delta < 0 || row + delta >= itemCount))) {
        return
      }

      if (!active.multiSelect) {
        onClear?.(active)
      }

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

    if (isStaged(active, row)) {
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
      if (isForeignKeystroke(event, formRef.current)) {
        return
      }

      const arrow = arrowMove(event.key, columns)

      if (arrow) {
        if (choices.length > 0) {
          event.preventDefault()
          move(arrow.delta, arrow.wrap)
        }

        return
      }

      const index = shortcuts ? shortcutIndex(event.key) : null

      if (index !== null) {
        pickByIndex(event, index)

        return
      }

      if (event.key === 'Enter') {
        event.preventDefault()
        activate()
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [
    activate,
    active,
    choices.length,
    columns,
    enabled,
    focusOther,
    formRef,
    move,
    other,
    pick,
    questionIndex,
    shortcuts
  ])

  return { activeQuestion: questionIndex, cursorRow: row, focusQuestion, focusRow, onOtherFocus, pick }
}

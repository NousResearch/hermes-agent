import { normalizeChoices } from '@/store/clarify'

import { parseMaybeObject } from '../tool/fallback-model/format'

export interface ClarifyArgs {
  questions?: { question: string; choices?: string[] | null; multiSelect?: boolean }[]
}

function stringField(row: Record<string, unknown>, key: string): string | undefined {
  const value = row[key]

  return typeof value === 'string' ? value : undefined
}

export function readClarifyArgs(args: unknown): ClarifyArgs {
  const row = parseMaybeObject(args)

  if (!Array.isArray(row.questions)) {
    return {}
  }

  // Entries are normalized leniently here (qid comes from the gateway request, not args).
  const questions = row.questions
    .map(entry => {
      const item = parseMaybeObject(entry)
      const text = stringField(item, 'question')

      if (!text) {
        return null
      }

      const itemChoices = normalizeChoices(item.choices)

      return {
        choices: itemChoices.length > 0 ? itemChoices : null,
        multiSelect: item.multi_select === true && itemChoices.length > 0,
        question: text
      }
    })
    .filter((entry): entry is NonNullable<typeof entry> => entry !== null)

  return questions.length > 0 ? { questions } : {}
}

export interface ClarifyResponse {
  question?: string
  answer?: string | string[]
  unanswered: boolean
}

/** Parse clarify tool JSON (`responses` array). */
export function readClarifyResult(result: unknown): { responses: ClarifyResponse[] } {
  const row = parseMaybeObject(result)

  if (!Array.isArray(row.responses)) {
    return { responses: [] }
  }

  const responses = row.responses.map((entry): ClarifyResponse => {
    const item = parseMaybeObject(entry)
    const answer = item.user_response

    return {
      answer: Array.isArray(answer) ? answer.map(String) : typeof answer === 'string' ? answer : undefined,
      question: stringField(item, 'question'),
      unanswered: item.status === 'unanswered'
    }
  })

  return { responses }
}

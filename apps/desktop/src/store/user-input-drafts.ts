export interface UserInputDraftQuestion {
  allowFreeText: boolean
  defaultValue?: unknown
  id: string
  options: string[]
  text?: string
}

export interface UserInputDraftRequest {
  questions: UserInputDraftQuestion[]
  requestId: string
  sessionId: string
}

export type UserInputDraft = Record<string, string>

const scalar = (value: unknown): string =>
  typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean' ? String(value) : ''

export const draftKey = (request: Pick<UserInputDraftRequest, 'requestId' | 'sessionId'>): string =>
  `${request.sessionId}:${request.requestId}`

function validAnswers(request: UserInputDraftRequest, answers: UserInputDraft): UserInputDraft {
  const next: UserInputDraft = {}

  for (const question of request.questions) {
    const value = answers[question.id]

    if (typeof value !== 'string' || !value) {continue}

    if (!question.allowFreeText && question.options.length && !question.options.includes(value)) {continue}
    next[question.id] = value
  }

  return next
}

function defaults(request: UserInputDraftRequest): UserInputDraft {
  return Object.fromEntries(
    request.questions.flatMap(question => {
      const value = scalar(question.defaultValue)

      return value ? [[question.id, value]] : []
    })
  )
}

export function createUserInputDraftStore() {
  const drafts = new Map<string, UserInputDraft>()

  return {
    delete(request: Pick<UserInputDraftRequest, 'requestId' | 'sessionId'>): void {
      drafts.delete(draftKey(request))
    },
    get(request: UserInputDraftRequest): UserInputDraft {
      const key = draftKey(request)
      const current = drafts.has(key) ? drafts.get(key) ?? {} : defaults(request)
      const next = validAnswers(request, current)
      drafts.set(key, next)

      return { ...next }
    },
    set(request: UserInputDraftRequest, answers: UserInputDraft): UserInputDraft {
      const next = validAnswers(request, answers)
      drafts.set(draftKey(request), next)

      return { ...next }
    }
  }
}

type ReplayState = {
  generation: number
  owner: unknown
}

type ReplayToken = {
  generation: number
  owner: unknown
  sessionId: string
}

export function createUserInputReplayGuard() {
  const states = new Map<string, ReplayState>()

  return {
    begin(sessionId: string, owner: unknown): ReplayToken {
      const current = states.get(sessionId) ?? { generation: 0, owner: null }
      const next = { generation: current.generation + 1, owner }
      states.set(sessionId, next)

      return { generation: next.generation, owner, sessionId }
    },

    invalidate(sessionId: string): void {
      const current = states.get(sessionId) ?? { generation: 0, owner: null }
      states.set(sessionId, { generation: current.generation + 1, owner: null })
    },

    isCurrent(token: ReplayToken | null | undefined): boolean {
      if (!token) {return false}
      const current = states.get(token.sessionId)

      return Boolean(current && token.generation === current.generation && token.owner === current.owner)
    }
  }
}

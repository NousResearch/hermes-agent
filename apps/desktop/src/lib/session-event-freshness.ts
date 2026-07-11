let terminalEventGeneration = 0
const terminalGenerationBySession = new Map<string, number>()

export function markSessionTerminalEvent(sessionId: string): number {
  terminalEventGeneration += 1
  terminalGenerationBySession.set(sessionId, terminalEventGeneration)

  return terminalEventGeneration
}

export function getTerminalEventGeneration(): number {
  return terminalEventGeneration
}

export function sessionTerminatedAfter(sessionId: string, generation: number): boolean {
  return (terminalGenerationBySession.get(sessionId) ?? 0) > generation
}

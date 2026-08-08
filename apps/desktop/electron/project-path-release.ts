import path from 'node:path'

export interface ReleasableTerminal {
  id: string
  launchCwd?: null | string
  remote: boolean
  writeExit: () => void
  isActive: () => boolean
}

export function pathContains(root: string, candidate: string) {
  const relative = path.relative(path.resolve(root), path.resolve(candidate))

  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative))
}

export async function gracefulReleaseSessions(
  projectPath: string,
  sessions: ReleasableTerminal[],
  options: { now?: () => number; pause?: (ms: number) => Promise<void>; timeoutMs?: number } = {}
) {
  const matching = sessions.filter(session => !session.remote && session.launchCwd && pathContains(projectPath, session.launchCwd))

  for (const session of matching) {
    session.writeExit()
  }

  const now = options.now ?? Date.now
  const pause = options.pause ?? (ms => new Promise(resolve => setTimeout(resolve, ms)))
  const deadline = now() + (options.timeoutMs ?? 3_000)

  while (now() < deadline) {
    const active = matching.filter(session => session.isActive())

    if (active.length === 0) {
      return { released: true, releasedTerminalIds: matching.map(session => session.id) }
    }

    await pause(50)
  }

  return {
    released: false,
    releasedTerminalIds: matching.filter(session => !session.isActive()).map(session => session.id),
    activeTerminalIds: matching.filter(session => session.isActive()).map(session => session.id),
    error: 'TERMINAL_DID_NOT_EXIT_GRACEFULLY'
  }
}

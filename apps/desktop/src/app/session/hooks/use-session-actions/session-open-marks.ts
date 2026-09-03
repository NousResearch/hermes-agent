/**
 * Fixed local-only timing marks for one Desktop session open. These names are
 * intentionally the entire payload: never attach transcript content, session
 * identifiers, profiles, or arbitrary `detail` values to the performance API.
 */
export const SESSION_OPEN_MARKS = [
  'hermes.session.select',
  'hermes.session.cache.commit',
  'hermes.session.rest.commit',
  'hermes.session.resume.ready',
  'hermes.session.agent.ready',
  'hermes.session.history.ready'
] as const

export type SessionOpenMark = (typeof SESSION_OPEN_MARKS)[number]

export function markSessionOpen(name: SessionOpenMark): void {
  performance.mark(name)
}

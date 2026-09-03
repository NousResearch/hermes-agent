const TEAM_HERMES_EXECUTABLE_RE =
  /\/desktop-builds\/team-hermes-(?:custom|desktop)-[^/]+\/win-unpacked\/(?:hermes|team hermes desktop)\.exe$/i

/** True only for immutable Team Hermes packages stored outside the updater checkout. */
export function isProtectedTeamHermesExecutable(executablePath: string): boolean {
  return TEAM_HERMES_EXECUTABLE_RE.test(String(executablePath || '').replace(/\\/g, '/'))
}

export const PROTECTED_TEAM_HERMES_UPDATE_MESSAGE =
  'Team Hermes is a protected custom edition and cannot replace itself with the official desktop build. ' +
  'Update the Hermes backend/source separately; this interface and its launcher will remain unchanged.'

/** Shared validation for room storage, sync, settings and the turn engine. */
export const GROUP_CHAT_DEFAULT_MAX_BOT_TURNS = 10
export const GROUP_CHAT_MIN_BOT_TURNS = 1
export const GROUP_CHAT_MAX_BOT_TURNS = 100

export function isValidGroupMaxBotTurns(value: unknown): value is number {
  return (
    typeof value === 'number' &&
    Number.isInteger(value) &&
    value >= GROUP_CHAT_MIN_BOT_TURNS &&
    value <= GROUP_CHAT_MAX_BOT_TURNS
  )
}

/** Old rooms and malformed remote/storage values retain the safe default. */
export function normalizeGroupMaxBotTurns(value: unknown): number {
  return isValidGroupMaxBotTurns(value) ? value : GROUP_CHAT_DEFAULT_MAX_BOT_TURNS
}

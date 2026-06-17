export type CompanionAvatarState = 'idle' | 'listening' | 'thinking' | 'speaking' | 'acting'

export type CompanionAvatarEvent = 'listen-start' | 'think-start' | 'speech-start' | 'action-start' | 'reset'

export function nextCompanionAvatarState(
  current: CompanionAvatarState,
  event: CompanionAvatarEvent
): CompanionAvatarState {
  if (event === 'listen-start') {
    return 'listening'
  }

  if (event === 'think-start') {
    return 'thinking'
  }

  if (event === 'speech-start') {
    return 'speaking'
  }

  if (event === 'action-start') {
    return 'acting'
  }

  if (event === 'reset') {
    return 'idle'
  }

  return current
}

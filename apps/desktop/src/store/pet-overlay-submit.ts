import {
  armNextReplyForProfile,
  armPetLiveSessionReply,
  disarmNextReplyForProfile,
  disarmPetReplyArm
} from './pet-live-session'

interface PetOverlayPromptSubmitInput {
  profile: string
  runtimeSessionId: string | null
  submitText: (text: string) => Promise<boolean>
  text: string
}

export async function submitPetOverlayPrompt({
  profile,
  runtimeSessionId,
  submitText,
  text
}: PetOverlayPromptSubmitInput): Promise<boolean> {
  const sessionId = runtimeSessionId?.trim() ?? ''

  if (sessionId) {
    armPetLiveSessionReply(profile, sessionId)
  } else {
    armNextReplyForProfile(profile)
  }

  let accepted = false

  try {
    accepted = await submitText(text)
  } catch {
    accepted = false
  }

  if (!accepted) {
    if (sessionId) {
      disarmPetReplyArm(profile, sessionId)
    } else {
      disarmNextReplyForProfile(profile)
    }
  }

  return accepted
}

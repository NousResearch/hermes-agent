export type TrayPetCommand = 'pop-out' | 'pop-in'

interface PetStartupPolicy {
  enabled: boolean
  available: boolean
  poppedOut: boolean
  alreadyRequested: boolean
}

export function petStartupCommand({
  enabled,
  available,
  poppedOut,
  alreadyRequested
}: PetStartupPolicy): TrayPetCommand | null {
  return enabled && available && !poppedOut && !alreadyRequested ? 'pop-out' : null
}

export const GUEST_ONBOARDING_ENV = 'HERMES_GUEST_ONBOARDING'
export const GUEST_ONBOARDING_FLAG = '--guest-onboarding'

export function guestOnboardingEnabled(
  argv: readonly string[] = process.argv,
  env: NodeJS.ProcessEnv = process.env
): boolean {
  return env[GUEST_ONBOARDING_ENV] === '1' || argv.includes(GUEST_ONBOARDING_FLAG)
}

export function desktopBackendSpawnEnv(base: NodeJS.ProcessEnv, guestOnboarding: boolean): NodeJS.ProcessEnv {
  return { ...base, [GUEST_ONBOARDING_ENV]: guestOnboarding ? '1' : '0' }
}

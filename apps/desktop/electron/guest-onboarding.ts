// The Nous free tier is gated by ONE launch-time decision. The Python backend
// reads HERMES_GUEST_ONBOARDING and treats exactly "1" as on; the desktop
// decides once at launch (env or `--guest-onboarding` argv) and stamps that
// answer onto every backend it spawns, so the app and its backends can never
// disagree about whether the free tier is live.

export const GUEST_ONBOARDING_ENV = 'HERMES_GUEST_ONBOARDING'
export const GUEST_ONBOARDING_FLAG = '--guest-onboarding'
// Skip the first-run film. A rehearsal aid: the intro is a one-time reveal,
// so anyone iterating on the guided chat behind it otherwise sits through it
// on every fresh HERMES_HOME. Renderer-only; the backend never sees it.
export const SKIP_INTRO_ENV = 'HERMES_SKIP_INTRO'
export const SKIP_INTRO_FLAG = '--skip-intro'

export function guestOnboardingEnabled(
  argv: readonly string[] = process.argv,
  env: NodeJS.ProcessEnv = process.env
): boolean {
  return env[GUEST_ONBOARDING_ENV] === '1' || argv.includes(GUEST_ONBOARDING_FLAG)
}

export function skipIntroEnabled(argv: readonly string[] = process.argv, env: NodeJS.ProcessEnv = process.env): boolean {
  return env[SKIP_INTRO_ENV] === '1' || argv.includes(SKIP_INTRO_FLAG)
}

// Outermost wrapper for a backend spawn env: the flag is written LAST so no
// earlier spread (process.env, backend.env) can resurrect a stray value, and
// "off" is an explicit '0' rather than an absent key so a '1' inherited from
// the parent's environment cannot leak into a backend the launch decided off.
export function desktopBackendSpawnEnv(base: NodeJS.ProcessEnv, guestOnboarding: boolean): NodeJS.ProcessEnv {
  return { ...base, [GUEST_ONBOARDING_ENV]: guestOnboarding ? '1' : '0' }
}

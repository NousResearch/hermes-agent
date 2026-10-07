export function isOnboardingEnabled(): boolean {
  return window.rabbitDesktop?.guestOnboardingEnabled === true
}

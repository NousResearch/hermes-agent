export function isOnboardingEnabled(): boolean {
  return window.hermesDesktop?.guestOnboardingEnabled === true
}

import { usePortalShell } from '@/store/shell-variant'

export function useChatLanding(
  showIntro: boolean,
  onboarding: boolean,
  setup: boolean,
  personality?: string,
  seed?: number
) {
  const enabled = usePortalShell()
  const centered = enabled && showIntro && !onboarding && !setup

  return { centered, intro: showIntro && !centered ? { personality, seed } : undefined }
}

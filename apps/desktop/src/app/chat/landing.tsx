import chatLandingArt from '@/assets/chat-landing.webp'
import { usePortalShell } from '@/store/shell-variant'

export function useChatLanding(showIntro: boolean, onboarding: boolean, setup: boolean, personality?: string, seed?: number) {
  const enabled = usePortalShell()
  const centered = enabled && showIntro && !onboarding && !setup

  return { centered, intro: showIntro && !centered ? { personality, seed } : undefined }
}

export function ChatLandingArt({ visible }: { visible: boolean }) {
  return visible ? <img alt="" aria-hidden className="chat-landing-art" src={chatLandingArt} /> : null
}

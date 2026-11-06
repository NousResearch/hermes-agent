import { usePortalRedesign } from '@/app/redesign/use-portal-redesign'
import chatLandingArt from '@/assets/chat-landing.webp'

export function useChatLanding(showIntro: boolean, onboarding: boolean, setup: boolean, personality?: string, seed?: number) {
  const enabled = usePortalRedesign()
  const centered = enabled && showIntro && !onboarding && !setup

  return { centered, intro: showIntro && !centered ? { personality, seed } : undefined }
}

export function ChatLandingArt({ visible }: { visible: boolean }) {
  return visible ? <img alt="" aria-hidden className="chat-landing-art" src={chatLandingArt} /> : null
}

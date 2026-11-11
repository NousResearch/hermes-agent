import chatLandingArt from '@/assets/chat-landing.webp'
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

/** The redesigned stage's hero photo: strong behind the new-session landing, a
 *  faint trace behind a conversation. One element across both states, so the
 *  first send fades it instead of remounting it. */
export function ChatStageArt({ landing }: { landing: boolean }) {
  const portal = usePortalShell()

  return portal ? (
    <img alt="" aria-hidden className="chat-stage-art" data-landing={landing ? '' : undefined} src={chatLandingArt} />
  ) : null
}

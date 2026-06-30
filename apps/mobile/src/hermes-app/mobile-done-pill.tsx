import { useNavigate } from 'react-router-dom'

import { NEW_CHAT_ROUTE } from './routes'

const mobileStandalone =
  typeof window !== 'undefined' &&
  Boolean((window as { __HERMES_MOBILE_STANDALONE__?: boolean }).__HERMES_MOBILE_STANDALONE__)

// Floating "Done" pill rendered by full-screen mobile views that don't wrap
// their content in OverlayView (Skills, Messaging, Artifacts). Mirrors the
// styling of OverlayView's mobile close affordance so the dismiss control
// looks identical across menus.
export function MobileDonePill() {
  const navigate = useNavigate()

  if (!mobileStandalone) {
    return null
  }

  return (
    <button
      aria-label="Done"
      className="fixed bottom-[calc(env(safe-area-inset-bottom)+0.75rem)] right-4 z-[60] rounded-full bg-(--theme-primary) px-4 py-2 text-base font-semibold text-white shadow-lg active:opacity-80"
      onClick={() => navigate(NEW_CHAT_ROUTE)}
      type="button"
    >
      Done
    </button>
  )
}

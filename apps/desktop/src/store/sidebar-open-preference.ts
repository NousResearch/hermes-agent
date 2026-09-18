import { Codecs, persistentAtom } from '@/lib/persisted'

// Sidebar rows can mix sessions from other profiles in all-profiles views, so
// this preference is intentionally desktop-global (not profile namespaced).
export const $sidebarSessionsOpenInNewTab = persistentAtom(
  'hermes.desktop.sidebarSessionsOpenInNewTab',
  false,
  Codecs.bool
)

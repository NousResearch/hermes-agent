import './portal-composer-edge.css'

import { composerInputBacking } from '@/components/chat/composer-dock'

import { usePortalRedesign } from './use-portal-redesign'

/** Decorative only: no editor remount, timers, or session state. */
export function ComposerSurfaceBacking({
  hudMode = false,
  guidedChat = false
}: {
  hudMode?: boolean
  guidedChat?: boolean
}) {
  const redesign = usePortalRedesign()

  return (
    <>
      {redesign && !hudMode && !guidedChat && <span aria-hidden className="portal-composer-edge" />}
      <div aria-hidden className={composerInputBacking} />
    </>
  )
}

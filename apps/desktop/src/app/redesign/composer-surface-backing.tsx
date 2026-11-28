import './portal-composer.css'

import { composerInputBacking } from '@/components/chat/composer-dock'
import { usePortalShell } from '@/store/shell-variant'

/** Decorative only: no editor remount, timers, or session state. */
export function ComposerSurfaceBacking({ guidedChat = false }: { guidedChat?: boolean }) {
  const portal = usePortalShell()

  return (
    <>
      {portal && !guidedChat && <span aria-hidden className="portal-composer-edge" />}
      <div aria-hidden className={composerInputBacking} />
    </>
  )
}

import type { CSSProperties } from 'react'
import { useEffect } from 'react'

import { TITLEBAR_HEIGHT } from '@/app/shell/titlebar'
import { useI18n } from '@/i18n'
import { installHubSkill } from '@/store/hub-actions'
import { notify, notifyError } from '@/store/notifications'

// The REAL Skills Hub page (docs site) embedded full-window — the same
// `?embed=picker` trick the Capabilities pane uses, but at 100% size in its
// own OS window (the embedded picker is scaled down to 0.75 and capped at 75%
// of the app window, which makes the catalog hard to read). Picks post
//   { type: 'hermes-skill-pick', name, identifier, installCmd, source }
// to the parent window; we validate the origin and route the install through
// the standard hub action pipeline, then broadcast the change to every window
// so the primary's Skills list refreshes.
const HUB_ORIGIN = 'https://hermes-agent.nousresearch.com'
const HUB_PICKER_URL = `${HUB_ORIGIN}/docs/skills?embed=picker`

interface SkillPickMessage {
  identifier?: string
  installCmd?: string
  name?: string
  source?: string
  type?: string
}

/**
 * Dedicated shell for `?win=skills-hub`: the Skills Hub catalog, full-window,
 * no session sidebar or layout tree. Same hub page the Capabilities pane
 * embeds — just readable.
 */
export function SkillsHubPopoutShell() {
  const { t } = useI18n()
  const h = t.skills.hub

  useEffect(() => {
    const onMessage = (event: MessageEvent) => {
      if (event.origin !== HUB_ORIGIN) {
        return
      }

      const data = event.data as SkillPickMessage | null

      if (!data || data.type !== 'hermes-skill-pick' || !data.name) {
        return
      }

      const target = String(data.identifier || data.name)
      const label = String(data.name)

      notify({ kind: 'success', title: h.installStarted(label), message: h.actionLog })
      void installHubSkill(target).catch(err => notifyError(err, h.actionFailed))
    }

    window.addEventListener('message', onMessage)

    return () => window.removeEventListener('message', onMessage)
  }, [h])

  return (
    <div
      className="flex h-screen min-h-0 w-screen flex-col bg-(--ui-bg-chrome) text-(--ui-text-primary)"
      data-contrib-shell=""
      style={{ '--titlebar-height': `${TITLEBAR_HEIGHT}px` } as CSSProperties}
    >
      <div aria-hidden="true" className="relative shrink-0 bg-(--ui-bg-chrome)" style={{ height: TITLEBAR_HEIGHT }}>
        {/* Same traffic-light / native-overlay carve-out as the main titlebar:
            a full-bar drag region would eat the window buttons. */}
        <div className="pointer-events-none absolute inset-y-0 left-0 w-(--titlebar-controls-left,14px) [-webkit-app-region:drag]" />
        <div className="pointer-events-none absolute inset-y-0 left-[calc(var(--titlebar-controls-left,14px)+(var(--titlebar-control-size,24px)*2)+0.75rem)] right-[calc(var(--titlebar-tools-right,0.75rem)+0.75rem)] [-webkit-app-region:drag]" />
      </div>
      <div className="relative min-h-0 min-w-0 flex-1 overflow-hidden">
        {/* allow-popups: the Docusaurus navbar / skill cards link out with
            target="_blank" (GitHub, Discord, source repos). Without it the
            sandbox swallows the popup silently. The main process routes
            http/https/mailto through openExternalUrl (audited allowlist) via
            decideHubWindowOpen; nothing else may open. */}
        <iframe
          sandbox="allow-scripts allow-same-origin allow-popups"
          src={HUB_PICKER_URL}
          style={{
            background: 'transparent',
            border: 'none',
            height: '100%',
            width: '100%'
          }}
          title={h.pickerTitle}
        />
      </div>
    </div>
  )
}

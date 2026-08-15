import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { installHubSkill } from '@/store/hub-actions'
import { notify, notifyError } from '@/store/notifications'

// The REAL Skills Hub page (docs site) embedded as a one-click picker — the
// same trick the Bot Mode agent editor uses. `?embed=picker` hides the docs
// chrome and adds a "+ Add to this Agent" button per card, which posts
//   { type: 'hermes-skill-pick', name, identifier, installCmd, source }
// to the parent window. We validate the origin and route the install through
// the standard hub action pipeline (background action + tailed log + Skills
// list invalidation), scoped to the Capabilities profile selector.
const HUB_ORIGIN = 'https://hermes-agent.nousresearch.com'
const HUB_PICKER_URL = `${HUB_ORIGIN}/docs/skills?embed=picker`

interface SkillPickMessage {
  identifier?: string
  installCmd?: string
  name?: string
  source?: string
  type?: string
}

interface EmbeddedHubPickerProps {
  /** Capabilities profile-scope override — installs land in THIS profile;
   *  undefined/null targets the app-wide active profile. */
  profile?: null | string
}

/** Collapsible "browse the full hub" strip for the Skills tab: a resizable
 *  iframe of the live Skills Hub where every card installs with one click. */
export function EmbeddedHubPicker({ profile }: EmbeddedHubPickerProps) {
  const { t } = useI18n()
  const h = t.skills.hub
  const [open, setOpen] = useState(false)

  // Picker messages from the embedded hub page. Origin-checked; installs route
  // through the same store pipeline the Browse Hub tab uses, so the action log,
  // optimistic flips, and Skills-list refresh all come for free.
  useEffect(() => {
    if (!open) {
      return undefined
    }

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
      void installHubSkill(target, profile).catch(err => notifyError(err, h.actionFailed))
    }

    window.addEventListener('message', onMessage)

    return () => window.removeEventListener('message', onMessage)
  }, [h, open, profile])

  return (
    <div className="border-b border-(--ui-stroke-secondary)">
      <div className="flex items-center justify-between px-3 py-1.5">
        <span className="text-[0.7rem] font-medium text-(--ui-text-tertiary)">{h.pickerTitle}</span>
        <Button onClick={() => setOpen(v => !v)} size="xs" variant="text">
          {open ? h.pickerHide : h.pickerBrowse}
        </Button>
      </div>
      {open && (
        <div className="grid gap-1 px-3 pb-2">
          {/* Resizable viewport: native CSS resize handle (bottom-right corner)
              lets the user drag it larger/smaller. The iframe is rendered
              oversized and scaled DOWN (133% × 0.75) so the hub page starts
              zoomed out — the cross-origin page itself can't be styled, but
              scaling the frame is ours. */}
          <div
            style={{
              border: '1px solid var(--ui-stroke-secondary)',
              borderRadius: 8,
              height: 480,
              maxWidth: '100%',
              minHeight: 240,
              minWidth: 320,
              overflow: 'hidden',
              position: 'relative',
              resize: 'vertical',
              width: '100%'
            }}
          >
            <iframe
              sandbox="allow-scripts allow-same-origin"
              src={HUB_PICKER_URL}
              style={{
                background: 'transparent',
                border: 'none',
                height: '133.34%',
                transform: 'scale(0.75)',
                transformOrigin: 'top left',
                width: '133.34%'
              }}
              title={h.pickerTitle}
            />
          </div>
          <p className="px-1 text-[0.65rem] leading-4 text-(--ui-text-quaternary)">{h.pickerHint}</p>
        </div>
      )}
    </div>
  )
}

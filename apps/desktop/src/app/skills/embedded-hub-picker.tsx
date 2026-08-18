import { useStore } from '@nanostores/react'
import { type PointerEvent as ReactPointerEvent, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $hubActions, installHubSkill, UPDATE_ALL_KEY, updateHubSkills } from '@/store/hub-actions'
import { notify, notifyError } from '@/store/notifications'
import { $paneHeightOverride, setPaneHeightOverride } from '@/store/panes'

// The REAL Skills Hub page (docs site) embedded as a one-click picker — the
// same trick the Bot Mode agent editor uses. `?embed=picker` hides the docs
// chrome and adds a "+ Add to this Agent" button per card, which posts
//   { type: 'hermes-skill-pick', name, identifier, installCmd, source }
// to the parent window. We validate the origin and route the install through
// the standard hub action pipeline (background action + tailed log + Skills
// list invalidation), scoped to the Capabilities profile selector.
const HUB_ORIGIN = 'https://hermes-agent.nousresearch.com'
const HUB_PICKER_URL = `${HUB_ORIGIN}/docs/skills?embed=picker`

// Hub viewport height: persisted through the shared pane store (same one the
// terminal/editor panes use), dragged from the section's TOP edge — "pull the
// hub up" — clamped so neither the hub nor the skills list above vanishes.
const HUB_PANE_ID = 'capabilities-hub'
const HUB_DEFAULT_PX = 380
const HUB_MIN_PX = 120
const HUB_MAX_VH = 0.75

interface SkillPickMessage {
  identifier?: string
  installCmd?: string
  name?: string
  source?: string
  type?: string
}

interface EmbeddedHubPickerProps {
  /** Names of skills already installed in the scoped profile — a pick that
   *  matches is refused with a toast instead of re-running the install. */
  installedNames: ReadonlySet<string>
  /** Capabilities profile-scope override — installs land in THIS profile;
   *  undefined/null targets the app-wide active profile. */
  profile?: null | string
}

/** The Skills Hub browser for the Skills tab: a resizable iframe of the live
 *  hub where every card installs with one click. Expanded by default —
 *  discovery IS the point — with a collapse toggle and an update-all action. */
export function EmbeddedHubPicker({ installedNames, profile }: EmbeddedHubPickerProps) {
  const { t } = useI18n()
  const h = t.skills.hub
  const [open, setOpen] = useState(true)
  const updating = useStore($hubActions)[UPDATE_ALL_KEY]?.running ?? false
  const heightOverride = useStore($paneHeightOverride(HUB_PANE_ID))
  const height = heightOverride ?? HUB_DEFAULT_PX
  const [dragging, setDragging] = useState(false)

  // Top-edge sash: dragging UP grows the hub (shrinking the skills list above,
  // which is the flex-1 sibling). Same gesture as DetailPane / the shell's
  // bottom panes; double-click resets to the default height. The iframe gets
  // pointer-events disabled for the duration or it swallows the pointermoves.
  const startDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) {
      return
    }

    event.preventDefault()
    const startY = event.clientY
    const startHeight = height
    const max = Math.round(window.innerHeight * HUB_MAX_VH)
    setDragging(true)

    const onMove = (move: globalThis.PointerEvent) => {
      setPaneHeightOverride(
        HUB_PANE_ID,
        Math.round(Math.min(max, Math.max(HUB_MIN_PX, startHeight + (startY - move.clientY))))
      )
    }

    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      setDragging(false)
    }

    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp, { once: true })
  }

  // Picker messages from the embedded hub page. Origin-checked; installs route
  // through the same store pipeline the hub rows use, so the action log,
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

      // Already installed in this scope → tell the user, don't reinstall.
      if (installedNames.has(label) || installedNames.has(target)) {
        notify({ kind: 'success', title: h.alreadyInstalled(label), message: '' })

        return
      }

      notify({ kind: 'success', title: h.installStarted(label), message: h.actionLog })
      void installHubSkill(target, profile).catch(err => notifyError(err, h.actionFailed))
    }

    window.addEventListener('message', onMessage)

    return () => window.removeEventListener('message', onMessage)
  }, [h, installedNames, open, profile])

  const updateAll = () => {
    notify({ kind: 'success', title: h.updateStarted, message: h.actionLog })
    void updateHubSkills(profile).catch(err => notifyError(err, h.actionFailed))
  }

  return (
    <div className="relative border-t border-(--ui-stroke-secondary)">
      {/* Top-edge drag sash — pull the whole hub section up/down. */}
      <div
        className="group/hubsash absolute inset-x-0 top-0 z-10 h-1 -translate-y-1/2 cursor-row-resize"
        onDoubleClick={() => setPaneHeightOverride(HUB_PANE_ID, undefined)}
        onPointerDown={startDrag}
      >
        <div
          className={cn(
            'absolute inset-x-0 top-1/2 h-px -translate-y-1/2 transition-colors',
            dragging ? 'bg-(--ui-stroke-secondary)' : 'group-hover/hubsash:bg-(--ui-stroke-secondary)'
          )}
        />
      </div>
      <div className="flex items-center justify-between px-3 py-1.5">
        <span className="text-[0.7rem] font-medium text-(--ui-text-tertiary)">{h.pickerTitle}</span>
        <div className="flex items-center gap-1">
          <Button disabled={updating} onClick={updateAll} size="xs" variant="text">
            {updating && <Loader2 className="size-3 animate-spin" />}
            {updating ? h.updating : h.updateAll}
          </Button>
          <Button onClick={() => setOpen(v => !v)} size="xs" variant="text">
            {open ? h.pickerHide : h.pickerBrowse}
          </Button>
        </div>
      </div>
      {open && (
        <div className="grid gap-1 px-3 pb-2">
          {/* Resizable viewport: height comes from the top-edge drag sash
              above (persisted; double-click resets). The iframe is rendered
              oversized and scaled DOWN (133% × 0.75) so the hub page starts
              zoomed out — the cross-origin page itself can't be styled, but
              scaling the frame is ours. */}
          <div
            style={{
              border: '1px solid var(--ui-stroke-secondary)',
              borderRadius: 8,
              height,
              maxWidth: '100%',
              minHeight: HUB_MIN_PX,
              minWidth: 320,
              overflow: 'hidden',
              position: 'relative',
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
                // While the sash drags, the cross-origin iframe must not eat
                // the pointermove stream.
                pointerEvents: dragging ? 'none' : 'auto',
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

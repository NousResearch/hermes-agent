import { useStore } from '@nanostores/react'
import { memo, type PointerEvent as ReactPointerEvent, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { Loader2 } from '@/lib/icons'
import { useStoreSelector } from '@/lib/use-session-slice'
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
// Collapse threshold, mirroring DetailPane: a persisted height at/below this
// reads as "collapsed to the header" (the toggle stores 0).
const HUB_COLLAPSED_PX = 4
// Room the sash must always leave for the content ABOVE the picker (the
// installed-skills list plus its strip) so dragging the hub up can never
// crush the list to zero and shove its chrome under the hub header.
const HUB_LIST_RESERVED_PX = 176

interface SkillPickMessage {
  identifier?: string
  installCmd?: string
  name?: string
  source?: string
  type?: string
}

interface EmbeddedHubPickerProps {
  /** Kept mounted but fully hidden (display:none). The Capabilities view uses
   *  this to preserve the loaded hub iframe across tab switches — a plain
   *  unmount would reload the whole docs site on every return to Skills. */
  hidden?: boolean
  /** Names of skills already installed in the scoped profile — a pick that
   *  matches is refused with a toast instead of re-running the install. */
  installedNames: ReadonlySet<string>
  /** Capabilities profile-scope override — installs land in THIS profile;
   *  undefined/null targets the app-wide active profile. */
  profile?: ProfileScope
}

/** The Skills Hub browser for the Skills tab: a resizable iframe of the live
 *  hub where every card installs with one click. Expanded by default —
 *  discovery IS the point — with a collapse toggle (persisted, like every
 *  other pane) and an update-all action. Memoized: the iframe must not sit in
 *  the parent's keystroke/re-render path. */
export const EmbeddedHubPicker = memo(function EmbeddedHubPicker({
  hidden = false,
  installedNames,
  profile
}: EmbeddedHubPickerProps) {
  const { t } = useI18n()
  const h = t.skills.hub
  // Subscribe to the ONE flag this header renders, not the whole action map —
  // $hubActions churns on every tailed log line during an install.
  const updating = useStoreSelector($hubActions, actions => actions[UPDATE_ALL_KEY]?.running ?? false)
  // Collapse state rides the same persisted height override the sash writes
  // (0 = collapsed to the header), so "Hide the hub browser" survives tab
  // switches and restarts instead of re-expanding — and re-loading the docs
  // site — on every visit. Same contract as DetailPane.
  const heightOverride = useStore($paneHeightOverride(HUB_PANE_ID))
  const height = heightOverride ?? HUB_DEFAULT_PX
  const open = height > HUB_COLLAPSED_PX
  const [dragging, setDragging] = useState(false)
  const sectionRef = useRef<HTMLElement>(null)

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
    // Clamp against the actual Capabilities column, not just the window: the
    // hub may never grow past "column minus the list's reserved strip", so
    // the installed list always keeps real height and its header/footer can't
    // end up sharing pixels with the hub header.
    const column = sectionRef.current?.parentElement
    const columnMax = column ? column.clientHeight - HUB_LIST_RESERVED_PX : Number.POSITIVE_INFINITY
    const max = Math.max(HUB_MIN_PX, Math.round(Math.min(window.innerHeight * HUB_MAX_VH, columnMax)))
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

      if (!data || data.type !== 'hermes-skill-pick' || !data.name || !data.identifier) {
        return
      }

      const target = String(data.identifier)
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
    <section
      className={cn(
        // Shrinkable (no shrink-0) + overflow-hidden: the picker is a flex
        // child of the Capabilities column. Before, its fixed-height viewport
        // made the section's min-content height rigid, so a short window (or a
        // tall persisted drag height) starved the installed list to 0px and
        // the list's strip/footer painted straight over this header. Now the
        // section clips its own content and gives height back to the list;
        // min-h keeps the header row itself always visible.
        'relative flex min-h-9 flex-col overflow-hidden border-t border-(--ui-stroke-secondary)',
        hidden && 'hidden'
      )}
      ref={sectionRef}
    >
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
      <div className="flex shrink-0 items-center justify-between px-3 py-1.5">
        <span className="text-[0.7rem] font-medium text-(--ui-text-tertiary)">{h.pickerTitle}</span>
        <div className="flex items-center gap-1">
          <Button disabled={updating} onClick={updateAll} size="xs" variant="text">
            {updating && <Loader2 className="size-3 animate-spin" />}
            {updating ? h.updating : h.updateAll}
          </Button>
          <Button onClick={() => setPaneHeightOverride(HUB_PANE_ID, open ? 0 : undefined)} size="xs" variant="text">
            {open ? h.pickerHide : h.pickerBrowse}
          </Button>
        </div>
      </div>
      {open && (
        <div className="flex min-h-0 flex-col gap-1 px-3 pb-2">
          {/* Resizable viewport: height comes from the top-edge drag sash
              above (persisted; double-click resets). flex-basis instead of a
              hard height so a short window shrinks the hub viewport rather
              than letting it spill over the list. The iframe is rendered
              oversized and scaled DOWN (133% × 0.75) so the hub page starts
              zoomed out — the cross-origin page itself can't be styled, but
              scaling the frame is ours. */}
          <div
            style={{
              border: '1px solid var(--ui-stroke-secondary)',
              borderRadius: 8,
              flex: `0 1 ${height}px`,
              maxWidth: '100%',
              minHeight: 0,
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
          <p className="shrink-0 px-1 text-[0.65rem] leading-4 text-(--ui-text-quaternary)">{h.pickerHint}</p>
        </div>
      )}
    </section>
  )
})

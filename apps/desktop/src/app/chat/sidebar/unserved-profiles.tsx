import { type ComponentProps, useEffect, useState } from 'react'

import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { useI18n } from '@/i18n'
import { hostHas } from '@/lib/host-capabilities'
import { cn } from '@/lib/utils'
import { normalizeProfileKey } from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

const NONE: ReadonlySet<string> = new Set()

/**
 * Profiles the Webapp's gateway lists but does not serve, so picking one is refused. Its roster marks
 * them `reachable: false`, the flag the Bots pane greys a bot on. Re-read whenever the menu's profile
 * list is (every list read sets a new array: mount, focus, gateway open), so both come from the same
 * moment. Kept out of the fleet roster ($fleetRoster), whose agents feed fleet-only consumers a
 * one-gateway Webapp must not join. Native Desktop reads nothing: its rosters never carry the flag.
 */
export function useUnservedProfiles(profiles: readonly ProfileInfo[]): ReadonlySet<string> {
  const [unserved, setUnserved] = useState(NONE)

  useEffect(() => {
    const roster = hostHas('servedProfileRoster') ? window.hermesDesktop?.getAgentRoster : undefined

    if (!roster || !profiles.length) {
      return
    }

    let live = true

    // A failed read keeps the last marks, as the bridge keeps its last served set.
    void roster()
      .then(({ agents, primaryConnectionId }) => {
        if (live) {
          const own = agents.filter(agent => agent.connectionId === primaryConnectionId)
          setUnserved(
            new Set(own.filter(agent => agent.reachable === false).map(agent => normalizeProfileKey(agent.profile)))
          )
        }
      })
      .catch(() => undefined)

    return () => {
      live = false
    }
  }, [profiles])

  return unserved
}

/** A profile menu row's mark, greyed when the row's profile is not served here. Both pickers' rows
 *  use it with `UnservedProfileNote`, so an unserved row looks the same in each. */
export function ProfileRowGlyph({
  className,
  unserved,
  ...props
}: ComponentProps<typeof ProfileGlyph> & { unserved: boolean }) {
  return <ProfileGlyph aria-hidden="true" className={cn(unserved && 'grayscale', className)} {...props} />
}

/** What an unserved row says beside its name; the row itself is disabled and greyed. Its accessible
 *  name reads `<profile>, <note>` the way a row's status does (one text node, so the comma's space
 *  survives name computation). */
export function UnservedProfileNote() {
  const { t } = useI18n()
  const note = t.profiles.notServedHere

  return (
    <>
      <span aria-hidden="true" className="ml-auto shrink-0 pl-2 text-[0.6875rem] text-(--ui-text-tertiary)">
        {note}
      </span>
      <span className="sr-only">{`, ${note}`}</span>
    </>
  )
}

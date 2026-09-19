import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { resetBrowseState } from '@/store/composer-input-history'
import { $activeGatewayProfile, $profileColors, $profiles, $showAllProfiles, normalizeProfileKey } from '@/store/profile'

import { pickPlaceholder } from '../composer-utils'

interface UseComposerPlaceholderOptions {
  disabled: boolean
  reconnecting: boolean
  sessionId: null | string | undefined
}

export interface ComposerPlaceholderResult {
  /**
   * Plain text the contenteditable's `::before` paints when the editor is empty.
   * In the sidebar's "Show all profiles" mode with multiple profiles, it is
   * prefixed with `<profile-name> · ` so the active profile is visible before
   * the first message is sent. Single-profile users, or users scoped to a
   * single profile, get the original string unchanged — the profile is
   * already implicit from the sidebar tabs.
   */
  text: string
  /**
   * Profile color to paint the prefix region, or `null` when the prefix is
   * absent (single-profile users, the disabled/reconnecting/starting states,
   * profiles with no resolved color, or when the sidebar is not in
   * "Show all profiles" mode). Picked up by the editor element as
   * `--composer-placeholder-profile-color`.
   */
  profileColor: null | string
}

/**
 * The composer's placeholder text. A resting starter (new session) / continuation
 * (existing session) is picked once and only re-rolled when we genuinely move to
 * a *different* conversation — the null→id persist of a freshly-started session
 * keeps its starter so the text doesn't flip mid-stream. While the transport is
 * down, it swaps to a reconnecting / starting message instead.
 */
export function useComposerPlaceholder({
  disabled,
  reconnecting,
  sessionId
}: UseComposerPlaceholderOptions): ComposerPlaceholderResult {
  const { t } = useI18n()
  const profiles = useStore($profiles)
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const profileColors = useStore($profileColors)
  const showAllProfiles = useStore($showAllProfiles)
  const newSessionPlaceholders = t.composer.newSessionPlaceholders
  const followUpPlaceholders = t.composer.followUpPlaceholders

  const [restingPlaceholder, setRestingPlaceholder] = useState(() =>
    pickPlaceholder(sessionId ? followUpPlaceholders : newSessionPlaceholders)
  )

  const prevSessionIdRef = useRef(sessionId)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const prev = prevSessionIdRef.current
    prevSessionIdRef.current = sessionId

    if (prev === sessionId) {
      return
    }

    // null → id: the new session we're already in just got persisted. Keep the
    // starter we showed instead of swapping to a follow-up under the user.
    if (prev == null && sessionId) {
      return
    }

    resetBrowseState(prev)
    setRestingPlaceholder(pickPlaceholder(sessionId ? followUpPlaceholders : newSessionPlaceholders))
  }, [followUpPlaceholders, newSessionPlaceholders, sessionId])

  // When the transport is disabled it's because the gateway isn't open.
  // Distinguish a cold start ("Starting Hermes...") from a dropped connection
  // we're trying to restore. During reconnect, keep the textbox editable so a
  // flaky network doesn't block drafting; only submit/backend actions stay
  // disabled until the gateway is open again.
  const baseText = disabled
    ? reconnecting
      ? t.composer.placeholderReconnecting
      : t.composer.placeholderStarting
    : restingPlaceholder

  // Multi-profile hint: prepend the active profile name so the empty
  // welcome canvas shows which profile the typed message will land in. The
  // chat-header profile chip is suppressed until a session exists (#72767),
  // so this is the only signal before the first send.
  //
  // Narrowed to the sidebar's "Show all profiles" mode — when scope is a
  // single profile (the default), the profile is already implicit from the
  // sidebar tabs, so the prefix would just be noise. Only the unified
  // all-profiles view (where multiple sessions from different profiles share
  // the same sidebar) needs the prefix to disambiguate which one will
  // receive the typed message.
  return useMemo<ComposerPlaceholderResult>(() => {
    if (disabled || !showAllProfiles || profiles.length <= 1) {
      return { profileColor: null, text: baseText }
    }

    const profileKey = normalizeProfileKey(activeGatewayProfile)
    const profileName = profiles.find(profile => normalizeProfileKey(profile.name) === profileKey)?.name ?? profileKey

    return {
      profileColor: resolveProfileColor(profileKey, profileColors),
      text: `${profileName} · ${baseText}`
    }
  }, [activeGatewayProfile, baseText, disabled, profileColors, profiles, showAllProfiles])
}

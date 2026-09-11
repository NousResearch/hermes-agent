import { getSession } from '@/api/sessions'
import type { SlashExecResponse } from '@/app/types'
import { translateNow } from '@/i18n'
import { activeGateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  $sessions,
  sessionMatchesStoredId,
  setSessions
} from '@/store/session'
import { assertSessionOwnerResolved } from '@/store/session-owner-resolution'
import { $sessionStates, knownOwnerForSession } from '@/store/session-states'
import { requestForSessionProfile, type SessionOwnerScope } from '@/store/session-request-router'

const inFlightRetitles = new Set<string>()

export interface RetitleSessionOptions {
  /**
   * Optional caller assertion.
   *
   * The action itself is deliberately current-session-only. A sidebar context
   * menu may pass its row id here; if that row is not the currently selected
   * conversation the action fails closed instead of retitling another session.
   */
  sessionId?: string | null

  /**
   * Optional caller assertion for profile-scoped sidebar rows.
   *
   * Routing never trusts this value; the authoritative owner is snapshotted
   * from the live session owner graph below. This only prevents a stale /
   * foreign row action from being mistaken for the foreground session.
   */
  profile?: string | null
}

const normalizeProfile = (profile?: string | null): string => profile?.trim() || 'default'

function retitleKey(owner: SessionOwnerScope, runtimeSessionId: string): string {
  if (typeof owner === 'string') {
    return `profile:${normalizeProfile(owner)}:${runtimeSessionId}`
  }

  if (owner) {
    return `connection:${owner.connectionId}:${normalizeProfile(owner.profile)}:${runtimeSessionId}`
  }

  return `ambient:${runtimeSessionId}`
}

export async function runSessionRetitle({
  sessionId,
  profile
}: RetitleSessionOptions = {}): Promise<boolean> {
  // Snapshot every piece of target identity before the first await.
  //
  // The foreground session can change while /retitle is running. Nothing below
  // re-reads selection / active-session globals to decide where the request or
  // refresh belongs, so a mid-flight session switch cannot retarget the action.
  const storedSessionId = $selectedStoredSessionId.get()
  const runtimeSessionId = $activeSessionId.get()
  const sessions = $sessions.get()

  if (!storedSessionId || !runtimeSessionId) {
    notify({
      kind: 'warning',
      message: translateNow('sidebar.row.sessionActions') || 'No active session selected'
    })
    return false
  }

  const selectedRow = sessions.find(row => sessionMatchesStoredId(row, storedSessionId))
  const runtimeStoredSessionId = $sessionStates.get()[runtimeSessionId]?.storedSessionId

  // The selected stored id and active runtime id are normally published as one
  // foreground binding. If we caught a route/profile transition between those
  // publications, fail closed rather than issue /retitle against an ambiguous
  // pair.
  if (
    runtimeStoredSessionId &&
    (selectedRow
      ? !sessionMatchesStoredId(selectedRow, runtimeStoredSessionId)
      : runtimeStoredSessionId !== storedSessionId)
  ) {
    notify({
      kind: 'warning',
      message: 'The active session is changing. Try again.'
    })
    return false
  }

  // Context-menu callers may name the row they were opened on. Scheme A is
  // intentionally current-session-only: never reinterpret an inactive row as
  // a request to retitle that historical session.
  if (
    sessionId &&
    sessionId !== storedSessionId &&
    !(selectedRow && sessionMatchesStoredId(selectedRow, sessionId))
  ) {
    notify({
      kind: 'warning',
      message: 'Regenerate title is only available for the current session'
    })
    return false
  }

  // Snapshot the exact owner once. Session ownership is routing authority;
  // activeGateway() is only the ambient request transport and must never be
  // used to infer which profile / connection owns this runtime.
  const owner = knownOwnerForSession(runtimeSessionId)

  try {
    assertSessionOwnerResolved(owner, {
      method: 'slash.exec',
      sessionId: runtimeSessionId
    })
  } catch (error) {
    notifyError(error, translateNow('sidebar.row.regenerateTitleFailed'))
    return false
  }

  // A sidebar row may additionally assert its profile. Do not use this for
  // routing — it is only a stale-row guard.
  if (profile) {
    const ownerProfile =
      typeof owner === 'string'
        ? owner
        : owner?.profile

    if (ownerProfile && normalizeProfile(ownerProfile) !== normalizeProfile(profile)) {
      notify({
        kind: 'warning',
        message: 'Regenerate title is only available for the current session'
      })
      return false
    }
  }

  const gateway = activeGateway()

  if (!gateway) {
    notifyError(new Error('Gateway not connected'), translateNow('sidebar.row.regenerateTitleFailed'))
    return false
  }

  // Snapshot the exact row/id used for the post-command authoritative refresh.
  // A projected lineage row may have a newer concrete id than the route-facing
  // stored id, so prefer the row's current id when available.
  const refreshSessionId = selectedRow?.id ?? storedSessionId
  const previousTitle = selectedRow?.title ?? null

  // REST uses an explicit connection/profile scope as well. targetProfile is
  // the backend-facing profile when an owner route aliases the Desktop profile.
  const refreshScope =
    typeof owner === 'string'
      ? owner
      : owner
        ? {
            connectionId: owner.connectionId,
            profile: owner.targetProfile ?? owner.profile
          }
        : selectedRow
          ? {
              connectionId: selectedRow.connection_id,
              profile: selectedRow.profile
            }
          : profile

  const key = retitleKey(owner, runtimeSessionId)

  if (inFlightRetitles.has(key)) {
    return false
  }

  inFlightRetitles.add(key)

  notify({
    kind: 'info',
    message: translateNow('sidebar.row.regeneratingTitle')
  })

  try {
    const ambientRequest = gateway.request.bind(gateway) as typeof gateway.request

    // Canonical backend path only.
    //
    // #96243 registers /retitle as a normal gateway slash command. Desktop's
    // established slash-command transport is slash.exec with the leading slash
    // removed. There is intentionally no plugin command, speculative RPC, or
    // empty-title REST fallback here.
    const result = await requestForSessionProfile<SlashExecResponse>(
      owner,
      ambientRequest,
      'slash.exec',
      {
        session_id: runtimeSessionId,
        command: 'retitle'
      }
    )

    // /retitle owns generation, provenance and persistence policy. Re-read the
    // row from its owning backend instead of parsing human-readable slash output
    // to discover what title was written.
    const refreshed = await getSession(refreshSessionId, refreshScope)

    // Publish the authoritative title safely without clobbering racing user renames.
    // Preserves reference identity on no-op so React avoids unnecessary re-renders.
    setSessions(current => {
      let changed = false
      const next = current.map(row => {
        if (!sessionMatchesStoredId(row, storedSessionId) && row.id !== refreshed.id) {
          return row
        }

        // Another title mutation (e.g. manual rename) won while our refresh was in flight
        if (row.title !== previousTitle && row.title !== refreshed.title) {
          return row
        }

        if (row.title === refreshed.title) {
          return row
        }

        changed = true
        return { ...row, title: refreshed.title }
      })

      return changed ? next : current
    })

    const output = result.output?.trim()
    const warning = result.warning?.trim()
    const titleChanged = refreshed.title !== previousTitle

    if (warning) {
      notify({
        kind: 'warning',
        message: output ? `${warning}\n${output}` : warning
      })
    } else {
      notify({
        kind: titleChanged ? 'success' : 'info',
        message:
          output ||
          (titleChanged
            ? translateNow('sidebar.row.regenerateTitleSuccess')
            : 'Session title unchanged')
      })
    }

    return true
  } catch (error) {
    notifyError(error, translateNow('sidebar.row.regenerateTitleFailed'))
    return false
  } finally {
    inFlightRetitles.delete(key)
  }
}

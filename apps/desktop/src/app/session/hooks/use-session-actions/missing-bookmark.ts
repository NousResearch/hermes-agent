import { routeSessionId, sessionRoute } from '@/app/routes'
import { $diskPluginsScanPending } from '@/contrib/plugins-store'
import { $gatewaySwitching } from '@/store/gateway-switch'
import { $activeGatewayProfile, $gatewaySwapTarget, $showAllProfiles, normalizeProfileKey } from '@/store/profile'
import type { SessionOwnerRoute } from '@/store/session-request-router'

import { sessionCreatedThisRun } from './created-this-run'
import { ambientResumeConnectionId } from './resume-owner'
import { goneSessionVerdict, isSessionGoneError } from './resume-verdicts'
import { cachedSessionRow, type StoredSessionProbe } from './utils'

interface BookmarkResumeContext {
  hadWarmRuntime: boolean
  initialConnectionId: string
  ownerRoute: SessionOwnerRoute | undefined
  routedSessionId: string | null
  wasSelected: boolean
}

export function canDismissMissingBookmark(
  storedSessionId: string,
  status: StoredSessionProbe['status'],
  { hadWarmRuntime, initialConnectionId, ownerRoute, routedSessionId, wasSelected }: BookmarkResumeContext
): boolean {
  // A plugin can claim the path while the metadata request is in flight.
  return (
    status === 'gone' &&
    storedSessionId === routedSessionId &&
    !$diskPluginsScanPending.get() &&
    routeSessionId(sessionRoute(storedSessionId)) === storedSessionId &&
    !ownerRoute &&
    !hadWarmRuntime &&
    !wasSelected &&
    !$showAllProfiles.get() &&
    initialConnectionId === ambientResumeConnectionId() &&
    missingResumeVerdict(storedSessionId, Boolean(cachedSessionRow(storedSessionId))) === 'draft'
  )
}

export function isMissingSessionWithoutTranscript(messageCount: number, error: unknown): boolean {
  return messageCount === 0 && isSessionGoneError(error)
}

export function missingResumeVerdict(storedSessionId: string, stillListed: boolean, ownerProfile?: string) {
  return goneSessionVerdict({
    createdThisRun: sessionCreatedThisRun(storedSessionId),
    stillListed,
    // A 404 on the wrong backend is inconclusive during a profile switch (#88540).
    switchInFlight:
      $gatewaySwitching.get() ||
      Boolean($gatewaySwapTarget.get()) ||
      Boolean(
        ownerProfile?.trim() && normalizeProfileKey(ownerProfile) !== normalizeProfileKey($activeGatewayProfile.get())
      )
  })
}

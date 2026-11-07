import { backendScopeKey } from '@hermes/shared'
import { atom } from 'nanostores'

import { fetchVoiceLiveStatus, type VoiceChatMode, type VoiceLiveStatus } from '@/lib/voice-live'
import { fetchVoiceLiveGrokStatus } from '@/lib/voice-live-grok'
import { requestGatewayForActiveRoute } from '@/store/active-route-request'
import { activeGatewayConnectionId, activeGatewayProfileKey } from '@/store/gateway'

/**
 * `voice.voice_chat_mode` as the backend resolves it, plus whether GPT-Live can
 * actually start (an OpenAI key resolves on the gateway host). The composer
 * mounts the chained, GPT-Live or Grok-Live conversation engine from this;
 * refreshed with the config snapshot so a Settings change applies to the next
 * conversation.
 *
 * Grok-Live gets its OWN status atom (`$voiceLiveGrokStatus`) rather than a
 * widened `VoiceLiveStatus` on this one — SPEC §11's deliberate choice to add
 * a parallel status route instead of retrofitting the existing gpt-live-only
 * endpoint mid-migration. Both atoms report the SAME `voice.voice_chat_mode`
 * value (the shared backend resolver), so `selectedVoiceChatMode` reads
 * whichever one answered.
 */
export const $voiceLiveStatus = atom<null | VoiceLiveStatus>(null)
export const $voiceLiveGrokStatus = atom<null | VoiceLiveStatus>(null)

// Bumped by every mode write: a fetch issued before the write reads the old
// mode, so it must neither be joined nor allowed to publish afterwards.
let writeEpoch = 0

let inflight: null | { key: string; promise: Promise<null | VoiceLiveStatus> } = null
let grokInflight: null | { key: string; promise: Promise<null | VoiceLiveStatus> } = null

// The single-flight slots are shared only within one (connection, profile,
// write) scope; a switch or a write strands the older fetch instead of joining it.
function voiceLiveScopeKey(): string {
  return `${backendScopeKey(activeGatewayConnectionId(), activeGatewayProfileKey())}::${writeEpoch}`
}

export async function refreshVoiceLiveStatus(): Promise<null | VoiceLiveStatus> {
  const key = voiceLiveScopeKey()

  if (inflight?.key === key) {
    return inflight.promise
  }

  const flight = {
    key,
    promise: fetchVoiceLiveStatus()
      .then(status => {
        // A late answer for a previous scope still resolves for its caller but
        // must not overwrite the status the current profile/connection reads.
        if (voiceLiveScopeKey() === key) {
          $voiceLiveStatus.set(status)
        }

        return status
      })
      .finally(() => {
        if (inflight === flight) {
          inflight = null
        }
      })
  }

  inflight = flight

  return flight.promise
}

export async function refreshVoiceLiveGrokStatus(): Promise<null | VoiceLiveStatus> {
  const key = voiceLiveScopeKey()

  if (grokInflight?.key === key) {
    return grokInflight.promise
  }

  const flight = {
    key,
    promise: fetchVoiceLiveGrokStatus()
      .then(status => {
        if (voiceLiveScopeKey() === key) {
          $voiceLiveGrokStatus.set(status)
        }

        return status
      })
      .finally(() => {
        if (grokInflight === flight) {
          grokInflight = null
        }
      })
  }

  grokInflight = flight

  return flight.promise
}

/** Refresh both status sources from the same call sites (config-change handler,
 *  conversation-mount effect) so they never drift out of sync (SPEC §13 risk 7). */
export async function refreshAllVoiceLiveStatuses(): Promise<void> {
  await Promise.all([refreshVoiceLiveStatus(), refreshVoiceLiveGrokStatus()])
}

/** Selected mode. `chained` until a status atom answers, or when the backend predates the mode.
 *  Default (no argument) consults BOTH status atoms: the gpt-live endpoint can lag or stay
 *  null while grok-live is the configured engine — reading only `$voiceLiveStatus` then
 *  silently mounts chained STT instead of duplex. */
export function selectedVoiceChatMode(status?: null | VoiceLiveStatus): VoiceChatMode {
  if (status !== undefined) {
    return status?.mode === 'gpt-live' || status?.mode === 'grok-live' ? status.mode : 'chained'
  }

  for (const candidate of [$voiceLiveStatus.get(), $voiceLiveGrokStatus.get()]) {
    if (candidate?.mode === 'gpt-live' || candidate?.mode === 'grok-live') {
      return candidate.mode
    }
  }

  return 'chained'
}

/**
 * Persist `voice.voice_chat_mode` on the live gateway (whichever profile/host
 * the app is talking to) and re-read the resolved status, so the menu shows
 * what the backend will actually mount next. Takes effect on the NEXT
 * conversation; an active one keeps its engine.
 *
 * Routed by (connection, profile), not by socket: `config.set` is
 * `@_profile_scoped`, so on the shared-primary route an unscoped write lands in
 * the LAUNCH profile's config.yaml, not the viewed one (#125969 class); and a
 * bare profile name would resolve through this machine's pool, missing a
 * profile served only by a remote gateway.
 */
export async function setVoiceChatMode(mode: VoiceChatMode): Promise<null | VoiceLiveStatus> {
  // The profile rides in the params too: on a backend shared by several
  // profiles an unscoped write edits its launch profile.
  const profile = activeGatewayProfileKey()
  await requestGatewayForActiveRoute(profile, 'config.set', { key: 'voice.voice_chat_mode', profile, value: mode })

  writeEpoch += 1
  await refreshAllVoiceLiveStatuses()

  return $voiceLiveStatus.get()
}

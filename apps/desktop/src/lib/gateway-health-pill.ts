/**
 * Status-bar "backend health" pill. The serve/chat socket and the messaging
 * gateway are different processes; this derivation is the one place that
 * names them so the painted label cannot be read as "Discord is up".
 */

import { runtimeReadinessDisplay, type RuntimeReadinessResult } from '@/lib/runtime-readiness'

export interface GatewayHealthPillCopy {
  backend: string
  checking: string
  connecting: string
  messagingDegraded: (name: string) => string
  messagingStopped: string
  needsSetup: string
  offline: string
  ready: string
  restarting: string
  unavailable: string
}

export interface GatewayHealthPlatform {
  state: string
}

export interface GatewayHealthPillInput {
  connectionState: string
  copy: GatewayHealthPillCopy
  inferenceStatus: RuntimeReadinessResult | null
  /** Backend's "is any messaging platform configured?" verdict. Authoritative over the
   *  state/platforms inference below: a stopped gateway projects an empty platform map, so
   *  without it a never-configured install reads every plain stop as a red alarm. */
  messagingConfigured?: boolean | null
  messagingRunning?: boolean
  messagingState?: null | string
  platforms?: null | Record<string, GatewayHealthPlatform>
  restarting?: boolean
}

export interface GatewayHealthPill {
  degraded: boolean
  detail: string
  label: string
  title?: string
}

const HEALTHY_PLATFORM_STATES = new Set(['connected'])
const IGNORED_PLATFORM_STATES = new Set(['disabled', 'not_configured', 'not configured'])

function platformLeaf(id: string): string {
  const sep = id.lastIndexOf(':')

  return sep >= 0 ? id.slice(sep + 1) : id
}

function unhealthyPlatformNames(platforms: GatewayHealthPillInput['platforms']): string[] {
  return Object.entries(platforms || {})
    .filter(([, platform]) => {
      const state = (platform?.state || '').trim().toLowerCase()

      return Boolean(state) && !HEALTHY_PLATFORM_STATES.has(state) && !IGNORED_PLATFORM_STATES.has(state)
    })
    .map(([id]) => platformLeaf(id))
}

function messagingWasConfigured(
  messagingState: GatewayHealthPillInput['messagingState'],
  platforms: GatewayHealthPillInput['platforms'],
  messagingConfigured?: boolean | null
): boolean {
  // The backend knows whether any platform is configured even while the gateway is down;
  // trust it when it answers. A plain "stopped" with an empty map means there is no bot to be
  // missing, and an alarm for it is a false alarm (a Desktop install with no messaging reads
  // "stopped" forever and the pill stays red after every boot). Only the plain stop is
  // suppressed: "startup_failed" keeps alarming through its own branch, and every other state
  // keeps its pre-verdict behavior exactly. A non-empty map always wins too — merged
  // per-profile entries carry failures the local config set cannot see.
  if (messagingConfigured === false
      && (messagingState || '').trim().toLowerCase() === 'stopped'
      && Object.keys(platforms || {}).length === 0) {
    return false
  }

  const state = (messagingState || '').trim().toLowerCase()

  if (state === 'stopped' || state === 'startup_failed') {
    return true
  }

  return Object.keys(platforms || {}).length > 0
}

function messagingDetail(copy: GatewayHealthPillCopy, names: string[]): string {
  return names.length > 0 ? copy.messagingDegraded(names[0]!) : copy.messagingStopped
}

function inferenceDetail(copy: GatewayHealthPillCopy, inferenceStatus: RuntimeReadinessResult | null): string {
  return {
    checking: copy.checking,
    needs_setup: copy.needsSetup,
    ready: copy.ready,
    unavailable: copy.unavailable
  }[runtimeReadinessDisplay(inferenceStatus)]
}

export function statusBarGatewayHealth({
  connectionState,
  copy,
  inferenceStatus,
  messagingConfigured,
  messagingRunning,
  messagingState,
  platforms,
  restarting = false
}: GatewayHealthPillInput): GatewayHealthPill {
  const connectionOpen = connectionState === 'open'
  const connectionConnecting = connectionState === 'connecting'
  const downNames = unhealthyPlatformNames(platforms)

  const messagingDown =
    messagingWasConfigured(messagingState, platforms, messagingConfigured) &&
    (messagingRunning === false || downNames.length > 0)

  const connectionDetail = connectionOpen
    ? inferenceDetail(copy, inferenceStatus)
    : connectionConnecting
      ? copy.connecting
      : copy.offline

  const detail = restarting
    ? copy.restarting
    : messagingDown && connectionOpen
      ? messagingDetail(copy, downNames)
      : connectionDetail

  return {
    degraded: Boolean(messagingDown && connectionOpen),
    detail,
    label: copy.backend,
    title: messagingDown && connectionOpen ? messagingDetail(copy, downNames) : undefined
  }
}

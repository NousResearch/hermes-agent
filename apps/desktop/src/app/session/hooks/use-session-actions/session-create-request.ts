import { requestGatewayForAgent } from '@/store/gateway'
import {
  $activeGatewayProfile,
  $newChatProfile,
  type AgentProfileRoute,
  ensureGatewayAgent,
  ensureGatewayProfile,
  normalizeProfileKey,
  resolveNewChatOwnerRoute
} from '@/store/profile'
import {
  $currentCwdExplicit,
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $currentServiceTier,
  getCurrentModelSource
} from '@/store/session'
import type { SessionCreateResponse } from '@/types/hermes'

type RequestGateway = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

/** A backend predating a `session.create` field rejects the whole create at
 *  admission (`tui_gateway/contracts/registry.py::validate_params`, code 4000,
 *  handler never runs) — e.g. a Hermes Cloud backend behind a Desktop that
 *  updates from main (#128971). Each field below is safe to drop for them:
 *  - `cwd_explicit` (#122899): those backends always honoured the client `cwd`.
 *  - `service_tier` (Ultrafast): `fast` still rides, so they get Priority.
 *  Matched on the stable prefix, not `isOutOfSyncRpcParams`: v0.21.3 already
 *  rejects but predates the "out of sync" suffix.
 *  Delete a field once no supported backend predates it. */
const DROPPABLE_CREATE_FIELDS = ['cwd_explicit', 'service_tier'] as const

function rejectedField(params: Record<string, unknown>, error: unknown): string | undefined {
  const message = error instanceof Error ? error.message : String(error)

  return DROPPABLE_CREATE_FIELDS.find(
    field => field in params && message.includes(`invalid params for session.create: ${field}:`)
  )
}

/** `session.create` on the captured owner route (or the window's gateway). */
export async function createGatewaySession(
  route: AgentProfileRoute | null,
  params: Record<string, unknown>,
  requestGateway: RequestGateway
): Promise<SessionCreateResponse> {
  const send = (requestParams: Record<string, unknown>) =>
    route
      ? requestGatewayForAgent<SessionCreateResponse>(
          route.connectionId,
          route.profile,
          'session.create',
          requestParams,
          undefined,
          undefined,
          { spawnPriority: 'foreground' }
        )
      : requestGateway<SessionCreateResponse>('session.create', requestParams)

  // One resend per dropped field: a backend predating both rejects them one at a time.
  const create = async (requestParams: Record<string, unknown>): Promise<SessionCreateResponse> => {
    try {
      return await send(requestParams)
    } catch (error) {
      const field = rejectedField(requestParams, error)

      if (!field) {
        throw error
      }

      const { [field]: _dropped, ...compatible } = requestParams

      return create(compatible)
    }
  }

  return create(params)
}

// `session.create` params from the current profile + sticky-UI model/effort/fast,
// ensuring the gateway is on that profile first. Shared by the primary send path
// and the "open in split" tile path; `cwd` is the one thing that differs (the
// live composer cwd for a send, the resolved new-session cwd for a fresh tile).
//
// Resolving null profile to the active gateway's is load-bearing: in global-remote
// mode one backend serves every profile, so an omitted profile silently lands the
// chat on the launch (default) profile — the "rubberbands back to default" bug.
// A no-op for single-profile/local-pooled users (a backend resolves its own launch
// profile to None). Effort/fast still ride as per-session overrides. Model and
// provider only ride when the composer source is 'manual' — a default-sourced
// value is a mirror of Settings → Model and must not pin the new chat.

// Exact `service_tier` words session.create pins (bounded policies +
// Ultrafast, #132275); Priority rides as `fast` only (a pre-Ultrafast backend
// rejects the field — createGatewaySession drops it).
const EXPLICIT_CREATE_TIERS: ReadonlySet<string> = new Set(['ultrafast', 'auto', 'cold'])

export async function desktopSessionCreateParams(
  cwd: string,
  capturedRoute = resolveNewChatOwnerRoute(),
  requestedProfile?: string,
  legacyProfileIntent = false,
  includeComposerSelection = true
): Promise<Record<string, unknown>> {
  // Treat Send as the linearization point for the visible selector state. The
  // profile handshake below can yield long enough for background config/model
  // refreshes to finish; reading atoms afterward would silently create the
  // session with a different selection than the one the user submitted.
  // Settings → Model while a session is live leaves $currentModel painted with
  // the live agent (applySavedMainModel) and only flips the source to 'default'.
  // Shipping that stale value as an override pins every new chat to the old
  // model. Omit model/provider unless the source is 'manual'.
  const isManualSelection = getCurrentModelSource() === 'manual'

  const selection = {
    effort: $currentReasoningEffort.get().trim(),
    fast: $currentFastMode.get(),
    serviceTier: $currentServiceTier.get().trim(),
    model: isManualSelection ? $currentModel.get().trim() : '',
    provider: isManualSelection ? $currentProvider.get().trim() : ''
  }

  const profile =
    capturedRoute?.profile ||
    requestedProfile ||
    $newChatProfile.get() ||
    normalizeProfileKey($activeGatewayProfile.get())

  if (capturedRoute) {
    await ensureGatewayAgent(capturedRoute.connectionId, profile)
  } else if (legacyProfileIntent) {
    await ensureGatewayProfile(profile, { forceLegacyRoute: true })
  } else {
    await ensureGatewayProfile(profile)
  }

  return {
    cols: 96,
    source: 'desktop',
    ...(cwd && { cwd }),
    // #52589: explicit provenance for the shipped cwd — an inherited app-global
    // workspace must not override the target profile's configured terminal.cwd.
    ...(cwd && { cwd_explicit: $currentCwdExplicit.get() }),
    ...(profile ? { profile: capturedRoute?.targetProfile || profile } : {}),
    ...(includeComposerSelection
      ? {
          ...(selection.model
            ? { model: selection.model, ...(selection.provider ? { provider: selection.provider } : {}) }
            : {}),
          ...(selection.effort ? { reasoning_effort: selection.effort } : {}),
          fast: selection.fast,
          // The bounded policies (auto/cold) and Ultrafast need the exact tier:
          // `fast` alone would pin Priority and destroy the policy
          // (`create_overrides` parses these via parse_exact_service_tier).
          // Priority rides as `fast` only: a pre-Ultrafast backend rejects the
          // field (createGatewaySession drops it).
          ...(EXPLICIT_CREATE_TIERS.has(selection.serviceTier ?? '')
            ? { service_tier: selection.serviceTier }
            : {})
        }
      : {})
  }
}

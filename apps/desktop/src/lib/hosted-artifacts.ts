import {
  hostedArtifactFrameOrigin,
  type HostedArtifactFramePolicy,
  isHostedArtifactId
} from '../../electron/hosted-artifact-frame'

export type { HostedArtifactFramePolicy }

/**
 * Server-hosted artifacts as they reach the transcript: the result text of
 * the artifacts MCP server's tools. The desktop has no MCP connection of its
 * own; the tool result is the whole contract.
 *
 * create / update / show answer `{id, version, viewer_url, content_url}`
 * (show adds `title`). `content_url` carries a short-lived view ticket for the
 * artifact's own origin; `viewer_url` is the durable page that always works.
 */

/** Registry prefix of the artifacts server's tools (`mcp__<server>__<tool>`). */
export const HOSTED_ARTIFACT_TOOL_PREFIX = 'mcp__hermes_artifacts__'

/** The tools whose result names one renderable artifact version. */
const RENDERED_TOOLS = new Set(['artifact_create', 'artifact_update', 'artifact_show'])

/** A view ticket's lifetime. A result older than this cannot open its frame
 *  any more (the content host answers its not-found page), so it links. */
export const HOSTED_ARTIFACT_TICKET_TTL_SECONDS = 300

export function isHostedArtifactCardTool(toolName: string | undefined): boolean {
  return (
    typeof toolName === 'string' &&
    toolName.startsWith(HOSTED_ARTIFACT_TOOL_PREFIX) &&
    RENDERED_TOOLS.has(toolName.slice(HOSTED_ARTIFACT_TOOL_PREFIX.length))
  )
}

export interface HostedArtifactResult {
  contentUrl: string
  id: string
  title: string
  version: number
  viewerUrl: string
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

function parseJsonObject(value: unknown): Record<string, unknown> | null {
  if (isRecord(value)) {
    return value
  }

  if (typeof value !== 'string' || !value.trim()) {
    return null
  }

  try {
    const parsed: unknown = JSON.parse(value)

    return isRecord(parsed) ? parsed : null
  } catch {
    return null
  }
}

function asWebUrl(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null
  }

  try {
    const url = new URL(value.trim())

    return url.protocol === 'https:' || url.protocol === 'http:' ? url.toString() : null
  } catch {
    return null
  }
}

function fromPayload(payload: Record<string, unknown>): HostedArtifactResult | null {
  const { id, version } = payload
  const viewerUrl = asWebUrl(payload.viewer_url)

  if (
    !isHostedArtifactId(id) ||
    typeof version !== 'number' ||
    !Number.isInteger(version) ||
    version < 1 ||
    !viewerUrl
  ) {
    return null
  }

  return {
    // Validated for framing separately; kept verbatim so the ticket survives.
    contentUrl: typeof payload.content_url === 'string' ? payload.content_url.trim() : '',
    id,
    title: typeof payload.title === 'string' ? payload.title.trim() : '',
    version,
    viewerUrl
  }
}

/**
 * The artifact version a tool result names, or null for errors and anything
 * else. Accepts the bare JSON text, the MCP handler's `{result: "<json>"}`
 * envelope, and its `structuredContent` copy.
 */
export function parseHostedArtifactResult(result: unknown): HostedArtifactResult | null {
  const outer = parseJsonObject(result)

  if (!outer || 'error' in outer) {
    return null
  }

  for (const candidate of [outer.structuredContent, outer.result, outer]) {
    const payload = parseJsonObject(candidate)

    if (payload && !('error' in payload)) {
      const parsed = fromPayload(payload)

      if (parsed) {
        return parsed
      }
    }
  }

  return null
}

/** This build's framing policy: development builds also frame the local
 *  artifacts server. */
export function hostedArtifactFramePolicy(): HostedArtifactFramePolicy {
  return { dev: Boolean(import.meta.env.DEV) }
}

/**
 * The origin to frame for this result, or null to show the link card. Null
 * when the content URL is not on this artifact's own origin under an allowed
 * frame domain, or when its ticket has already expired.
 */
export function hostedArtifactFrameFor(
  artifact: HostedArtifactResult,
  {
    completedAt,
    nowSeconds = Date.now() / 1000,
    policy = hostedArtifactFramePolicy()
  }: { completedAt?: number; nowSeconds?: number; policy?: HostedArtifactFramePolicy } = {}
): string | null {
  if (
    typeof completedAt === 'number' &&
    Number.isFinite(completedAt) &&
    completedAt > 0 &&
    nowSeconds - completedAt >= HOSTED_ARTIFACT_TICKET_TTL_SECONDS
  ) {
    return null
  }

  return hostedArtifactFrameOrigin(artifact.contentUrl, policy, artifact.id)
}

/** Longest text a view may submit as one user message. Rejected, never cut. */
export const MAX_UI_MESSAGE_LENGTH = 4000
/** One message per frame per second: a person clicks, a loop floods. */
export const UI_MESSAGE_THROTTLE_MS = 1000

export type UiMessageDecision =
  { kind: 'reject'; reason: 'invalid' | 'throttled' | 'too_long' } | { kind: 'submit'; text: string }

/**
 * Map a view's `ui/message` request to the prompt the composer submits.
 * Only `role: "user"` text blocks count; anything else (images, resources)
 * is not something a typed message can carry, so a message without text is
 * refused rather than half-sent.
 */
export function uiMessageToPrompt(params: unknown, nowMs: number, lastAcceptedMs: number): UiMessageDecision {
  if (!isRecord(params) || params.role !== 'user' || !Array.isArray(params.content)) {
    return { kind: 'reject', reason: 'invalid' }
  }

  const texts = params.content
    .filter(
      (block): block is { text: string; type: 'text' } =>
        isRecord(block) && block.type === 'text' && typeof block.text === 'string'
    )
    .map(block => block.text.trim())
    .filter(Boolean)

  if (texts.length === 0 || texts.length !== params.content.length) {
    return { kind: 'reject', reason: 'invalid' }
  }

  const text = texts.join('\n\n')

  if (text.length > MAX_UI_MESSAGE_LENGTH) {
    return { kind: 'reject', reason: 'too_long' }
  }

  if (nowMs - lastAcceptedMs < UI_MESSAGE_THROTTLE_MS) {
    return { kind: 'reject', reason: 'throttled' }
  }

  return { kind: 'submit', text }
}

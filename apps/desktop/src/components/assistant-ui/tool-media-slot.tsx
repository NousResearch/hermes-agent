import { useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type FC, type ReactNode, useEffect, useMemo } from 'react'

import { useComposerScope } from '@/app/chat/composer/scope'
import { useSessionView } from '@/app/chat/session-view'
import { ErrorBoundary } from '@/components/error-boundary'
import { type Contribution, useContributions } from '@/contrib'
import { $pluginRecords } from '@/contrib/plugins-store'
import { mediaKind, mediaMime, mediaName, resolveMediaDisplaySrc, resolveMediaPlaybackSrc } from '@/lib/media'
import {
  claimPatternMatches,
  effectiveToolCall,
  findProducingToolCall,
  refusedClaimPatterns,
  resolveToolMediaClaim,
  TOOL_MEDIA_RENDERER_AREA,
  type ToolCallLike,
  type ToolMediaFile,
  type ToolMediaRendererContribution,
  type ToolMediaRendererProps
} from '@/lib/tool-media-renderers'
import { $gateway } from '@/store/gateway'
import { $connection } from '@/store/session'
import {
  $toolMediaOwners,
  ensureToolMediaOwners,
  packageOfContributionSource,
  revalidateToolMediaOwners,
  toolMediaOwnerKey
} from '@/store/tool-media-owners'

/**
 * Where a `MEDIA:` attachment meets the tool media renderer area. With no
 * renderer registered this is a pass-through (no subscriptions); otherwise it
 * finds the tool call in this turn that produced the path and, when the plugin
 * that OWNS that tool claims it, mounts the plugin's component with the
 * built-in player as both its `fallback` prop and its error fallback.
 */

const warned = new Set<string>()

function warnOnce(key: string, message: string) {
  if (!warned.has(key)) {
    warned.add(key)
    console.warn(message)
  }
}

function warnRefused(claim: Pick<Contribution, 'id' | 'source'>, tool: string) {
  warnOnce(
    `${claim.id}\u0000${tool}`,
    `[${claim.source ?? 'core'}] tool media renderer refused for "${tool}": it is not a tool this plugin's package registered`
  )
}

// The producer, read straight from the thread store so the selector returns
// the store's own part object (a stable reference across deltas). Outside a
// message (a Bot group room, a preview) there is no transcript: no producer.
function useProducingToolCall(path: string): ToolCallLike | null {
  return useAuiState(state => {
    try {
      return findProducingToolCall(state.thread.messages, state.message.id, path)
    } catch {
      return null
    }
  })
}

export const ToolMediaSlot: FC<{ children: ReactNode; path: string }> = ({ children, path }) => {
  const claims = useContributions(TOOL_MEDIA_RENDERER_AREA)

  if (claims.length === 0) {
    return <>{children}</>
  }

  return (
    <ClaimedToolMediaSlot claims={claims} path={path}>
      {children}
    </ClaimedToolMediaSlot>
  )
}

const ClaimedToolMediaSlot: FC<{ children: ReactNode; claims: readonly Contribution[]; path: string }> = ({
  children,
  claims,
  path
}) => {
  const part = useProducingToolCall(path)
  const producer = useMemo(() => (part ? effectiveToolCall(part) : null), [part])
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const runtimeId = useStore(view.$runtimeId)
  const sessionId = storedId || runtimeId || null
  const entries = useStore($toolMediaOwners)
  // A transcript painted before the socket is up fetches once it is.
  const gateway = useStore($gateway)
  const entry = entries[toolMediaOwnerKey(sessionId)]
  // Re-resolve when a plugin record (enable state, package) changes.
  useStore($pluginRecords)

  useEffect(() => {
    if (producer) {
      ensureToolMediaOwners(sessionId)
    }
  }, [producer, sessionId, entry, gateway])

  const ready = entry?.status === 'ready' ? entry.owners : null

  useEffect(() => {
    if (!ready) {
      return
    }

    for (const { claim, pattern } of refusedClaimPatterns(claims, ready, packageOfContributionSource)) {
      warnRefused(claim, pattern)
    }
  }, [claims, ready])

  const claim =
    producer?.toolName && ready
      ? resolveToolMediaClaim(claims, producer.toolName, {
          onRefused: warnRefused,
          owners: ready,
          packageOf: packageOfContributionSource
        })
      : null

  // A claimed tool nobody owns yet may belong to an MCP server that has not
  // connected: re-read ownership on a bounded backoff, keeping this answer.
  const unowned = Boolean(ready && producer?.toolName && !claim && claimsTool(claims, producer.toolName))

  useEffect(() => {
    if (unowned) {
      revalidateToolMediaOwners(sessionId)
    }
  }, [unowned, sessionId, entry])

  if (!claim || !producer?.toolName) {
    return <>{children}</>
  }

  return (
    <ErrorBoundary fallback={() => children} label={`contrib:${claim.id}`}>
      <ToolMediaLeaf contribution={claim} fallback={children} path={path} producer={producer} sessionId={sessionId} />
    </ErrorBoundary>
  )
}

const ToolMediaLeaf: FC<{
  contribution: Contribution
  fallback: ReactNode
  path: string
  producer: ToolCallLike
  sessionId: null | string
}> = ({ contribution, fallback, path, producer, sessionId }) => {
  const render = (contribution.data as ToolMediaRendererContribution).render
  const owner = useMediaOwner()

  const file = useMemo<ToolMediaFile>(
    () => ({
      kind: mediaKind(path),
      mime: mediaMime(path),
      name: mediaName(path),
      path,
      // Read from the backend that owns this chat, like the built-in image:
      // a split-view tile can belong to another connection than the foreground.
      resolveUrl: () =>
        ['audio', 'video'].includes(mediaKind(path))
          ? resolveMediaPlaybackSrc(path, owner, sessionId ?? undefined)
          : resolveMediaDisplaySrc(path, owner)
    }),
    [owner, path, sessionId]
  )

  const args = useMemo(
    () => (producer.args && typeof producer.args === 'object' ? (producer.args as Record<string, unknown>) : {}),
    [producer.args]
  )

  // Stable component identity per renderer, so a re-render never remounts the widget.
  // Nothing drawn (null/undefined) keeps the built-in player.
  const Leaf = useMemo(
    () =>
      function ToolMediaRendererHost(props: ToolMediaRendererProps) {
        return <>{render(props) ?? props.fallback}</>
      },
    [render]
  )

  return (
    <Leaf
      args={args}
      fallback={fallback}
      file={file}
      result={producer.result}
      sessionId={sessionId}
      toolCallId={producer.toolCallId ?? ''}
      toolName={producer.toolName ?? ''}
    />
  )
}

function claimsTool(claims: readonly Contribution[], toolName: string): boolean {
  return claims.some(claim => {
    const tools = (claim.data as Partial<ToolMediaRendererContribution> | undefined)?.tools

    return (
      Array.isArray(tools) &&
      tools.some(pattern => typeof pattern === 'string' && claimPatternMatches(pattern, toolName))
    )
  })
}

/** The chat's owning connection/profile, as the built-in image reads it. */
function useMediaOwner() {
  const connection = useStore($connection)
  const scope = useComposerScope()
  const connectionId = scope.connectionId || connection?.connectionId
  const profile = scope.profile ?? connection?.profile

  return useMemo(
    () => (scope.connectionId || scope.profile ? { connectionId, profile } : undefined),
    [scope.connectionId, scope.profile, connectionId, profile]
  )
}

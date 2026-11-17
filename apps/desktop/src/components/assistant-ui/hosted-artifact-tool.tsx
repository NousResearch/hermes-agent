import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type FC, useEffect, useMemo, useRef, useState } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { useComposerScope, useComposerSurfaceId } from '@/app/chat/composer/scope'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/external-link'
import {
  hostedArtifactFrameFor,
  type HostedArtifactFramePolicy,
  type HostedArtifactResult,
  parseHostedArtifactResult
} from '@/lib/hosted-artifacts'
import { cn } from '@/lib/utils'
import { $hostedArtifacts, recordHostedArtifactVersion } from '@/store/artifacts'

import { useHostedArtifactBridge } from './hosted-artifact-bridge'

const MIN_HEIGHT = 160
const MAX_HEIGHT = 1200
const DEFAULT_HEIGHT = 420
/** Same column cap as inline `::preview` frames. */
const MAX_FRAME_WIDTH = 960

type HostedArtifactPartProps = ToolCallMessagePartProps & {
  completedAt?: number
  /** Framing policy override (tests). Defaults to this build's policy. */
  framePolicy?: HostedArtifactFramePolicy
  timestamp?: number
}

/** The artifact version a settled, successful call named, or null to let the
 *  generic tool row render (pending, failed, stale-base errors, malformed). */
export function hostedArtifactFromPart(
  part: Pick<ToolCallMessagePartProps, 'isError' | 'result'>
): HostedArtifactResult | null {
  if (part.isError || part.result === undefined) {
    return null
  }

  return parseHostedArtifactResult(part.result)
}

function resultText(result: unknown): string {
  return typeof result === 'string' ? result : JSON.stringify(result ?? null)
}

/**
 * A hosted artifact in the transcript: the live artifact framed from its own
 * origin through the MCP Apps bridge when this build may frame it, otherwise a
 * link card. The viewer link is always there — it is the durable way in once
 * the frame's short-lived ticket has expired.
 */
export const HostedArtifactCard: FC<HostedArtifactPartProps & { artifact: HostedArtifactResult }> = ({
  args,
  artifact,
  completedAt,
  framePolicy,
  result
}) => {
  const { t } = useI18n()
  const copy = t.hostedArtifact
  const cached = useStore($hostedArtifacts)[artifact.id]

  useEffect(() => {
    recordHostedArtifactVersion(artifact)
  }, [artifact])

  // Decided once per mount: a ticket that was fresh when the card mounted
  // has already been exchanged for the frame's cookie.
  const [origin] = useState(() =>
    hostedArtifactFrameFor(artifact, { completedAt, ...(framePolicy ? { policy: framePolicy } : {}) })
  )

  const title = artifact.title || cached?.title || copy.untitled
  const newer = cached && cached.headVersion > artifact.version ? cached.headVersion : null

  return (
    <div className="my-1.5 w-full" data-slot="aui_hosted-artifact" style={{ maxWidth: MAX_FRAME_WIDTH }}>
      <div className={cn(WIDGET_SHELL_CLASS, 'flex w-full items-center gap-2.5 overflow-hidden')}>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            {title}
          </span>
          <span className="block truncate text-[length:var(--conversation-tool-font-size)] text-muted-foreground">
            {copy.version(artifact.version)}
            {newer !== null && ` · ${copy.newerVersion(newer)}`}
            {origin === null && ` · ${copy.linkOnly}`}
          </span>
        </span>
        <ExternalLink
          className="shrink-0 text-[length:var(--conversation-tool-font-size)] font-medium"
          data-slot="aui_hosted-artifact-viewer-link"
          href={artifact.viewerUrl}
          native
          showExternalIcon
        >
          {copy.openViewer}
        </ExternalLink>
      </div>
      {origin !== null && (
        <HostedArtifactFrame
          args={args as Record<string, unknown>}
          contentUrl={artifact.contentUrl}
          origin={origin}
          resultText={resultText(result)}
          title={copy.frameTitle(title)}
        />
      )}
    </div>
  )
}

function HostedArtifactFrame({
  args,
  contentUrl,
  origin,
  resultText,
  title
}: {
  args: Record<string, unknown>
  contentUrl: string
  origin: string
  resultText: string
  title: string
}) {
  const frame = useRef<HTMLIFrameElement>(null)
  const [height, setHeight] = useState(DEFAULT_HEIGHT)
  const [offered, setOffered] = useState<string | null>(null)
  const offeredRef = useRef<string | null>(null)

  const offer = (text: string | null) => {
    offeredRef.current = text
    setOffered(text)
  }

  useHostedArtifactBridge(frame, {
    args,
    // One message waits for the user at a time; the view hears that a second one was refused.
    offerMessage: text => {
      if (offeredRef.current !== null) {
        return false
      }

      offer(text)

      return true
    },
    onHeight: next => setHeight(Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, Math.round(next)))),
    origin,
    resultText
  })

  return (
    <>
      <iframe
        className="mt-1.5 block w-full rounded-md border border-border/60 bg-transparent"
        data-slot="aui_hosted-artifact-frame"
        ref={frame}
        referrerPolicy="no-referrer"
        sandbox="allow-scripts allow-same-origin"
        src={contentUrl}
        style={{ height }}
        title={title}
      />
      {offered !== null && <HostedArtifactMessagePrompt onDone={() => offer(null)} text={offered} />}
    </>
  )
}

/**
 * The confirmation an artifact's `ui/message` needs before it reaches the agent. An artifact can
 * be shared by another user, so its message is shown, never sent on its own, and Send delivers it
 * to the conversation that rendered this frame (its composer target and mounted surface), never to
 * whichever composer is active; when that surface is gone the send fails and the prompt stays.
 */
export function HostedArtifactMessagePrompt({ onDone, text }: { onDone: () => void; text: string }) {
  const { t } = useI18n()
  const copy = t.hostedArtifact
  const { target } = useComposerScope()
  const surfaceId = useComposerSurfaceId()

  const send = () => {
    if (requestComposerSubmit(text, { surfaceId, target })) {
      onDone()
    }
  }

  return (
    <div
      className={cn(WIDGET_SHELL_CLASS, 'mt-1.5 flex w-full flex-col gap-1.5')}
      data-slot="aui_hosted-artifact-message"
      role="group"
    >
      <span className="text-[length:var(--conversation-tool-font-size)] text-muted-foreground">
        {copy.messagePrompt}
      </span>
      <span className="whitespace-pre-wrap break-words text-[length:var(--conversation-text-font-size)] text-foreground">
        {text}
      </span>
      <span className="flex justify-end gap-1.5">
        <Button data-slot="aui_hosted-artifact-message-dismiss" onClick={onDone} size="sm" variant="ghost">
          {copy.messageDismiss}
        </Button>
        <Button data-slot="aui_hosted-artifact-message-send" onClick={send} size="sm">
          {copy.messageSend}
        </Button>
      </span>
    </div>
  )
}

/** Tool-part entry: the card for a successful result, else `fallback`. */
export function HostedArtifactToolPart({
  fallback,
  ...props
}: HostedArtifactPartProps & { fallback: FC<HostedArtifactPartProps> }) {
  const { isError, result } = props
  const artifact = useMemo(() => hostedArtifactFromPart({ isError, result }), [isError, result])

  if (!artifact) {
    const Fallback = fallback

    return <Fallback {...props} />
  }

  return <HostedArtifactCard {...props} artifact={artifact} />
}

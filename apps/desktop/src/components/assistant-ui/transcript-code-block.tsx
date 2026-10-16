import { useStore } from '@nanostores/react'
import { createContext, createElement, type ReactNode, useContext, useEffect, useMemo } from 'react'

import { requestComposerSubmitConfirmed } from '@/app/chat/composer/focus'
import { useComposerScope, useComposerSurfaceId } from '@/app/chat/composer/scope'
import { useSessionView } from '@/app/chat/session-view'
import { ErrorBoundary } from '@/components/error-boundary'
import { registry, useContributions } from '@/contrib'
import {
  sourceFences,
  TRANSCRIPT_CODE_BLOCK_AREA,
  type TranscriptCodeBlockContribution,
  type TranscriptCodeBlockProps
} from '@/lib/transcript-code-blocks'
import { isStoredTranscriptReadOnly } from '@/store/read-only-transcript'
import { knownOwnerForSession } from '@/store/session-states'

interface MessageCodeScope {
  source: string
  messageId: string
  partIndex: number
}

const MessageCodeContext = createContext<MessageCodeScope | null>(null)

/** Only the assistant's visible native text part supplies this scope. */
export function TranscriptCodeBlockScope({
  children,
  source,
  messageId,
  partIndex
}: MessageCodeScope & { children: ReactNode }) {
  const value = useMemo(() => ({ source, messageId, partIndex }), [source, messageId, partIndex])

  return <MessageCodeContext.Provider value={value}>{children}</MessageCodeContext.Provider>
}

function ownerKey(owner: ReturnType<typeof knownOwnerForSession>): string {
  return typeof owner === 'string' ? `profile:${owner}` : owner ? `${owner.connectionId}:${owner.profile}` : ''
}

interface TranscriptCodeBlockHostProps {
  code: string
  language: string
  streaming: boolean
  fallback: ReactNode
}

export function TranscriptCodeBlock({ code, language, streaming, fallback }: TranscriptCodeBlockHostProps) {
  const message = useContext(MessageCodeContext)
  const contributions = useContributions(TRANSCRIPT_CODE_BLOCK_AREA)
  const view = useSessionView()
  const composer = useComposerScope()
  const surfaceId = useComposerSurfaceId()
  const runtimeId = useStore(view.$runtimeId)
  const storedId = useStore(view.$storedId)
  const busy = useStore(view.$busy)
  const owner = knownOwnerForSession(runtimeId || storedId)
  const key = ownerKey(owner)
  const blocks = useMemo(() => sourceFences(message?.source ?? ''), [message?.source])
  const claim = contributions.find(c => (c.data as TranscriptCodeBlockContribution | undefined)?.language === language)

  const matches = blocks.filter(
    b => b.language === language && b.code.trimEnd() === code.replace(/\r\n/g, '\n').trimEnd()
  )

  // Streamdown restarts node positions per chunk. Equal payloads cannot be
  // distinguished from those positions; keep code rather than guess an owner.
  const block = matches.length === 1 ? matches[0] : undefined

  // Route, source and registry ownership are checked again on every call.
  const lifetime = useMemo(() => ({ mounted: false, pending: false }), [])

  useEffect(() => {
    lifetime.mounted = true

    return () => {
      lifetime.mounted = false
    }
  }, [lifetime])

  if (!message || !claim || !block?.closed || streaming) {
    return <>{fallback}</>
  }

  const readOnly =
    message.partIndex < 0 || !runtimeId || !storedId || !key || !surfaceId || isStoredTranscriptReadOnly(storedId)

  const props: TranscriptCodeBlockProps = {
    code,
    language,
    fallback,
    busy,
    readOnly,
    origin: Object.freeze({
      connectionKey: key,
      profile: typeof owner === 'string' ? owner : (owner?.profile ?? ''),
      storedSessionId: storedId ?? '',
      messageId: message.messageId,
      partIndex: message.partIndex,
      blockIndex: block.index
    }),
    submitInteraction: async prompt => {
      const currentMessage = view.$messages.get().find(m => m.id === message.messageId && m.role === 'assistant')
      const currentPart = currentMessage?.parts[message.partIndex]

      if (
        !lifetime.mounted ||
        lifetime.pending ||
        readOnly ||
        !prompt.trim() ||
        new TextEncoder().encode(prompt).length > 64 * 1024 ||
        view.$busy.get() ||
        view.$storedId.get() !== storedId ||
        view.$runtimeId.get() !== runtimeId ||
        ownerKey(knownOwnerForSession(runtimeId)) !== key ||
        isStoredTranscriptReadOnly(storedId) ||
        currentPart?.type !== 'text' ||
        currentPart.text !== message.source ||
        !registry.getArea(TRANSCRIPT_CODE_BLOCK_AREA).includes(claim)
      ) {
        return false
      }

      lifetime.pending = true

      try {
        return await requestComposerSubmitConfirmed(prompt, composer.target, surfaceId!)
      } finally {
        lifetime.pending = false
      }
    }
  }

  return (
    <ErrorBoundary
      fallback={() => <>{fallback}</>}
      key={`${message.messageId}:${message.partIndex}:${block.index}:${claim.id}`}
      label={`contrib:${claim.id}`}
    >
      <CodeBlockLeaf props={props} render={(claim.data as TranscriptCodeBlockContribution).render} />
    </ErrorBoundary>
  )
}

function CodeBlockLeaf({
  render,
  props
}: {
  render: TranscriptCodeBlockContribution['render']
  props: TranscriptCodeBlockProps
}) {
  return createElement(render, props)
}

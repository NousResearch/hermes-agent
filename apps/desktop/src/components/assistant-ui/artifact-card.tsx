'use client'

import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { CodeCardIcon } from '@/components/chat/code-card'
import { useI18n } from '@/i18n'
import type { ArtifactDetection } from '@/lib/artifact-detect'
import { codiconForLanguage } from '@/lib/markdown-code'
import { cn } from '@/lib/utils'
import {
  $artifactRegistry,
  artifactsForSession,
  openArtifactTab,
  selectArtifactVersion,
  upsertArtifact
} from '@/store/artifacts'

interface ArtifactCardProps {
  code: string
  detection: ArtifactDetection
  streaming?: boolean
}

const KIND_ICON: Record<ArtifactDetection['kind'], string> = {
  code: 'code',
  html: 'browser',
  svg: 'symbol-color'
}

function detectionIcon(detection: ArtifactDetection): string {
  return detection.kind === 'code' ? codiconForLanguage(detection.language) : KIND_ICON[detection.kind]
}

/**
 * Transcript stand-in for a fenced block that was promoted to an artifact.
 * Replaces the wall of code with a compact, openable card: icon, title, kind,
 * version badge. While the fence is still streaming
 * it shows a shimmer + line count instead of the growing source.
 *
 * Registration is automatic on completion (so version history accumulates
 * even if the user never opens the card) but opening the rail is strictly
 * click-driven — a background stream never steals the pane.
 */
export function ArtifactCard({ code, detection, streaming = false }: ArtifactCardProps) {
  const { t } = useI18n()
  const copy = t.artifactCard
  const view = useSessionView()
  const runtimeId = useStore(view.$runtimeId)
  const storedId = useStore(view.$storedId)
  const registry = useStore($artifactRegistry)
  const sessionId = storedId || runtimeId || ''

  const trimmed = code.trim()

  // Register/version the artifact once its fence has finished streaming.
  // upsertArtifact dedupes on content hash, so re-renders and transcript
  // replays are no-ops.
  useEffect(() => {
    if (!streaming && sessionId && trimmed) {
      upsertArtifact(sessionId, detection, trimmed)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- detection derives from code
  }, [detection.kind, detection.language, detection.title, sessionId, streaming, trimmed])

  const record = useMemo(() => {
    void registry

    const slugMatch = artifactsForSession(sessionId).find(
      candidate => candidate.kind === detection.kind && candidate.versions.some(v => v.content === trimmed)
    )

    return slugMatch ?? null
  }, [detection.kind, registry, sessionId, trimmed])

  const lineCount = useMemo(() => trimmed.split('\n').length, [trimmed])
  const kindLabel = copy.kind[detection.kind]
  const versionCount = record?.versions.length ?? 0
  const title = (record?.title || detection.title || kindLabel).trim() || kindLabel

  const open = () => {
    if (streaming || !sessionId || !trimmed) {
      return
    }

    // Ensure the registry row exists even if the completion effect hasn't
    // fired yet (e.g. clicked in the same frame the stream sealed).
    const result = upsertArtifact(sessionId, detection, trimmed)

    if (!result) {
      return
    }

    openArtifactTab(result.artifactId)

    // An older card opens at ITS version, not silently the newest — the user
    // clicked this specific iteration. openArtifactTab resets to newest, so
    // re-pin when this card's content is a historical version.
    const versionIndex = result.record.versions.findIndex(version => version.content === trimmed)

    if (versionIndex !== -1 && versionIndex < result.record.versions.length - 1) {
      selectArtifactVersion(result.artifactId, versionIndex)
    }
  }

  return (
    <button
      className={cn(
        'group/artifact my-1.5 flex w-full max-w-md items-center gap-2.5 overflow-hidden rounded-[0.625rem] border border-border px-3 py-2.5 text-left transition-colors',
        streaming ? 'cursor-default' : 'cursor-pointer hover:bg-accent/40'
      )}
      data-slot="aui_artifact-card"
      disabled={streaming}
      onClick={open}
      type="button"
    >
      <span className="grid size-8 shrink-0 place-items-center rounded-md bg-muted/55 text-muted-foreground">
        <CodeCardIcon className="text-[1rem]" name={detectionIcon(detection)} />
      </span>
      <span className="min-w-0 flex-1">
        <span
          className={cn(
            'block truncate text-[length:var(--conversation-text-font-size)] font-medium text-foreground',
            streaming && 'shimmer text-foreground/55'
          )}
        >
          {title}
        </span>
        <span className="block truncate text-[length:var(--conversation-tool-font-size)] text-muted-foreground">
          {streaming
            ? copy.generating(lineCount)
            : versionCount > 1
              ? `${kindLabel} · ${copy.versionBadge(versionCount)}`
              : kindLabel}
        </span>
      </span>
      {!streaming && (
        <span className="shrink-0 text-[length:var(--conversation-tool-font-size)] font-medium text-muted-foreground opacity-0 transition-opacity group-hover/artifact:opacity-100">
          {copy.open}
        </span>
      )}
    </button>
  )
}

import { MessagePrimitive, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type FC, useState } from 'react'

import { useComposerScope } from '@/app/chat/composer/scope'
import { MarkdownTextContent } from '@/components/assistant-ui/markdown-text'
import { messageContentText } from '@/components/assistant-ui/thread/content'
import { MessageTimelineTimestamp } from '@/components/assistant-ui/thread/timeline-timestamp'
import { UserMessageText } from '@/components/assistant-ui/thread/user-message-text'
import { SCAFFOLD_GLYPH_CLASS, SCAFFOLD_LABEL_CLASS, ScaffoldRow } from '@/components/chat/scaffold-row'
import { Codicon } from '@/components/ui/codicon'
import { LogView } from '@/components/ui/log-view'
import { ToolIcon } from '@/components/ui/tool-icon'
import { useI18n } from '@/i18n'
import { LinkifiedText } from '@/lib/external-link'
import { cn } from '@/lib/utils'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

const SLASH_STATUS_RE = /^slash:(?<command>\/[^\n]+)\n(?<output>[\s\S]*)$/
const STEER_NOTE_RE = /^steer:(?<text>[\s\S]+)$/
const REVIEW_NOTE_RE = /^review:(?<label>[^:\n]+):?\s*(?<detail>[\s\S]*)$/
// A usage-window plan gauge exactly as the backend renders it: `[` + 10 cells of
// `|` then spaces + `]`. Only a window plan's /usage output carries one.
const USAGE_WINDOW_GAUGE_RE = /\[(?=[| ]{10}\])\|* *\]/
const PENDING_SKILL_DIFF_COMMAND_RE = /^\/skills$/i
const PENDING_SKILL_WRITE_HEADER_RE = /^# Pending skill write \S+:/i
const PENDING_SKILL_LIST_HEADER_RE = /^Pending skills? writes \(\d+\):?$/i

function labelMissingSkillNameInPatchHeading(output: string): string {
  return output.replace(/^(## op \d+\/\d+: patch)\s+(?:''|"")\s+/gim, '$1 [missing skill name] ')
}

interface PendingSkillListEntry {
  automatic: boolean
  id: string
  summary: string
}

function PendingSkillList({ output }: { output: string }) {
  const [heading = '', ...lines] = output.split('\n')
  const entries: PendingSkillListEntry[] = []
  const footer: string[] = []
  let readingEntries = true

  for (const line of lines) {
    const match = line.match(/^\s*([a-f\d]{8,})\s+(.+?)\s*$/i)

    if (readingEntries && match) {
      const summary = match[2]
      const automatic = /^\[auto\]\s*/i.test(summary)

      entries.push({
        automatic,
        id: match[1],
        summary: automatic ? summary.replace(/^\[auto\]\s*/i, '') : summary
      })

      continue
    }

    if (line.trim()) {
      readingEntries = false
    }

    if (!readingEntries) {
      footer.push(line)
    }
  }

  return (
    <div className="mt-1 min-w-0" data-slot="pending-skill-list">
      <div className="mb-1 text-xs font-medium text-foreground/80">{heading}</div>
      <div className="max-h-80 overflow-auto rounded border border-border/40 bg-muted/10">
        {entries.map(entry => (
          <div
            className="flex min-w-0 items-start gap-2 border-b border-border/30 px-2 py-1.5 last:border-b-0"
            data-slot="pending-skill-entry"
            key={entry.id}
          >
            <code className="shrink-0 rounded bg-muted/60 px-1.5 py-0.5 font-mono text-[0.65rem] text-foreground/80">
              {entry.id}
            </code>
            {entry.automatic && (
              <span className="shrink-0 rounded bg-muted/60 px-1.5 py-0.5 font-mono text-[0.6rem] text-muted-foreground">
                auto
              </span>
            )}
            <span className="min-w-0 whitespace-pre-wrap break-words pt-0.5 font-mono text-[0.65rem] text-muted-foreground">
              {entry.summary}
            </span>
          </div>
        ))}
      </div>
      {footer.join('\n').trim() && (
        <LinkifiedText
          className="mt-2 block whitespace-pre-wrap text-xs text-muted-foreground"
          explicitOnly
          pretty={false}
          text={footer.join('\n').trim()}
        />
      )}
    </div>
  )
}

interface BackgroundResultProps {
  text: string
  report: string
  process?: boolean
}

export const BackgroundResult: FC<BackgroundResultProps> = ({ text, report, process }) => {
  const [open, setOpen] = useState(false)

  return (
    <div
      className="flex w-full min-w-0 flex-col self-start py-1 pl-(--message-text-indent)"
      data-slot="aui_background-result"
    >
      <div data-conversation-scaffold="">
        <ScaffoldRow
          onToggle={report ? () => setOpen(!open) : undefined}
          open={open}
          trailing={
            <>
              {' '}
              <MessageTimelineTimestamp />
            </>
          }
        >
          {process && (
            <span className={SCAFFOLD_GLYPH_CLASS}>
              <ToolIcon className="text-(--ui-text-tertiary)" name="terminal" size="0.875rem" />
            </span>
          )}
          <span className={cn(SCAFFOLD_LABEL_CLASS, 'min-w-0 truncate')}>{text}</span>
        </ScaffoldRow>
      </div>
      {open &&
        (process ? (
          <LogView className="mt-2 max-h-80 overscroll-x-contain overscroll-y-auto">{report}</LogView>
        ) : (
          <div className="mt-2 max-h-80 min-w-0 max-w-full overflow-auto overscroll-x-contain overscroll-y-auto wrap-anywhere">
            <MarkdownTextContent isRunning={false} text={report} />
          </div>
        ))}
    </div>
  )
}

/** A safety-refused turn kept for display: the user's own message rendered
 *  greyed and non-interactive (no edit, no actions), with a short notice that
 *  the model's safety filter refused it. Right-aligned like the prompt it was,
 *  but a dashed muted card — never the blue user bubble, never a reply. */
const ContentRefusedNote: FC<{ notice: string; userText: string }> = ({ notice, userText }) => (
  <MessagePrimitive.Root
    className="flex w-full min-w-0 max-w-full flex-col items-stretch gap-1 self-end pb-(--conversation-turn-gap)"
    data-role="system"
    data-slot="aui_content-refused"
  >
    <div className="ml-auto flex min-w-0 max-w-full flex-col gap-1.5 rounded-xl border border-dashed border-(--ui-stroke-tertiary) px-3 py-2 text-left opacity-70">
      {userText && <UserMessageText className="wrap-anywhere text-(--ui-text-tertiary)" text={userText} />}
      <span className="flex items-center gap-1.5 text-[0.6875rem] leading-5 text-muted-foreground/70">
        <Codicon className="shrink-0 text-muted-foreground/60" name="shield" size="0.75rem" />
        <span className="wrap-anywhere">{notice}</span>
      </span>
    </div>
    <MessageTimelineTimestamp className="self-end pr-1.5" />
  </MessagePrimitive.Root>
)

export const SystemMessage: FC = () => {
  const { t } = useI18n()
  const text = useAuiState(s => messageContentText(s.message.content))
  const asyncResult = useAuiState(s => s.message.metadata.custom?.asyncResult)
  const processResult = useAuiState(s => s.message.metadata.custom?.asyncResultKind === 'process')
  const { t } = useI18n()
  const { profile: scopedProfile } = useComposerScope()
  const activeProfile = useStore($activeGatewayProfile)
  const pendingSkillProfile = normalizeProfileKey(scopedProfile || activeProfile)

  const contentRefused = useAuiState(
    s => s.message.metadata.custom?.contentRefused as undefined | { notice?: string }
  )

  // A safety-refused turn, kept for display only: the user's own message, greyed
  // and non-interactive, with a short notice. Not a user bubble, not a reply.
  if (contentRefused) {
    return <ContentRefusedNote notice={contentRefused.notice ?? t.assistant.thread.contentRefusedNotice} userText={text} />
  }

  if (!text) {
    return null
  }

  if (processResult || (typeof asyncResult === 'string' && asyncResult)) {
    return (
      <MessagePrimitive.Root
        className="w-full min-w-0 self-start"
        data-role="system"
        data-slot="aui_system-message-root"
      >
        <BackgroundResult
          process={processResult}
          report={typeof asyncResult === 'string' ? asyncResult : ''}
          text={text}
        />
      </MessagePrimitive.Root>
    )
  }

  // The self-improvement review saved something to memory/skills — the same
  // kind of event as a landed `memory` write, so it wears the same chrome:
  // brain glyph with the gold→purple glow, gradient label, purple detail,
  // left-aligned in the reading column like every other scaffold line.
  const reviewNote = text.match(REVIEW_NOTE_RE)

  if (reviewNote?.groups) {
    const detail = reviewNote.groups.detail.trim()

    return (
      <MessagePrimitive.Root
        className="flex w-full min-w-0 max-w-full items-start gap-1.5 self-start py-0.5"
        data-role="system"
        data-slot="aui_system-message-root"
      >
        <span className="tool-memory-legendary-glyph flex h-(--conversation-line-height) w-3.5 shrink-0 items-center justify-center">
          <ToolIcon className="text-(--tool-memory-legendary-icon)" name="brain" size="0.875rem" />
        </span>
        <span className={cn(SCAFFOLD_LABEL_CLASS, 'tool-memory-legendary-title shrink-0 text-transparent')}>
          {reviewNote.groups.label.trim()}
        </span>
        {detail && (
          <span className={cn(SCAFFOLD_LABEL_CLASS, 'tool-memory-legendary-meta min-w-0 wrap-anywhere')}>{detail}</span>
        )}
      </MessagePrimitive.Root>
    )
  }

  const steerNote = text.match(STEER_NOTE_RE)

  if (steerNote?.groups) {
    return (
      <MessagePrimitive.Root
        className="flex max-w-[min(86%,44rem)] items-center gap-1.5 self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/60"
        data-role="system"
        data-slot="aui_system-message-root"
      >
        <Codicon className="text-muted-foreground/55" name="compass" size="0.75rem" />
        <span className="text-muted-foreground/55">steered</span>
        <span className="text-muted-foreground/35">·</span>
        <span className="whitespace-pre-wrap">{steerNote.groups.text.trim()}</span> <MessageTimelineTimestamp />
      </MessagePrimitive.Root>
    )
  }

  const slashStatus = text.match(SLASH_STATUS_RE)

  if (slashStatus?.groups) {
    const output = slashStatus.groups.output.trim()

    const isPendingSkillDiff =
      PENDING_SKILL_DIFF_COMMAND_RE.test(slashStatus.groups.command.trim()) &&
      PENDING_SKILL_WRITE_HEADER_RE.test(output)

    const isPendingSkillList =
      PENDING_SKILL_DIFF_COMMAND_RE.test(slashStatus.groups.command.trim()) &&
      PENDING_SKILL_LIST_HEADER_RE.test(output.split('\n', 1)[0]?.trim() ?? '')

    if (isPendingSkillDiff || isPendingSkillList) {
      const diffOutput = isPendingSkillDiff ? labelMissingSkillNameInPatchHeading(output) : output

      return (
        <MessagePrimitive.Root
          className="w-full max-w-[44rem] min-w-0 self-center px-2 py-1 text-left"
          data-role="system"
          data-slot="aui_system-message-root"
        >
          <div className="mb-1 flex min-w-0 items-center justify-between gap-3">
            <span className="font-mono text-[0.65rem] text-muted-foreground/55">{slashStatus.groups.command}</span>
            <span
              className="min-w-0 truncate font-mono text-[0.65rem] text-muted-foreground/55"
              data-slot="pending-skill-profile"
              title={t.sidebar.row.ownedByProfile(pendingSkillProfile)}
            >
              {t.sidebar.row.ownedByProfile(pendingSkillProfile)}
            </span>
          </div>
          {isPendingSkillDiff ? (
            <MarkdownTextContent isRunning={false} text={diffOutput} />
          ) : (
            <PendingSkillList output={output} />
          )}
          <MessageTimelineTimestamp className="mt-1 block" />
        </MessagePrimitive.Root>
      )
    }

    // Single-line status (e.g. "model → x") reads best centered inline; padded
    // multiline output (catalogs, usage tables) needs left-aligned, wider room
    // or the column alignment breaks.
    const multiline = output.includes('\n')
    // A window plan's /usage (token table + `[||||||    ]` plan gauges) only lines
    // up in a monospace face. Every other output keeps the default face.
    const tabular = multiline && USAGE_WINDOW_GAUGE_RE.test(output)

    return (
      <MessagePrimitive.Root
        className={cn(
          'w-[60%] max-w-[44rem] self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/60',
          multiline ? 'text-left' : 'text-center'
        )}
        data-role="system"
        data-slot="aui_system-message-root"
      >
        <span className="font-mono text-muted-foreground/55">{slashStatus.groups.command}</span>
        {multiline ? (
          <LinkifiedText
            className={cn('mt-0.5 block whitespace-pre-wrap', tabular && 'font-mono text-[0.65rem] leading-[1.15rem]')}
            explicitOnly
            pretty={false}
            text={output}
          />
        ) : (
          <>
            <span className="mx-1.5 text-muted-foreground/35">·</span>
            <LinkifiedText className="whitespace-pre-wrap" explicitOnly pretty={false} text={output} />
          </>
        )}{' '}
        <MessageTimelineTimestamp className={cn(multiline ? 'mt-0.5 block' : 'ml-1.5')} />
      </MessagePrimitive.Root>
    )
  }

  const multiline = text.includes('\n')

  return (
    <MessagePrimitive.Root
      className={cn(
        'w-[60%] max-w-[44rem] self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/55',
        multiline ? 'text-left' : 'text-center'
      )}
      data-role="system"
      data-slot="aui_system-message-root"
    >
      <LinkifiedText className="whitespace-pre-wrap" explicitOnly pretty={false} text={text} />{' '}
      <MessageTimelineTimestamp className={cn(multiline ? 'mt-0.5 block' : 'ml-1.5')} />
    </MessagePrimitive.Root>
  )
}

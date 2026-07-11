import * as React from 'react'

import { writeClipboardText } from '@/components/ui/copy-button'
import { Tip } from '@/components/ui/tooltip'
import { Check, Copy, iconSize } from '@/lib/icons'
import {
  SLASH_REFERENCE_DYNAMIC_ROUTES_NOTE,
  slashReferenceAliasText,
  type SlashReferenceCommand,
  slashReferenceCommands,
  slashReferenceCommandToken,
  slashReferenceDisplayDescription,
  type SlashReferenceSection,
  slashReferenceSections,
  slashReferenceSurfaceTag,
  type SlashReferenceSurfaceTag
} from '@/lib/slash-command-reference'
import { cn } from '@/lib/utils'

const ACCENT_CLASS: Record<SlashReferenceSection['accent'], string> = {
  cyan: 'slash-ref-accent-cyan',
  magenta: 'slash-ref-accent-magenta',
  orange: 'slash-ref-accent-orange',
  yellow: 'slash-ref-accent-yellow'
}

const TAG_LABEL: Record<SlashReferenceSurfaceTag, string> = {
  both: 'BOTH',
  cfg: 'CFG',
  chat: 'CHAT',
  cli: 'CLI'
}

const ALIAS_DISPLAY_LIMITS: Readonly<Record<string, number>> = {
  background: 1,
  'codex-runtime': 0,
  journey: 1
}

function TagBadge({ tag }: { tag: SlashReferenceSurfaceTag }) {
  return <span className={cn('slash-ref-tag', `slash-ref-tag-${tag}`)}>{TAG_LABEL[tag]}</span>
}

function CommandRow({
  command,
  copied,
  onCopy
}: {
  command: SlashReferenceCommand
  copied: boolean
  onCopy: (command: SlashReferenceCommand) => void
}) {
  const commandText = slashReferenceCommandToken(command)
  const aliases = slashReferenceAliasText(command, ALIAS_DISPLAY_LIMITS[command.name])
  const description = slashReferenceDisplayDescription(command)
  const tag = slashReferenceSurfaceTag(command)

  const commandDetails = [
    `Copy ${commandText}`,
    description,
    aliases ? `also known as ${aliases}` : '',
    command.argsHint ? `arguments ${command.argsHint}` : ''
  ]
    .filter(Boolean)
    .join('. ')

  return (
    <button aria-label={commandDetails} className="slash-ref-command" onClick={() => onCopy(command)} type="button">
      <span className="slash-ref-command-token">{commandText}</span>
      <span className="slash-ref-command-summary">
        <span className="slash-ref-command-description">{description}</span>
        {aliases && <span className="slash-ref-command-aka">aka {aliases}</span>}
      </span>
      <span className="slash-ref-command-meta">
        <TagBadge tag={tag} />
        <Tip label={copied ? 'Copied' : `Copy ${commandText}`} side="left">
          <span className="slash-ref-copy-icon" role="presentation">
            {copied ? <Check className={iconSize.sm} /> : <Copy className={iconSize.sm} />}
          </span>
        </Tip>
      </span>
    </button>
  )
}

function Section({
  copiedCommand,
  onCopy,
  section
}: {
  copiedCommand: null | string
  onCopy: (command: SlashReferenceCommand) => void
  section: SlashReferenceSection
}) {
  return (
    <section className={cn('slash-ref-section', ACCENT_CLASS[section.accent])}>
      <div className="slash-ref-section-header">
        <h2>{section.title}</h2>
        <span>{section.commands.length}</span>
      </div>
      <div className="slash-ref-command-list">
        {section.commands.map(command => (
          <CommandRow command={command} copied={copiedCommand === command.name} key={command.name} onCopy={onCopy} />
        ))}
      </div>
    </section>
  )
}

export function CommandsTab() {
  const [query, setQuery] = React.useState('')
  const [copiedCommand, setCopiedCommand] = React.useState<null | string>(null)
  const resetRef = React.useRef<number | null>(null)

  const needle = query.trim().toLowerCase()

  const sections = React.useMemo(() => {
    if (!needle) {
      return slashReferenceSections
    }

    return slashReferenceSections
      .map(section => ({
        ...section,
        commands: section.commands.filter(command => {
          const commandText = slashReferenceCommandToken(command).toLowerCase()
          const description = slashReferenceDisplayDescription(command).toLowerCase()
          const aliases = slashReferenceAliasText(command).toLowerCase()

          return commandText.includes(needle) || description.includes(needle) || aliases.includes(needle)
        })
      }))
      .filter(section => section.commands.length > 0)
  }, [needle])

  const copiedReferenceCommand = copiedCommand
    ? slashReferenceCommands.find(command => command.name === copiedCommand)
    : null

  const copiedToken = copiedReferenceCommand ? slashReferenceCommandToken(copiedReferenceCommand) : null

  React.useEffect(() => {
    return () => {
      if (resetRef.current !== null) {
        window.clearTimeout(resetRef.current)
      }
    }
  }, [])

  const onCopy = React.useCallback(async (command: SlashReferenceCommand) => {
    await writeClipboardText(slashReferenceCommandToken(command))
    setCopiedCommand(command.name)

    if (resetRef.current !== null) {
      window.clearTimeout(resetRef.current)
    }

    resetRef.current = window.setTimeout(() => {
      setCopiedCommand(null)
      resetRef.current = null
    }, 1_400)
  }, [])

  return (
    <div className="mbc-tab-panel" data-tab="commands">
      <div className="mbc-search-row">
        <input
          aria-label="Filter slash commands"
          className="mbc-search"
          onChange={event => setQuery(event.target.value)}
          placeholder="Filter commands…"
          type="search"
          value={query}
        />
        <span className="mbc-count-chip">{slashReferenceCommands.length}</span>
      </div>

      <div aria-label="Tags" className="slash-ref-legend">
        <span>Tags</span>
        <span className="slash-ref-tag slash-ref-tag-cli">CLI</span>
        <span className="slash-ref-tag slash-ref-tag-chat">CHAT</span>
        <span className="slash-ref-tag slash-ref-tag-cfg">CFG</span>
        <span className="slash-ref-tag slash-ref-tag-both">BOTH</span>
      </div>

      <div aria-label="Slash commands" className="slash-ref-scroll">
        {sections.map(section => (
          <Section copiedCommand={copiedCommand} key={section.title} onCopy={onCopy} section={section} />
        ))}
        <footer className="slash-ref-dynamic-note">
          <strong>Dynamic routes</strong>
          <span>{SLASH_REFERENCE_DYNAMIC_ROUTES_NOTE}</span>
        </footer>
      </div>

      <span aria-live="polite" className="slash-ref-sr-only">
        {copiedToken ? `${copiedToken} copied` : ''}
      </span>
    </div>
  )
}

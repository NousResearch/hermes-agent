import { Codicon } from '@/components/ui/codicon'

interface AgentSettingsLinkProps {
  label: string
  description: string
  onOpen: () => void
}

/** The Settings rail's door to per-agent settings (Figma 856:2764 "Agent settings" card). */
export function AgentSettingsLink({ label, description, onOpen }: AgentSettingsLinkProps) {
  return (
    <button className="settings-agent-link" data-tour="nav-agent-settings" onClick={onOpen} type="button">
      <span className="settings-agent-link-text">
        <span>{label}</span>
        <span>{description}</span>
      </span>
      <Codicon aria-hidden name="chevron-right" />
    </button>
  )
}

import { pluginInstallShareUrl } from '@hermes/shared'

import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import { Tip } from '@/components/ui/tooltip'
import type { PluginRecord } from '@/contrib/plugins-store'
import { useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'
import { Upload } from '@/lib/icons'
import { SUBMIT_MOD_URL } from '@/lib/plugin-catalog'
import type { AgentPluginRow } from '@/store/agent-plugins'

interface PluginShareProps {
  /** The agent half in the selected profile; absent for desktop-only packages. */
  agent: AgentPluginRow | null
  /** The Desktop half of the package; absent for agent-only packages. */
  desktop: null | PluginRecord
  pluginName: string
}

/**
 * The share hand-off on a package row. A catalog package hands its curated
 * name (the recipient's backend resolves the reviewed pin), a git package its
 * repo; either copies an install link any Hermes user can open. A hand-copied
 * folder has neither origin, so it is pointed at the catalog's publish flow.
 * Bundled plugins have neither and render nothing.
 */
export function PluginShare({ agent, desktop, pluginName }: PluginShareProps) {
  const { t } = useI18n()
  const copy = t.skills.plugins

  const shareUrl = pluginInstallShareUrl({
    catalogName: agent?.catalog_name ?? desktop?.packageOrigin?.catalogName,
    repo: desktop?.packageOrigin?.repo
  })

  if (shareUrl) {
    return <CopyButton appearance="icon" label={copy.shareTip(pluginName)} text={shareUrl} />
  }

  const handCopied =
    agent?.source === 'user' || (desktop?.kind === 'disk' && !desktop.packageName && !desktop.packageOrigin)

  if (!handCopied) {
    return null
  }

  return (
    <Tip label={copy.publishTip(pluginName)}>
      <Button
        aria-label={copy.publish(pluginName)}
        onClick={() => openExternalLink(SUBMIT_MOD_URL)}
        size="icon"
        variant="ghost"
      >
        <Upload className="size-[0.85rem]" />
      </Button>
    </Tip>
  )
}

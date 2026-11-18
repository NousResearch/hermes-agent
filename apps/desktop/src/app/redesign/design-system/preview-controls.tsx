import { useStore } from '@nanostores/react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import { $portalAccent, pickPortalAccent, PORTAL_ACCENTS } from '@/store/portal-accent'
import { useTheme } from '@/themes/context'

import { useFirstTimeCopy } from '../../discovery/first-time-copy'

import type { DesignSystemCopy } from './design-system-copy'

/**
 * Light / dark and the five accents, through the real switches. The tokens are scoped to the root
 * (`:root[data-shell='portal']:not(.dark)`, `[data-portal-accent]`, the skin's `--ui-base` on <html>),
 * so a part of the page cannot preview another theme without restating them; these set the person's
 * own appearance, and the page says so.
 */
export function PreviewControls({ copy }: { copy: DesignSystemCopy }) {
  const { renderedMode, setMode } = useTheme()
  const accent = useStore($portalAccent)
  const colors = useFirstTimeCopy().splash.theme.colors

  return (
    <div aria-label={copy.preview} className="ds-preview" role="group">
      <div className="ds-preview-control">
        <span className="ds-caption">{copy.mode}</span>
        <SegmentedControl
          onChange={setMode}
          options={[
            { id: 'light', label: copy.light },
            { id: 'dark', label: copy.dark }
          ]}
          value={renderedMode}
        />
      </div>
      <div className="ds-preview-control">
        <span className="ds-caption">{copy.accent}</span>
        <SegmentedControl
          onChange={pickPortalAccent}
          options={PORTAL_ACCENTS.map(id => ({ id, label: colors[id] }))}
          sizing="content"
          value={accent}
        />
      </div>
      <p className="ds-preview-note">{copy.previewNote}</p>
    </div>
  )
}

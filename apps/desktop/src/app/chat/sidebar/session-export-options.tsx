import { Checkbox } from '@/components/ui/checkbox'
import { useI18n } from '@/i18n'
import type { SaveFormat } from '@/lib/save-transcript'
import type { TranscriptOptions } from '@/lib/transcript-document'

interface SessionExportOptionsProps {
  busy: boolean
  format: SaveFormat
  onChange: (key: keyof TranscriptOptions, value: boolean) => void
  options: TranscriptOptions
  redact: boolean
}

export function SessionExportOptions({ busy, format, onChange, options, redact }: SessionExportOptionsProps) {
  const { t } = useI18n()

  const labels: Array<[keyof TranscriptOptions, string]> = [
    ['include_images', t.desktop.sessionExportImages],
    ['include_videos', t.desktop.sessionExportVideos],
    ['include_files', t.desktop.sessionExportFiles],
    ['include_links', t.desktop.sessionExportLinks],
    ['include_asides', t.desktop.sessionExportAsides],
    ['include_tools', t.desktop.sessionExportTools],
    ['include_reasoning', t.desktop.sessionExportReasoning],
    ['collapse_details', t.desktop.sessionExportCollapse]
  ]

  const foldHints: Record<SaveFormat, string> = {
    docx: t.desktop.sessionExportFoldDocx,
    odt: t.desktop.sessionExportFoldOdt,
    html: t.desktop.sessionExportFoldHtml,
    md: t.desktop.sessionExportFoldMd,
    json: t.desktop.sessionExportFoldJson
  }

  return (
    <section aria-labelledby="export-contents-label" className="grid gap-2">
      <h3 className="text-sm font-medium" id="export-contents-label">
        {t.desktop.sessionExportContents}
      </h3>
      <div className="grid gap-x-4 gap-y-1 sm:grid-cols-2">
        {labels.map(([key, label]) => (
          <label className="flex min-h-8 items-center gap-2 text-sm" key={key}>
            <Checkbox
              checked={options[key]}
              disabled={busy || (key === 'collapse_details' && (format === 'json' || format === 'odt'))}
              onCheckedChange={value => onChange(key, value === true)}
            />
            <span>{label}</span>
          </label>
        ))}
      </div>
      <p className="text-xs leading-relaxed text-(--ui-text-tertiary)">{t.desktop.sessionExportMediaHint}</p>
      <p className="text-xs leading-relaxed text-(--ui-text-tertiary)">{t.desktop.sessionExportLinkHint}</p>
      <p className="text-xs leading-relaxed text-(--ui-text-tertiary)">{foldHints[format]}</p>
      {redact && options.include_images && (
        <p className="text-xs leading-relaxed text-(--ui-text-secondary)" role="note">
          {t.desktop.sessionExportImagePrivacy}
        </p>
      )}
    </section>
  )
}

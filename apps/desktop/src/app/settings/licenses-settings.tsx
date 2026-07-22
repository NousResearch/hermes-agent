import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { FileText, Loader2, AlertCircle } from '@/lib/icons'

import { ListRow, SectionHeading, SettingsContent } from './primitives'

type LicenseSource = {
  key: string
  filename: string
  label: string
  description: string
}

const LICENSE_SOURCES: readonly LicenseSource[] = [
  {
    key: 'js',
    filename: 'dependencies.txt',
    label: 'JavaScript Dependencies',
    description: 'npm packages bundled with the desktop app renderer and Electron main process.'
  },
  {
    key: 'python',
    filename: 'dependencies-python.txt',
    label: 'Python Dependencies',
    description: 'Python packages from the Hermes Agent runtime environment.'
  }
]

async function loadLicenseFile(filename: string): Promise<string | null> {
  const bridge = window.hermesDesktop
  if (!bridge?.readBundledLicenseFile) {
    return null
  }

  return bridge.readBundledLicenseFile(filename)
}

function LicenseFileBlock({ source }: { source: LicenseSource }) {
  const [content, setContent] = useState<string | null | undefined>(undefined)

  useEffect(() => {
    let cancelled = false
    void loadLicenseFile(source.filename).then(result => {
      if (!cancelled) {
        setContent(result)
      }
    })
    return () => {
      cancelled = true
    }
  }, [source.filename])

  const isLoading = content === undefined
  const isMissing = content === null
  const isEmpty = content !== null && content !== undefined && content.trim() === ''

  return (
    <div className="mt-4">
      <SectionHeading icon={FileText} title={source.label} />

      <div className="overflow-hidden rounded-xl border border-border/70 bg-muted/20">
        {isLoading ? (
          <div className="flex items-center gap-2 px-4 py-3 text-sm text-muted-foreground">
            <Loader2 className="size-3 animate-spin" />
            Loading…
          </div>
        ) : isMissing ? (
          <div className="flex items-start gap-2 px-4 py-3 text-sm text-muted-foreground">
            <AlertCircle className="mt-0.5 size-3.5 shrink-0 text-amber-500" />
            <div>
              <p className="font-medium text-foreground">License file not found</p>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Run <code className="rounded bg-muted px-1 py-0.5 text-xs">npm run generate-licenses</code> in{' '}
                <code className="rounded bg-muted px-1 py-0.5 text-xs">apps/desktop</code> to generate it, or it was
                not bundled with this build.
              </p>
            </div>
          </div>
        ) : isEmpty ? (
          <div className="px-4 py-3 text-sm text-muted-foreground">
            License file is empty.
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between border-b border-border/50 px-4 py-2">
              <span className="text-xs text-muted-foreground">
                {source.description}
              </span>
              <Button
                onClick={() => void navigator.clipboard.writeText(content || '')}
                size="sm"
                variant="text"
              >
                <Codicon className="size-3" name="copy" />
                Copy
              </Button>
            </div>
            <pre className="max-h-[400px] overflow-auto px-4 py-3 text-xs leading-relaxed text-foreground/80 whitespace-pre-wrap break-words font-mono">
              {content}
            </pre>
          </>
        )}
      </div>
    </div>
  )
}

export function LicensesSettings() {
  return (
    <SettingsContent>
      <div className="mx-auto w-full max-w-4xl">
        <SectionHeading icon={FileText} title="Third-Party Licenses" />

        <ListRow
          description="License attributions for open-source dependencies bundled with Hermes. These files are generated at build time and shipped with the application."
          title="Open-Source License Attribution"
        />

        {LICENSE_SOURCES.map(source => (
          <LicenseFileBlock key={source.key} source={source} />
        ))}

        <div className="mt-6 px-1 pb-4 text-xs text-muted-foreground">
          Hermes Agent is licensed under the MIT License. For the full text, see the LICENSE file in the{' '}
          <a
            href="https://github.com/NousResearch/hermes-agent"
            onClick={e => {
              e.preventDefault()
              void window.hermesDesktop?.openExternal?.('https://github.com/NousResearch/hermes-agent')
            }}
            className="text-primary hover:underline"
            rel="noreferrer"
            target="_blank"
          >
            source repository
          </a>
          .
        </div>
      </div>
    </SettingsContent>
  )
}

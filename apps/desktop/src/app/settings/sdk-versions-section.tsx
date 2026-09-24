// Settings -> Providers: collapsible "SDK and Runtime Versions" panel
// (#120879). Read-only visibility into which agent SDK + runtime versions the
// install carries, so troubleshooting and bug reports can quote them. The
// main process reads each version from the package's installed package.json
// at request time; nothing here is hardcoded.
import { useEffect, useState } from 'react'

import { CopyButton } from '@/components/ui/copy-button'
import type { DesktopSdkVersions } from '@/global'
import { translateNow, useI18n } from '@/i18n'
import { ChevronDown, ChevronRight, Package } from '@/lib/icons'

/** Plain-text block the copy-all button puts on the clipboard for bug reports. */
export function formatSdkVersionsForCopy(info: DesktopSdkVersions, notInstalled: string): string {
  const lines = [
    translateNow('settings.sdkVersions.title'),
    `Node.js: ${info.node || notInstalled}`,
    `Electron: ${info.electron || notInstalled}`,
    ...info.sdks.map(sdk => `${sdk.name}: ${sdk.version ?? notInstalled}`)
  ]

  return lines.join('\n')
}

function VersionRow({ name, version }: { name: string; version: null | string }) {
  const { t } = useI18n()

  return (
    <div className="flex items-baseline justify-between gap-3 px-1 py-1 text-[length:var(--conversation-caption-font-size)]">
      <span className="min-w-0 break-all text-muted-foreground">{name}</span>
      <span className="shrink-0 font-mono text-foreground">{version ?? t.settings.sdkVersions.notInstalled}</span>
    </div>
  )
}

export function SdkVersionsSection() {
  const { t } = useI18n()
  const c = t.settings.sdkVersions
  const [open, setOpen] = useState(false)
  const [info, setInfo] = useState<DesktopSdkVersions | null>(null)
  const bridge = window.hermesDesktop?.getSdkVersions

  useEffect(() => {
    if (!bridge) {
      return
    }

    let cancelled = false

    void bridge()
      .then(next => {
        if (!cancelled) {
          setInfo(next)
        }
      })
      .catch(() => {
        // A failed read leaves the panel on its loading state rather than
        // taking the Providers page down with it.
      })

    return () => {
      cancelled = true
    }
  }, [bridge])

  // Web dashboard, or an older preload mid-upgrade: no door, no panel.
  if (!bridge) {
    return null
  }

  const Chevron = open ? ChevronDown : ChevronRight

  return (
    <section className="mt-6 rounded-xl border border-border/70 bg-muted/20">
      <div className="flex items-center gap-1 pr-2">
        <button
          aria-expanded={open}
          className="flex flex-1 items-center gap-2 px-3 py-2.5 text-left text-[length:var(--conversation-text-font-size)] font-medium"
          onClick={() => setOpen(prev => !prev)}
          type="button"
        >
          <Chevron className="size-4 shrink-0 text-muted-foreground" />
          <Package className="size-4 shrink-0 text-muted-foreground" />
          <span>{c.title}</span>
        </button>
        {info && (
          <CopyButton buttonSize="sm" buttonVariant="textStrong" label={c.copyAll} text={() => formatSdkVersionsForCopy(info, c.notInstalled)}>
            {c.copyAll}
          </CopyButton>
        )}
      </div>

      {open && (
        <div className="border-t border-border/60 px-3 py-2">
          {info ? (
            <>
              <p className="px-1 pt-1 text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-tertiary)">
                {c.sdksLabel}
              </p>
              {info.sdks.map(sdk => (
                <VersionRow key={sdk.name} name={sdk.name} version={sdk.version} />
              ))}
              <p className="px-1 pt-2 text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-text-tertiary)">
                {c.runtimeLabel}
              </p>
              <VersionRow name="Node.js" version={info.node || null} />
              <VersionRow name="Electron" version={info.electron || null} />
            </>
          ) : (
            <p className="px-1 py-2 text-[length:var(--conversation-caption-font-size)] text-muted-foreground">
              {c.loading}
            </p>
          )}
        </div>
      )}
    </section>
  )
}

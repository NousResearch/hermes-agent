import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { parseOrbConfigUrl } from '@/components/orb/orb-url'
import { OrbView } from '@/components/orb/OrbView'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { $orbConfigUrl, $orbEnabled, $orbParams, setOrbConfigUrl, setOrbEnabled } from '@/store/orb'

import { ListRow, SectionHeading, ToggleRow } from './primitives'

const CONFIGURATOR_URL = 'https://lersent001.github.io/orb/'

/**
 * Thinking-orb customization. The orb is a WebGPU liquid-glass animation
 * (vendored lersent001/orb renderer); pasting a configurator URL brings your
 * own orb into the app — the URL hash fully describes the orb to render.
 */
export function OrbSettings() {
  const { t } = useI18n()
  const copy = t.settings.appearance.orb
  const enabled = useStore($orbEnabled)
  const configUrl = useStore($orbConfigUrl)
  const params = useStore($orbParams)
  const [draft, setDraft] = useState(configUrl)

  // Live validation of what the user typed; the committed URL's own error
  // ($orbUrlError) only matters for programmatic writes, which don't happen here.
  const draftResult = draft.trim() === '' ? null : parseOrbConfigUrl(draft)
  const showError = draftResult && 'error' in draftResult ? draftResult.error : null

  const commitDraft = () => {
    setOrbConfigUrl(draft)
    triggerHaptic('crisp')
  }

  return (
    <div>
      <SectionHeading icon={SparklesIcon} title={copy.title} />
      <p className="max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {copy.intro}{' '}
        <a
          className="underline underline-offset-2 hover:text-(--ui-text-secondary)"
          href={CONFIGURATOR_URL}
          rel="noreferrer"
          target="_blank"
        >
          {copy.openConfigurator}
        </a>
      </p>

      <div className="mt-2">
        <ToggleRow
          checked={enabled}
          description={copy.enableDesc}
          label={copy.enableTitle}
          onChange={on => setOrbEnabled(on)}
        />
        <ListRow
          below={
            <div className="mt-3 flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <OrbView
                  aria-hidden="true"
                  className="size-16 shrink-0 rounded-full"
                  fallbackType="original-thinking"
                  params={params}
                />
                <input
                  aria-label={copy.urlTitle}
                  className="w-full rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-3 py-1.5 text-[length:var(--conversation-caption-font-size)] outline-none placeholder:text-(--ui-text-tertiary) focus:border-(--ui-stroke-secondary)"
                  onBlur={commitDraft}
                  onChange={event => {
                    setDraft(event.target.value)
                  }}
                  onKeyDown={event => {
                    if (event.key === 'Enter') {commitDraft()}
                  }}
                  placeholder={copy.urlPlaceholder}
                  spellCheck={false}
                  value={draft}
                />
              </div>
              {showError && (
                <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-red)">
                  {showError === 'not-an-orb-url' ? copy.urlErrorNotOrb : copy.urlErrorNoParams}
                </p>
              )}
              {configUrl && (
                <div>
                  <Button
                    onClick={() => {
                      setDraft('')
                      setOrbConfigUrl('')
                      triggerHaptic('selection')
                    }}
                    size="inline"
                    variant="text"
                  >
                    {copy.reset}
                  </Button>
                </div>
              )}
            </div>
          }
          className={cn(!enabled && 'opacity-60')}
          description={copy.urlDesc}
          title={copy.urlTitle}
        />
      </div>
    </div>
  )
}

function SparklesIcon({ className }: { className?: string }) {
  return (
    <svg aria-hidden="true" className={className} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
      <path
        d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3zM19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9L19 15z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

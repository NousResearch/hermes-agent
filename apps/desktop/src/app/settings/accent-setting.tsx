import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Chip } from '@/onboarding/visuals/chip'
import { AccentSwatch } from '@/onboarding/visuals/options'
import { accentsFor, customAccentId, resolveAccent } from '@/themes/accents'
import { useTheme } from '@/themes/context'

import { ListRow } from './primitives'

/**
 * The live profile's accent pick. Nous blue is the default skin's own accent,
 * so "Theme default" stands for it and it has no swatch of its own here.
 */
export function AccentSetting({ id }: { id: string }) {
  const { t } = useI18n()
  const copy = t.settings.appearance
  const { accent, renderedMode, setAccent, theme } = useTheme()
  const dark = renderedMode === 'dark'
  const swatches = accentsFor(dark).filter(swatch => swatch.id !== 'nous')
  const custom = accent?.startsWith('custom:') ? resolveAccent(accent, dark) : null

  const pick = (next: null | string) => {
    triggerHaptic('selection')
    setAccent(next)
  }

  return (
    <ListRow
      below={
        <div className="mt-3 flex flex-wrap items-center gap-2.5" role="group">
          <Chip label={copy.accentThemeDefault} on={accent === null} onToggle={() => pick(null)} variant="pill" />
          {swatches.map(swatch => (
            <AccentSwatch
              active={accent === swatch.id}
              hex={swatch.hex}
              key={swatch.id}
              name={swatch.name}
              onPick={() => pick(swatch.id)}
            />
          ))}
          <AccentSwatch
            active={custom !== null}
            hex={custom ?? theme.colors.primary}
            name={copy.accentCustom}
            onColorChange={hex => pick(customAccentId(hex))}
          />
        </div>
      }
      description={copy.accentDesc}
      id={id}
      title={copy.accentTitle}
      wide
    />
  )
}

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'

import type { DesignSelection } from './preview-design-in-page'

interface PreviewDesignLayersProps {
  onSelectLayer: (index: number) => void
  selection: DesignSelection | null
}

/** Inspect the selected element's DOM ancestry, never inferred source files. */
export function PreviewDesignLayers({ onSelectLayer, selection }: PreviewDesignLayersProps) {
  const { t } = useI18n()
  const copy = t.preview.design

  return (
    <section
      aria-label={copy.layers}
      className="flex h-48 max-h-[40%] min-h-24 shrink-0 flex-col border-t border-border/60 bg-background"
    >
      <header className="flex h-8 shrink-0 items-center gap-2 border-b border-border/50 px-2 text-xs font-medium text-foreground">
        <Codicon name="layers" size="0.8125rem" />
        <span>{copy.layers}</span>
        <span className="ml-auto truncate text-[0.6875rem] font-normal text-muted-foreground">{copy.domOnly}</span>
      </header>
      <div className="min-h-0 flex-1 overflow-auto">
        {selection?.layers.length ? (
          <table className="w-full table-fixed text-xs">
            <caption className="sr-only">{copy.selectedPath}</caption>
            <tbody>
              {selection.layers.map((layer, index) => (
                <tr key={index}>
                  <td>
                    <Button
                      aria-pressed={selection.selectedLayer === index}
                      className="w-full justify-start"
                      onClick={() => onSelectLayer(index)}
                      size="xs"
                      type="button"
                      variant={selection.selectedLayer === index ? 'secondary' : 'ghost'}
                    >
                      <span className="truncate" style={{ marginInlineStart: `${index * 0.75}rem` }}>
                        {layer.tag}
                      </span>
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="px-3 py-2 text-xs text-muted-foreground">{copy.empty}</p>
        )}
      </div>
    </section>
  )
}

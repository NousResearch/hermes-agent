/**
 * Catalog-content translations for toolsets. `name` (the id every toggle and API call uses) is
 * never changed; `label`/`description` show the translation and the original stays searchable.
 */
import { type ContentDictionary, contentKeys, contentSearchText, hasContentDictionary, lookupContent } from '@/i18n/catalog-content'
import type { ToolsetInfo } from '@/types/hermes'

export type LocalizedToolset = ToolsetInfo & { search?: string; originalLabel?: string }

export function localizeToolsets(toolsets: ToolsetInfo[], locale: string, overlay?: ContentDictionary | null): LocalizedToolset[] {
  if (!hasContentDictionary(locale) && !overlay) {
    return toolsets
  }

  return toolsets.map(toolset => {
    const text = lookupContent('toolset', contentKeys(toolset.name), locale, overlay)

    if (!text) {
      return toolset
    }

    const toolText = toolset.tools.map(tool => contentSearchText(lookupContent('tool', contentKeys(tool), locale, overlay)))

    return {
      ...toolset,
      label: text.name,
      description: text.description ?? toolset.description,
      originalLabel: toolset.label || toolset.name,
      search: [contentSearchText(text), String(toolset.label ?? '').toLowerCase(), String(toolset.description ?? '').toLowerCase(), ...toolText].join(' ')
    }
  })
}

/** Russian title for one tool (chips), or null: the tool name itself stays the id. */
export const toolTitle = (tool: string, locale: string, overlay?: ContentDictionary | null) =>
  lookupContent('tool', contentKeys(tool), locale, overlay)?.name ?? null

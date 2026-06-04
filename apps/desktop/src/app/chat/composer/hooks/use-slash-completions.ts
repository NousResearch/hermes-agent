import type { Unstable_TriggerAdapter, Unstable_TriggerItem } from '@assistant-ui/core'
import { useCallback } from 'react'

import type { HermesGateway } from '@/hermes'
import {
  type CommandsCatalogLike,
  desktopSlashDescription,
  filterDesktopCommandsCatalog,
  isDesktopSlashSuggestion
} from '@/lib/desktop-slash-commands'

import type { CompletionEntry, CompletionPayload } from './use-live-completion-adapter'
import { useLiveCompletionAdapter } from './use-live-completion-adapter'

interface SlashItemMetadata extends Record<string, string> {
  command: string
  display: string
  meta: string
}

function textValue(value: unknown, fallback = ''): string {
  if (typeof value === 'string') {
    return value
  }

  if (Array.isArray(value)) {
    return value
      .map(part => (Array.isArray(part) ? String(part[1] ?? '') : typeof part === 'string' ? part : ''))
      .join('')
      .trim()
  }

  return fallback
}

function commandText(value: string): string {
  return value.startsWith('/') ? value : `/${value}`
}

/** Live `/` completions backed by the gateway's `complete.slash` RPC. */
export function useSlashCompletions(options: { gateway: HermesGateway | null }): {
  adapter: Unstable_TriggerAdapter
  loading: boolean
} {
  const { gateway } = options
  const enabled = Boolean(gateway)

  const fetcher = useCallback(
    async (query: string): Promise<CompletionPayload> => {
      if (!gateway) {
        return { items: [], query }
      }

      const text = `/${query}`

      try {
        if (!query) {
          const catalog = filterDesktopCommandsCatalog(await gateway.request<CommandsCatalogLike>('commands.catalog'))

          const items = (catalog.pairs ?? []).map(([command, meta]) => ({
            text: command,
            display: command,
            meta
          }))

          return { items, query }
        }

        const result = await gateway.request<{ items?: CompletionEntry[] }>('complete.slash', { text })

        const items = (result.items ?? [])
          .filter(item => isDesktopSlashSuggestion(item.text))
          .map(item => ({
            ...item,
            meta: desktopSlashDescription(item.text, textValue(item.meta))
          }))

        return { items, query }
      } catch {
        return { items: [], query }
      }
    },
    [gateway]
  )

  const toItem = useCallback((entry: CompletionEntry, index: number): Unstable_TriggerItem => {
    const command = commandText(entry.text)
    const display = textValue(entry.display, commandText(entry.text))
    const meta = textValue(entry.meta)

    const metadata: SlashItemMetadata = {
      command,
      display,
      meta
    }

    return {
      id: `${entry.text}|${index}`,
      type: 'slash',
      label: display.startsWith('/') ? display.slice(1) : display,
      ...(meta ? { description: meta } : {}),
      metadata
    }
  }, [])

  return useLiveCompletionAdapter({ enabled, fetcher, toItem })
}

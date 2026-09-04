import type { Unstable_TriggerAdapter, Unstable_TriggerItem } from '@assistant-ui/core'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect } from 'react'

import type { HermesGateway } from '@/hermes'
import { sessionTitle } from '@/lib/chat-runtime'
import {
  canonicalDesktopSlashCommand,
  type CommandsCatalogLike,
  desktopSkinSlashCompletions,
  desktopSlashDescription,
  type DesktopThemeCommandOption,
  filterDesktopCommandsCatalog,
  isDesktopSlashExtensionCommand,
  isDesktopSlashSuggestion,
  rankSkillCommands,
  slashCompletionGroup
} from '@/lib/desktop-slash-commands'
import { $slashCompletionsEpoch, cachedSlashCompletion, hasCachedSlashCompletion } from '@/lib/slash-completion-cache'
import { normalize } from '@/lib/text'
import { $sessions } from '@/store/session'

import type { CompletionEntry, CompletionPayload } from './use-live-completion-adapter'
import { useLiveCompletionAdapter } from './use-live-completion-adapter'

interface SlashItemMetadata extends Record<string, string> {
  command: string
  display: string
  meta: string
  group: string
  rawText: string
  /** Completion-action id; empty for ordinary insert-a-chip completions. */
  action: string
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

export function canonicalizeSlashCommandCompletions(
  items: readonly CompletionEntry[],
  query: string
): CompletionEntry[] {
  const canonicalItems = new Map<string, CompletionEntry>()
  const normalizedQuery = commandText(query).toLowerCase()

  for (const item of items) {
    const command = commandText(item.text).trim()
    const normalizedCommand = command.toLowerCase()
    const canonical = canonicalDesktopSlashCommand(command)
    const matchedAlias =
      !canonical.startsWith(normalizedQuery) &&
      normalizedCommand !== canonical &&
      normalizedCommand.startsWith(normalizedQuery)
        ? normalizedCommand
        : null

    const canonicalItem = {
      ...item,
      text: canonical,
      display: matchedAlias ? `${canonical} (${matchedAlias.slice(1)})` : canonical
    }

    // A command token resolves to one canonical command in the registry/catalog,
    // so duplicate canonical and alias rows share this key. Prefer the labelled
    // alias row only when the alias itself — not the canonical token — matched.
    if (!canonicalItems.has(canonical) || matchedAlias) {
      canonicalItems.set(canonical, canonicalItem)
    }
  }

  return [...canonicalItems.values()]
}

/** How many recent sessions to surface inline before the "Browse all…" entry. */
const SESSION_INLINE_LIMIT = 7

/** Briefly prefer catalog-backed alias labels without blocking ordinary rows. */
const CATALOG_ALIAS_GRACE_MS = 200

function waitForCatalogAliasGrace(catalogReady: Promise<void>): Promise<void> {
  return new Promise(resolve => {
    const timer = globalThis.setTimeout(resolve, CATALOG_ALIAS_GRACE_MS)
    void catalogReady.then(() => {
      globalThis.clearTimeout(timer)
      resolve()
    })
  })
}

/** Live `/` completions backed by the gateway's `complete.slash` RPC. */
export function useSlashCompletions(options: {
  gateway: HermesGateway | null
  /** Desktop theme list — `/skin` is owned client-side, so its arg completions
   *  come from here, not the backend (whose skin list is CLI/TUI-only). */
  skinThemes?: DesktopThemeCommandOption[]
  activeSkin?: string
}): {
  adapter: Unstable_TriggerAdapter
  loading: boolean
} {
  const { gateway, skinThemes, activeSkin } = options
  const enabled = Boolean(gateway)
  const epoch = useStore($slashCompletionsEpoch)

  // Warm argument_mode before the first `/` so Space treats /review as text.
  useEffect(() => {
    if (!gateway) {
      return
    }

    void cachedSlashCompletion('catalog', () => gateway.request<CommandsCatalogLike>('commands.catalog'))
      .then(catalog => {
        filterDesktopCommandsCatalog(catalog)
      })
      .catch(() => {
        // Next keystroke retries; don't block the composer on a warm-up miss.
      })
  }, [gateway, epoch])

  const fetcher = useCallback(
    async (query: string): Promise<CompletionPayload> => {
      if (!gateway) {
        return { items: [], query }
      }

      const text = `/${query}`

      // The desktop owns /skin entirely (client-side theme context). Surface its
      // theme list inside this single popover instead of a bespoke one, and skip
      // the backend skin completions (which describe CLI/TUI skins that don't
      // apply here). Matches once we're past `/skin ` into the arg stage.
      const skinArg = /^\/skin\s+(.*)$/is.exec(text)

      if (skinArg && skinThemes) {
        const items = desktopSkinSlashCompletions(skinThemes, activeSkin ?? '', skinArg[1] ?? '').map(entry => ({
          text: entry.text,
          display: entry.display,
          meta: entry.meta,
          group: 'Themes'
        }))

        return { items, query }
      }

      // /resume (and its aliases) completes recent sessions inline — the same
      // client-side list the picker overlay shows — instead of the backend
      // (whose /resume opens an interactive TUI picker we can't render here).
      const sessionArg = /^\/(?:resume|sessions|switch)\s+(.*)$/is.exec(text)

      if (sessionArg) {
        const needle = normalize(sessionArg[1])

        const matches = (
          needle
            ? $sessions
                .get()
                .filter(
                  session =>
                    sessionTitle(session).toLowerCase().includes(needle) ||
                    (session.preview ?? '').toLowerCase().includes(needle) ||
                    session.id.toLowerCase().includes(needle)
                )
            : $sessions.get()
        ).slice(0, SESSION_INLINE_LIMIT)

        const items: CompletionEntry[] = matches.map(session => ({
          text: `/resume ${session.id}`,
          display: sessionTitle(session),
          meta: (session.preview ?? '').trim(),
          group: 'Sessions'
        }))

        // Trailing "more" affordance (Cursor-style): picking it opens the full
        // session picker overlay directly. `text` stays a bare `/resume` so that
        // submitting it (Enter) still opens the overlay if the action is skipped.
        items.push({
          text: '/resume',
          display: 'Browse all sessions…',
          meta: '',
          group: 'Sessions',
          action: 'session-picker'
        })

        return { items, query }
      }

      try {
        if (!query) {
          const catalog = filterDesktopCommandsCatalog(
            await cachedSlashCompletion('catalog', () => gateway.request<CommandsCatalogLike>('commands.catalog'))
          )

          // Prefer the categorized layout so the popover renders section headers
          // (Session, Tools & Skills, ...). Fall back to the flat list when the
          // backend didn't categorize.
          const sections = catalog.categories?.length ? catalog.categories : [{ name: '', pairs: catalog.pairs ?? [] }]

          const items = sections.flatMap<CompletionEntry>(section =>
            section.pairs.map(([command, meta]) => ({
              text: command,
              display: command,
              group: section.name || undefined,
              meta
            }))
          )

          // Skill commands reach us only through the flat `pairs` list — the
          // backend categorizes registry commands but appends skills
          // uncategorized, so the categorized layout alone drops every skill
          // from the bare `/` list even though typing `/wo` offers them.
          // Re-add the leftovers under one Skills header (which also gives them
          // the skill pill accent and makes them offerable mid-message).
          const categorized = new Set(items.map(item => item.text.toLowerCase()))
          const skillRows: CompletionEntry[] = []

          for (const [command, meta] of catalog.pairs ?? []) {
            if (!categorized.has(command.toLowerCase()) && isDesktopSlashExtensionCommand(command)) {
              skillRows.push({ text: command, display: command, group: 'Skills', meta })
            }
          }

          // Browsing, not searching: rank the skills the user actually reaches
          // for to the top and drop never-used built-ins entirely. Typing a
          // query takes the other branch, where nothing is hidden.
          items.push(...rankSkillCommands(skillRows, catalog.skills, { pruneUnusedBuiltins: true }))

          return { items, query }
        }

        // Start the catalog and completion requests together. Dynamic aliases
        // live in the catalog, so command-row canonicalization gives that
        // shared warm-up a short grace period; otherwise a fast complete.slash
        // response can briefly render `/btw` instead of `/background (btw)`.
        // The grace is bounded so a stalled catalog cannot withhold ordinary
        // completions until the gateway's request timeout.
        const catalogReady = cachedSlashCompletion('catalog', () =>
          gateway.request<CommandsCatalogLike>('commands.catalog')
        )
          .then(catalog => {
            filterDesktopCommandsCatalog(catalog)
          })
          .catch(() => undefined)
        const result = await cachedSlashCompletion(`slash:${text.toLowerCase()}`, () =>
          gateway.request<{ items?: CompletionEntry[]; replace_from?: number }>('complete.slash', { text })
        )

        // Arg-completion items (replace_from > 1) carry just the arg stub —
        // e.g. complete.slash returns `{text: "alice"}` for `/personality alic`
        // with replace_from = 14. Rewrite those entries so the popover inserts
        // the full `/personality alice` token instead of stranding `/alice`.
        const replaceFrom = typeof result.replace_from === 'number' ? result.replace_from : 1
        const isArgCompletion = replaceFrom > 1
        const prefix = isArgCompletion ? text.slice(0, replaceFrom) : ''

        if (!isArgCompletion) {
          await waitForCatalogAliasGrace(catalogReady)
        }

        const completionItems = isArgCompletion
          ? (result.items ?? [])
          : canonicalizeSlashCommandCompletions(result.items ?? [], text)

        const decorated = completionItems
          .map(item => {
            if (!isArgCompletion) {
              return item
            }

            const argText = typeof item.text === 'string' ? item.text : ''

            return { ...item, text: `${prefix}${argText}` }
          })
          .filter(item => isArgCompletion || isDesktopSlashSuggestion(item.text))
          .map(item => ({
            ...item,
            // Arg suggestions (e.g. `/handoff <platform>`) live under one
            // header; otherwise split skills out from built-in commands.
            // Kind comes from the backend — the desktop table is a visibility
            // gate (`isDesktopSlashSuggestion`), not a classifier.
            group: isArgCompletion ? 'Options' : slashCompletionGroup(item.text, item.kind),
            // Arg items carry their own meta (the personality/toolset/platform
            // blurb). Only command rows get the registry description — looking
            // one up for `/personality none` would clobber it with the parent
            // command's text.
            meta: isArgCompletion ? textValue(item.meta) : desktopSlashDescription(item.text, textValue(item.meta))
          }))

        // Keep each group contiguous so headers render once: Commands before
        // Skills (stable within a group, preserving backend relevance order).
        // Do not re-sort skills by usage here — complete.slash already ranked
        // by fuzzy score, then usage. A second usage pass buried exact name
        // matches that the table had mis-filed as skills.
        const groupOrder = ['Commands', 'Skills', 'Options']

        if (isArgCompletion) {
          return { items: decorated, query }
        }

        const items = [...decorated].sort(
          (a, b) => groupOrder.indexOf(a.group ?? '') - groupOrder.indexOf(b.group ?? '')
        )

        return { items, query }
      } catch {
        return { items: [], query }
      }
    },
    [gateway, skinThemes, activeSkin]
  )

  const toItem = useCallback((entry: CompletionEntry, index: number): Unstable_TriggerItem => {
    const command = commandText(entry.text)
    const display = textValue(entry.display, commandText(entry.text))
    const meta = textValue(entry.meta)

    const metadata: SlashItemMetadata = {
      command,
      display,
      meta,
      group: textValue(entry.group),
      action: textValue(entry.action),
      // Provide rawText so hermesDirectiveFormatter.serialize uses the
      // direct-insertion path instead of the legacy @type:id fallback.
      // Without this, the item.id (which includes a "|index" suffix for
      // trigger-adapter uniqueness) leaks into the serialized chip text
      // and the submitted command.
      rawText: command
    }

    return {
      id: `${entry.text}|${index}`,
      type: 'slash',
      label: display.startsWith('/') ? display.slice(1) : display,
      ...(meta ? { description: meta } : {}),
      metadata
    }
  }, [])

  // Mirrors the fetcher's branching: the `/skin` and `/resume` arg stages are
  // answered from client-side state, so they never wait on the network; every
  // other query is served from the completion cache when it's still warm.
  const isCached = useCallback(
    (query: string) => {
      const text = `/${query}`

      if ((skinThemes && /^\/skin\s+/is.test(text)) || /^\/(?:resume|sessions|switch)\s+/is.test(text)) {
        return true
      }

      return hasCachedSlashCompletion(query ? `slash:${text.toLowerCase()}` : 'catalog')
    },
    [skinThemes]
  )

  return useLiveCompletionAdapter({ enabled, epoch, fetcher, isCached, toItem })
}

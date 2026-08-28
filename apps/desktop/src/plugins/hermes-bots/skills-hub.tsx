/**
 * The real skills hub page embedded as a picker, plus its offline search
 * fallback.
 *
 * A leaf: the advanced profile editor and the create dialog both mount it, and
 * it reaches back into neither.
 */

import { Button, host, Input } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { useBots } from './i18n'

// ── skills hub section: the REAL hub page (docs) embedded as a picker ──────
// https://hermes-agent.nousresearch.com/docs/skills?embed=picker hides the
// docs chrome and adds "+ Add to this Agent" per card, posting
// {type: 'hermes-skill-pick', ...} to us (hermes-agent#86243). We validate
// the origin, install via skills.manage, and bubble onInstalled so the
// checklist above gains the row. Search-box fallback kept for offline use.

const HUB_ORIGIN = 'https://hermes-agent.nousresearch.com'
const HUB_PICKER_URL = HUB_ORIGIN + '/docs/skills?embed=picker'
/** One `skills.manage action=search` hit. */
interface HubSkillResult {
  description?: string
  name: string
}
interface HubSkillsSectionProps {
  /** Install target: a bare profile name, a connection-scoped descriptor for a
   *  bot on another gateway, or null for the launch profile (create time). */
  forProfile: null | string | { connectionId?: null | string; profile?: null | string }
  onInstalled?: (name: string) => void
}

export function HubSkillsSection({ forProfile, onInstalled }: HubSkillsSectionProps) {
  const b = useBots()
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<HubSkillResult[] | null>(null)
  const [searching, setSearching] = useState(false)
  const [installing, setInstalling] = useState<null | string>(null)
  const [installed, setInstalled] = useState<Record<string, boolean>>({})
  const [browseHub, setBrowseHub] = useState(false)
  const installRef = useRef<((name: string, displayName?: string) => Promise<void>) | null>(null)
  const frameRef = useRef<HTMLIFrameElement | null>(null)

  // Picker messages from the embedded hub page. Origin- AND source-checked —
  // only OUR frame may ask for an install (the hub origin alone would let any
  // other window on it, e.g. an OAuth popup, trigger installs too); installs
  // route through the same install() the search fallback uses.
  useEffect(() => {
    if (!browseHub) {
      return undefined
    }

    const onMessage = (event: MessageEvent) => {
      if (event.origin !== HUB_ORIGIN) {
        return
      }

      if (!frameRef.current || event.source !== frameRef.current.contentWindow) {
        return
      }

      const data = event.data

      if (!data || data.type !== 'hermes-skill-pick' || !data.name) {
        return
      }

      const target = String(data.identifier || data.name)

      // Skill identifiers are slugs / owner-name paths — keep anything
      // else out of skills.manage.
      if (!/^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(target)) {
        return
      }

      if (installRef.current) {
        void installRef.current(target, String(data.name))
      }
    }

    window.addEventListener('message', onMessage)

    return () => window.removeEventListener('message', onMessage)
  }, [browseHub])

  const search = async () => {
    const q = query.trim()

    if (!q || searching) {
      return
    }

    setSearching(true)
    setResults(null)

    try {
      const res: { results?: HubSkillResult[] } = await host.request('skills.manage', {
        action: 'search',
        query: q
      })

      setResults(res.results || [])
    } catch {
      setResults([])
    } finally {
      setSearching(false)
    }
  }

  const install = async (name: string, displayName?: string) => {
    const label = displayName || name

    if (installing) {
      return
    }

    setInstalling(label)

    try {
      // With forProfile the install lands in that bot's skills dir
      // (gateway skills.manage profile scoping); null = launch profile,
      // which is right at create time — the new bot clones/copies from it.
      await host.request('skills.manage', {
        action: 'install',
        query: name,
        ...(forProfile
          ? {
              profile: forProfile
            }
          : {})
      })
      setInstalled(prev => ({
        ...prev,
        [label]: true
      }))
      host.notify({
        kind: 'success',
        message: `Skill "${label}" installed`
      })

      if (typeof onInstalled === 'function') {
        onInstalled(label)
      }
    } catch (err) {
      host.notifyError(err, `Installing "${label}" failed`)
    } finally {
      setInstalling(null)
    }
  }

  installRef.current = install

  return (
    <div className="grid gap-1.5 border-t border-(--ui-stroke-secondary) pt-2">
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-[0.7rem] font-medium text-(--ui-text-secondary)">Skills Hub</div>
        <Button
          className="text-[0.65rem] text-(--ui-text-quaternary) hover:text-(--ui-text-secondary)"
          onClick={() => setBrowseHub(v => !v)}
          size="inline"
          variant="text"
        >
          {browseHub ? 'hide the hub browser' : 'browse the full hub ▾'}
        </Button>
      </div>
      {browseHub ? (
        <div className="grid gap-1">
          {/* Resizable viewport: native CSS resize handle (bottom-right */
          /* corner) lets the user drag it larger/smaller. The iframe */
          /* inside is rendered oversized and scaled DOWN (133% × 0.75) */
          /* so the hub page starts zoomed out — we can't style the */
          /* cross-origin page itself, but scaling the frame is ours. */}
          <div
            style={{
              width: '100%',
              height: 560,
              minHeight: 240,
              minWidth: 320,
              maxWidth: '100%',
              resize: 'both',
              overflow: 'hidden',
              border: '1px solid var(--ui-stroke-secondary)',
              borderRadius: 8,
              position: 'relative'
            }}
          >
            <iframe
              ref={frameRef}
              sandbox="allow-scripts allow-same-origin"
              src={HUB_PICKER_URL}
              style={{
                width: '133.34%',
                height: '133.34%',
                border: 'none',
                background: 'transparent',
                transform: 'scale(0.75)',
                transformOrigin: 'top left'
              }}
              title={b.tools.skillsHub}
            />
          </div>
          <div className="px-1 text-[0.65rem] leading-4 text-(--ui-text-quaternary)">
            {installing
              ? `Installing "${installing}"…`
              : 'Hit "+ Add to this Agent" on any skill — it installs and appears in the list above. Drag the corner to resize.'}
          </div>
        </div>
      ) : null}
      <div className="flex gap-1.5">
        <Input
          className="h-7 flex-1 text-xs"
          onChange={event => setQuery(event.target.value)}
          onKeyDown={event => {
            // IME guard: Enter confirming a composed word must not search.
            if (event.nativeEvent?.isComposing || event.keyCode === 229) {
              return
            }

            if (event.key === 'Enter') {
              event.preventDefault()
              void search()
            }
          }}
          placeholder={b.tools.searchHub}
          value={query}
        />
        <Button disabled={searching || !query.trim()} onClick={() => void search()} size="sm" variant="secondary">
          {searching ? 'Searching…' : 'Search'}
        </Button>
      </div>
      {searching ? (
        <div className="px-1 text-[0.65rem] text-(--ui-text-quaternary)">
          Searching community + well-known sources — can take ~10s…
        </div>
      ) : null}
      {results === null ? null : results.length === 0 ? (
        <div className="px-1 py-1.5 text-[0.7rem] text-(--ui-text-quaternary)">No hub skills matched.</div>
      ) : (
        <div
          className="overflow-y-auto overscroll-contain"
          style={{
            maxHeight: 150
          }}
        >
          <div className="grid gap-1">
            {results.map(r => (
              <div className="flex items-center gap-2 text-xs" key={r.name}>
                <div className="min-w-0 flex-1">
                  <div className="truncate font-medium">{r.name}</div>
                  {r.description ? (
                    <div className="truncate text-[0.65rem] text-(--ui-text-quaternary)">{r.description}</div>
                  ) : null}
                </div>
                {installed[r.name] ? (
                  <span className="shrink-0 text-[0.65rem] text-(--ui-text-tertiary)">✓ added</span>
                ) : (
                  <Button
                    className="shrink-0 px-2 font-semibold"
                    disabled={installing !== null}
                    onClick={() => void install(r.name)}
                    size="sm"
                    title={`Install "${r.name}" and add it to the list above`}
                    variant="ghost"
                  >
                    {installing === r.name ? '…' : '+'}
                  </Button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

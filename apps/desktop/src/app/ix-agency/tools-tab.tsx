import { useCallback, useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { normalize } from '@/lib/text'

import { DetailColumn, ListColumn, ListStrip, MasterDetail } from '../master-detail'
import { PanelEmpty, PanelListRow, PanelMeta, PanelSectionLabel } from '../overlays/panel'

import mcpTilesData from './data/mcp-tiles.json'
import type { IxMcpTileItem } from './types'

// Bundled admin-mcp registry snapshot; the Refresh action swaps in the live
// directory from the gateway when a token is configured (Connect tab).
const BUNDLED_TILES: IxMcpTileItem[] = Array.isArray(mcpTilesData.items) ? (mcpTilesData.items as IxMcpTileItem[]) : []

const BUNDLED_DETAIL = 'Bundled registry snapshot — configure the gateway token in Connect and refresh for live data.'

export function ToolsTab({ query }: { query: string }) {
  const [tiles, setTiles] = useState<IxMcpTileItem[]>(BUNDLED_TILES)
  const [detail, setDetail] = useState(BUNDLED_DETAIL)
  const [busy, setBusy] = useState(false)
  const [selectedId, setSelectedId] = useState<null | string>(null)

  const refresh = useCallback(async () => {
    const bridge = window.hermesDesktop?.ixAgency

    if (!bridge) {
      return
    }

    setBusy(true)

    try {
      const live = await bridge.listMcpTiles()
      setTiles(live.tiles.length > 0 ? live.tiles : BUNDLED_TILES)
      setDetail(live.detail)
    } catch (error) {
      setDetail(`Gateway unavailable — showing bundled snapshot. ${error instanceof Error ? error.message : ''}`.trim())
    } finally {
      setBusy(false)
    }
  }, [])

  // Try the live directory once on mount; failures quietly keep the snapshot.
  useEffect(() => {
    void refresh()
  }, [refresh])

  const q = normalize(query)

  const filtered = tiles.filter(
    tile => !q || normalize(`${tile.id} ${tile.label} ${tile.blurb ?? ''} ${tile.group ?? ''}`).includes(q)
  )

  const groups = [...new Set(filtered.map(tile => tile.group || 'Other'))]

  const selected = tiles.find(tile => tile.id === selectedId) ?? null

  return (
    <MasterDetail split="wide">
      <ListColumn
        header={
          <ListStrip
            left={<span className="truncate text-[0.65rem] text-muted-foreground/60">{detail}</span>}
            right={
              <Button disabled={busy} onClick={() => void refresh()} size="icon-xs" title="Refresh" variant="ghost">
                <Codicon name={busy ? 'loading~spin' : 'refresh'} size="0.8125rem" />
              </Button>
            }
          />
        }
      >
        {groups.map(group => (
          <div key={group}>
            <PanelSectionLabel className="px-2 pb-0.5 pt-2">{group}</PanelSectionLabel>
            {filtered
              .filter(tile => (tile.group || 'Other') === group)
              .map(tile => (
                <PanelListRow
                  active={tile.id === selectedId}
                  icon="plug"
                  key={tile.id}
                  meta={tile.hasDefaultToken ? 'key ✓' : undefined}
                  onSelect={() => setSelectedId(tile.id)}
                  title={tile.label}
                />
              ))}
          </div>
        ))}
      </ListColumn>
      <DetailColumn>
        {selected ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <h3 className="min-w-0 flex-1 truncate text-sm font-semibold text-foreground">{selected.label}</h3>
              {selected.group && <Badge variant="muted">{selected.group}</Badge>}
            </div>
            {selected.blurb && <p className="text-xs leading-relaxed text-muted-foreground">{selected.blurb}</p>}
            <PanelMeta
              rows={[
                { label: 'Tile id', value: <code className="font-mono text-[0.68rem]">{selected.id}</code> },
                {
                  label: 'MCP URL',
                  value: <code className="break-all font-mono text-[0.68rem]">{selected.mcpUrl}</code>
                },
                ...(selected.mcpAuthHint ? [{ label: 'Auth', value: selected.mcpAuthHint }] : [])
              ]}
            />
            <p className="text-[0.68rem] leading-relaxed text-muted-foreground/70">
              Add this MCP server to the agent via Capabilities → MCP, or reach every tile at once through the
              admin-mcp gateway (Connect tab).
            </p>
          </div>
        ) : (
          <PanelEmpty
            description="Every MCP tool the org's admin-mcp gateway fans out to — email, WhatsApp, calls, socials, analytics, commerce. Select a tile for its endpoint and auth."
            icon="plug"
            title="Org MCP tools"
          />
        )}
      </DetailColumn>
    </MasterDetail>
  )
}

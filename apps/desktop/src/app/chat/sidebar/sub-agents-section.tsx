import { useEffect, useState } from 'react'

import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { SidebarGroup, SidebarGroupContent } from '@/components/ui/sidebar'
import { listSubAgents, type SubAgentPeer } from '@/hermes'

import { SidebarPanelLabel } from '../../shell/sidebar-label'

import { SubAgentChatDialog } from './sub-agent-chat-dialog'

// 下级在线状态由后台广播刷新(无 UI 信号),轮询让上/下线几秒内反映。
const POLL_MS = 10_000

/**
 * 侧栏「子agent」分区:列出局域网自发现到的在线下级节点,点开走轻量多轮对话。
 *
 * 自取数据 + 自管开合(不往父组件穿 props),挂载只需一行。没发现任何下级时整块隐藏。
 */
export function SidebarSubAgentsSection() {
  const [open, setOpen] = useState(false)
  const [peers, setPeers] = useState<SubAgentPeer[]>([])
  const [chatPeer, setChatPeer] = useState<null | SubAgentPeer>(null)

  useEffect(() => {
    let cancelled = false

    const load = () =>
      listSubAgents()
        .then(p => {
          if (!cancelled) {
            setPeers(p)
          }
        })
        .catch(() => {})

    void load()

    const id = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void load()
      }
    }, POLL_MS)

    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  if (peers.length === 0) {
    return null
  }

  return (
    <SidebarGroup className="shrink-0 p-0 pb-1">
      <div className="group/section flex shrink-0 items-center justify-between pb-1 pt-1.5">
        <button
          className="group/section-label flex w-fit items-center gap-1 bg-transparent text-left leading-none"
          onClick={() => setOpen(o => !o)}
          type="button"
        >
          <SidebarPanelLabel>子agent</SidebarPanelLabel>
          <span className="text-[0.6875rem] font-medium text-(--ui-text-quaternary)">{peers.length}</span>
          <DisclosureCaret
            className="text-(--ui-text-tertiary) opacity-0 transition group-hover/section-label:opacity-100"
            open={open}
          />
        </button>
      </div>
      {open && (
        <SidebarGroupContent className="flex max-h-72 flex-col gap-px overflow-x-hidden overflow-y-auto overscroll-contain pb-1.75">
          {peers.map(p => (
            <button
              className="group/sa flex min-h-[1.625rem] min-w-0 items-center gap-1.5 rounded-md py-0.5 pl-2 pr-1 text-left hover:bg-(--chrome-action-hover)"
              key={p.uid}
              onClick={() => setChatPeer(p)}
              title={p.name || p.uid}
              type="button"
            >
              <span className="grid w-3.5 shrink-0 place-items-center">
                <span aria-hidden className="size-1.5 rounded-full bg-emerald-500/70" />
              </span>
              <span className="min-w-0 truncate text-[0.8125rem] text-(--ui-text-secondary) group-hover/sa:text-foreground">
                {p.name || p.uid}
              </span>
            </button>
          ))}
        </SidebarGroupContent>
      )}
      <SubAgentChatDialog
        onOpenChange={o => {
          if (!o) {
            setChatPeer(null)
          }
        }}
        open={chatPeer !== null}
        peer={chatPeer}
      />
    </SidebarGroup>
  )
}

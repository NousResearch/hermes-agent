import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import {
  type AgentEmployeeInfo,
  getAiEmployees,
  getProfileSoul,
  updateAiEmployeeMetadata,
  updateProfileSoul
} from '@/hermes'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { $newChatProfile, requestFreshSession } from '@/store/profile'

import { NEW_CHAT_ROUTE } from '../routes'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

interface AiEmployeesViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

type EmployeeTab = 'overview' | 'training' | 'capabilities' | 'try'

interface TrainingDraft {
  category: string
  displayName: string
  emoji: string
  mission: string
  role: string
  soul: string
}

function draftFromEmployee(employee: AgentEmployeeInfo, soul = ''): TrainingDraft {
  return {
    category: employee.category,
    displayName: employee.display_name_zh,
    emoji: employee.emoji,
    mission: employee.mission_zh,
    role: employee.role_zh,
    soul
  }
}

function modelLabel(employee: AgentEmployeeInfo): string {
  return [employee.provider, employee.model].filter(Boolean).join(' / ') || '未配置'
}

function EmployeePill({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-card-background) px-3 py-2">
      <div className="text-[0.66rem] font-medium uppercase tracking-wider text-(--ui-text-tertiary)">{label}</div>
      <div className="mt-1 min-w-0 break-words text-sm text-(--ui-text-primary)">{value}</div>
    </div>
  )
}

export function AiEmployeesView({ className, setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: AiEmployeesViewProps) {
  const navigate = useNavigate()
  const [employees, setEmployees] = useState<AgentEmployeeInfo[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [tab, setTab] = useState<EmployeeTab>('overview')
  const [draft, setDraft] = useState<TrainingDraft | null>(null)

  const selected = useMemo(
    () => employees.find(employee => employee.profile_id === selectedId) ?? employees[0] ?? null,
    [employees, selectedId]
  )

  const refresh = useCallback(async () => {
    setLoading(true)

    try {
      const response = await getAiEmployees()
      const list = [...response.agents].sort((a, b) => a.sort_order - b.sort_order || a.profile_id.localeCompare(b.profile_id))

      setEmployees(list)
      setSelectedId(current => (current && list.some(employee => employee.profile_id === current) ? current : list[0]?.profile_id ?? null))
    } catch (err) {
      notifyError(err, 'AI 员工加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  useEffect(() => {
    if (!selected) {
      setDraft(null)

      return
    }

    let cancelled = false
    setDraft(draftFromEmployee(selected))
    void getProfileSoul(selected.profile_id)
      .then(soul => {
        if (!cancelled) {
          setDraft(draftFromEmployee(selected, soul.content || ''))
        }
      })
      .catch(err => {
        if (!cancelled) {
          notifyError(err, 'SOUL.md 加载失败')
          setDraft(draftFromEmployee(selected))
        }
      })

    return () => {
      cancelled = true
    }
  }, [selected])

  const updateDraft = useCallback((key: keyof TrainingDraft, value: string) => {
    setDraft(current => (current ? { ...current, [key]: value } : current))
  }, [])

  const saveTraining = useCallback(async () => {
    if (!selected || !draft) {
      return
    }

    setSaving(true)

    try {
      await updateAiEmployeeMetadata(selected.profile_id, {
        category: draft.category,
        display_name_zh: draft.displayName,
        emoji: draft.emoji,
        mission_zh: draft.mission,
        role_zh: draft.role
      })
      await updateProfileSoul(selected.profile_id, draft.soul)
      notify({ kind: 'success', title: '训练已保存', message: '新的身份和 SOUL.md 会在新会话中生效。' })
      await refresh()
    } catch (err) {
      notifyError(err, '保存训练失败')
    } finally {
      setSaving(false)
    }
  }, [draft, refresh, selected])

  const startEmployeeChat = useCallback(() => {
    if (!selected) {
      return
    }

    $newChatProfile.set(selected.profile_id)
    requestFreshSession()
    navigate(NEW_CHAT_ROUTE)
  }, [navigate, selected])

  return (
    <section
      className={cn(
        'flex h-full min-h-0 flex-col overflow-hidden bg-(--ui-chat-surface-background) text-(--ui-text-primary)',
        className
      )}
      {...props}
    >
      <header className="shrink-0 border-b border-(--ui-stroke-tertiary) px-6 py-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-lg font-semibold">
              <Codicon className="text-(--ui-text-tertiary)" name="organization" />
              AI 员工
            </div>
            <p className="mt-1 text-sm text-(--ui-text-secondary)">
              每个员工都是一个 Hermes Agent/Profile：中文名负责显示，profile id 负责系统调度。
            </p>
          </div>
          <Button onClick={() => void refresh()} size="sm" variant="secondary">
            刷新员工
          </Button>
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(220px,320px)_1fr] overflow-hidden">
        <aside className="min-h-0 border-r border-(--ui-stroke-tertiary) bg-(--ui-sidebar-surface-background) p-3">
          <div className="mb-3 px-2 text-xs text-(--ui-text-tertiary)">
            {loading ? '正在加载员工…' : `${employees.length} 个员工`}
          </div>
          <div className="flex min-h-0 flex-col gap-1 overflow-y-auto pr-1">
            {employees.map(employee => {
              const active = employee.profile_id === selected?.profile_id

              return (
                <button
                  className={cn(
                    'flex w-full items-center gap-3 rounded-xl border border-transparent px-3 py-2.5 text-left transition-colors hover:bg-(--ui-control-hover-background)',
                    active && 'border-(--ui-stroke-tertiary) bg-(--ui-control-active-background)'
                  )}
                  key={employee.profile_id}
                  onClick={() => {
                    setSelectedId(employee.profile_id)
                    setTab('overview')
                  }}
                  type="button"
                >
                  <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-(--ui-card-background) text-lg">
                    {employee.emoji || '🤖'}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium text-(--ui-text-primary)">{employee.display_name_zh}</span>
                    <span className="block truncate font-mono text-[0.68rem] text-(--ui-text-tertiary)">{employee.profile_id}</span>
                  </span>
                </button>
              )
            })}
          </div>
        </aside>

        <main className="min-h-0 overflow-y-auto p-6">
          {!selected ? (
            <div className="grid h-full place-items-center text-sm text-(--ui-text-secondary)">
              {loading ? '正在加载 AI 员工…' : '还没有 AI 员工'}
            </div>
          ) : (
            <div className="mx-auto flex max-w-5xl flex-col gap-5">
              <section className="rounded-2xl border border-(--ui-stroke-tertiary) bg-(--ui-card-background) p-5 shadow-sm">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="flex min-w-0 items-center gap-4">
                    <div className="grid size-14 shrink-0 place-items-center rounded-2xl bg-(--ui-control-active-background) text-3xl">
                      {selected.emoji || '🤖'}
                    </div>
                    <div className="min-w-0">
                      <h1 className="truncate text-2xl font-semibold">{selected.display_name_zh}</h1>
                      <div className="mt-1 font-mono text-xs text-(--ui-text-tertiary)">{selected.profile_id}</div>
                      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-(--ui-text-secondary)">{selected.mission_zh}</p>
                    </div>
                  </div>
                  <Button onClick={startEmployeeChat}>用这个员工新建会话</Button>
                </div>
              </section>

              <div className="flex flex-wrap gap-2">
                {[
                  ['overview', '概览'],
                  ['training', '训练员工'],
                  ['capabilities', '技能与能力'],
                  ['try', '试用员工']
                ].map(([id, label]) => (
                  <Button
                    aria-pressed={tab === id}
                    key={id}
                    onClick={() => setTab(id as EmployeeTab)}
                    size="sm"
                    variant={tab === id ? 'default' : 'secondary'}
                  >
                    {label}
                  </Button>
                ))}
              </div>

              {tab === 'overview' && (
                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                  <EmployeePill label="岗位" value={selected.role_zh || '未填写'} />
                  <EmployeePill label="模型" value={modelLabel(selected)} />
                  <EmployeePill label="技能数" value={`${selected.skill_count}`} />
                  <EmployeePill label="分类" value={selected.category} />
                  <EmployeePill label="Gateway" value={selected.gateway_running ? '运行中' : '未运行'} />
                  <EmployeePill label="SOUL.md" value={<span className="font-mono text-xs">{selected.soul_path}</span>} />
                </div>
              )}

              {tab === 'training' && draft && (
                <section className="grid gap-4 rounded-2xl border border-(--ui-stroke-tertiary) bg-(--ui-card-background) p-5">
                  <div className="rounded-xl border border-amber-400/30 bg-amber-500/10 px-3 py-2 text-xs leading-relaxed text-amber-700 dark:text-amber-200">
                    训练会写入 registry、profile.yaml 和 SOUL.md；由于提示词缓存，通常需要新会话才会完全生效。
                  </div>
                  <div className="grid gap-4 md:grid-cols-2">
                    <label className="grid gap-1 text-sm font-medium">
                      中文显示名
                      <Input onChange={event => updateDraft('displayName', event.target.value)} value={draft.displayName} />
                    </label>
                    <label className="grid gap-1 text-sm font-medium">
                      岗位
                      <Input onChange={event => updateDraft('role', event.target.value)} value={draft.role} />
                    </label>
                    <label className="grid gap-1 text-sm font-medium">
                      分类
                      <Input onChange={event => updateDraft('category', event.target.value)} value={draft.category} />
                    </label>
                    <label className="grid gap-1 text-sm font-medium">
                      Emoji
                      <Input onChange={event => updateDraft('emoji', event.target.value)} value={draft.emoji} />
                    </label>
                  </div>
                  <label className="grid gap-1 text-sm font-medium">
                    任务说明
                    <Textarea onChange={event => updateDraft('mission', event.target.value)} rows={3} value={draft.mission} />
                  </label>
                  <label className="grid gap-1 text-sm font-medium">
                    SOUL.md
                    <Textarea
                      className="font-mono text-xs"
                      onChange={event => updateDraft('soul', event.target.value)}
                      rows={12}
                      value={draft.soul}
                    />
                  </label>
                  <div className="flex justify-end">
                    <Button disabled={saving} onClick={() => void saveTraining()}>
                      {saving ? '保存中…' : '保存训练'}
                    </Button>
                  </div>
                </section>
              )}

              {tab === 'capabilities' && (
                <section className="grid gap-3 rounded-2xl border border-(--ui-stroke-tertiary) bg-(--ui-card-background) p-5 text-sm text-(--ui-text-secondary)">
                  <p>当前能力摘要：{selected.skill_count} 个技能，模型 {modelLabel(selected)}。</p>
                  <p>下一阶段会在这里接入每个员工的技能包、工具集、模型选择和训练预设。</p>
                </section>
              )}

              {tab === 'try' && (
                <section className="grid gap-3 rounded-2xl border border-(--ui-stroke-tertiary) bg-(--ui-card-background) p-5">
                  <p className="text-sm text-(--ui-text-secondary)">
                    先用该员工开启新会话；后续可以接入 Kanban，把当前任务直接派给员工执行。
                  </p>
                  <div>
                    <Button onClick={startEmployeeChat}>用这个员工新建会话</Button>
                  </div>
                </section>
              )}
            </div>
          )}
        </main>
      </div>
    </section>
  )
}

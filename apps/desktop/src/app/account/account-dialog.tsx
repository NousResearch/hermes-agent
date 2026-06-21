import { useStore } from '@nanostores/react'
import { type FormEvent, useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import type { DesktopAccountStatus } from '@/global'
import { Loader2, LogIn, Sparkles, X } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $accountDialogOpen, closeAccountDialog, setAccountDialogOpen } from '@/store/account'

import { PermissionsSection, ProfileSection, RechargeSection, RolesSection, TeamSection, UsageSection } from './index'

const DEFAULT_CLOUD_URL =
  (import.meta.env?.VITE_KARI_CLOUD_URL as string | undefined)?.trim() || 'https://lotjc.com/hermes'

function notifyAccountChanged() {
  window.dispatchEvent(new Event('hermes-account-changed'))
}

type Section = 'usage' | 'recharge' | 'profile' | 'team' | 'roles' | 'permissions'

const MENU: { key: Section; label: string; icon: string }[] = [
  { key: 'usage', label: '消费明细', icon: 'history' },
  { key: 'recharge', label: '充值', icon: 'credit-card' },
  { key: 'profile', label: '个人', icon: 'account' },
  { key: 'team', label: '团队账号', icon: 'type-hierarchy' },
  { key: 'roles', label: '角色管理', icon: 'tag' },
  { key: 'permissions', label: '权限管理', icon: 'key' }
]

export function AccountDialog() {
  const open = useStore($accountDialogOpen)
  const [status, setStatus] = useState<DesktopAccountStatus | null>(null)
  const [balance, setBalance] = useState<number | undefined>(undefined)
  const [cloudBaseUrl, setCloudBaseUrl] = useState(DEFAULT_CLOUD_URL)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showCloud, setShowCloud] = useState(false)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [section, setSection] = useState<Section>('usage')
  const [isSub, setIsSub] = useState(false)

  const refresh = useCallback(async () => {
    const account = window.hermesDesktop?.account

    if (!account?.status) {
      setLoading(false)
      setError('桌面账号服务不可用')

      return
    }

    setLoading(true)
    setError('')

    try {
      const next = await account.status()
      setStatus(next)
      setBalance(next.balance)

      if (next.email) {
        setEmail(next.email)
      }

      if (next.cloudBaseUrl) {
        setCloudBaseUrl(next.cloudBaseUrl)
      }

      // 子账号(parent_id 非空)只看 个人 + 自己的消费明细;主账号看全部。
      try {
        const me = await account.me?.()
        setIsSub(Boolean(me?.parent_id))
      } catch {
        setIsSub(false)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  // Load fresh state every time the popup opens.
  useEffect(() => {
    if (open) {
      void refresh()
    }
  }, [open, refresh])

  const submit = async (mode: 'login' | 'register', event?: FormEvent) => {
    event?.preventDefault()
    const account = window.hermesDesktop?.account
    const fn = mode === 'register' ? account?.register : account?.login

    if (!fn) {
      setError('桌面账号服务不可用')

      return
    }

    setBusy(true)
    setError('')

    try {
      const result = await fn({ cloudBaseUrl, email, password })

      if (!result.ok) {
        setError(result.error || (mode === 'register' ? '注册失败' : '登录失败'))

        return
      }

      setPassword('')
      notifyAccountChanged()
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const loggedIn = Boolean(status?.loggedIn)
  const name = status?.username || status?.email || '已登录'

  const allowedKeys: Section[] = isSub
    ? ['usage', 'profile']
    : ['usage', 'recharge', 'profile', 'team', 'roles', 'permissions']

  const menu = MENU.filter(item => allowedKeys.includes(item.key))
  const activeSection: Section = allowedKeys.includes(section) ? section : 'profile'

  return (
    <Dialog onOpenChange={setAccountDialogOpen} open={open}>
      <DialogContent
        className={cn(
          'gap-0 overflow-hidden p-0',
          loggedIn ? 'h-[min(82vh,42rem)] w-[min(94vw,62rem)] max-w-none' : 'max-w-md'
        )}
        showCloseButton={!loggedIn}
      >
        {loading ? (
          <div className="grid place-items-center px-6 py-16">
            <DialogTitle className="sr-only">账号</DialogTitle>
            <Loader2 className="size-5 animate-spin text-muted-foreground" />
          </div>
        ) : loggedIn ? (
          <div className="flex h-full min-h-0">
            {/* 左侧菜单 */}
            <nav className="flex w-44 shrink-0 flex-col gap-0.5 border-r border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-2.5">
              <div className="flex items-center gap-2.5 px-1 pb-3 pt-1">
                <span className="grid size-9 shrink-0 place-items-center rounded-full bg-gradient-to-br from-(--ui-accent,#7c83ff) to-[#b794ff] text-sm font-bold text-[#0b0b14]">
                  {(name[0] || '?').toUpperCase()}
                </span>
                <div className="min-w-0">
                  <DialogTitle className="truncate text-[0.8125rem] font-semibold leading-tight">{name}</DialogTitle>
                  {status?.email && status.email !== name ? (
                    <div className="truncate text-[0.6875rem] text-muted-foreground">{status.email}</div>
                  ) : null}
                </div>
              </div>

              {menu.map(item => (
                <button
                  className={cn(
                    'flex h-8 items-center gap-2 rounded-md px-2.5 text-[0.8125rem] text-(--ui-text-secondary) transition-colors hover:bg-(--ui-control-hover-background) hover:text-foreground',
                    activeSection === item.key && 'bg-(--ui-control-active-background) font-medium text-foreground'
                  )}
                  key={item.key}
                  onClick={() => setSection(item.key)}
                  type="button"
                >
                  <Codicon name={item.icon} size="0.9rem" />
                  {item.label}
                </button>
              ))}

              <div className="mt-auto rounded-lg border border-(--ui-stroke-tertiary) bg-[color:var(--ui-bg-elevated)] px-2.5 py-2">
                <div className="text-[0.625rem] text-muted-foreground">余额</div>
                <div className="mt-0.5 text-sm font-bold tabular-nums">
                  {balance != null ? balance : '—'}
                  <span className="ml-1 text-[0.625rem] font-normal text-muted-foreground">积分</span>
                </div>
              </div>
            </nav>

            {/* 右侧内容 */}
            <div className="flex min-h-0 flex-1 flex-col bg-(--ui-chat-surface-background)">
              <div className="flex h-9 shrink-0 items-center justify-end px-2">
                <Button
                  aria-label="关闭"
                  className="text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
                  onClick={() => closeAccountDialog()}
                  size="icon-xs"
                  type="button"
                  variant="ghost"
                >
                  <X className="size-4" />
                </Button>
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-6">
                {activeSection === 'usage' ? (
                  <UsageSection />
                ) : activeSection === 'recharge' ? (
                  <RechargeSection onChanged={() => void refresh()} />
                ) : activeSection === 'team' ? (
                  <TeamSection />
                ) : activeSection === 'roles' ? (
                  <RolesSection />
                ) : activeSection === 'permissions' ? (
                  <PermissionsSection />
                ) : (
                  <ProfileSection onChanged={() => void refresh()} status={status} />
                )}
              </div>
            </div>
          </div>
        ) : (
          /* 未登录:登录 / 注册 */
          <form className="flex flex-col gap-3.5 px-5 pb-5 pt-6" onSubmit={e => void submit('login', e)}>
            <DialogTitle className="flex items-center gap-2 text-base font-semibold">
              <Sparkles className="size-4 text-(--ui-accent,#7c83ff)" /> 登录 EasyHermes
            </DialogTitle>
            <p className="-mt-1.5 text-xs leading-5 text-muted-foreground">
              登录后启用 Kari 图片/视频、云端模型、消费明细与充值。不登录也能用本地聊天和工作流。
            </p>
            <label className="flex flex-col gap-1 text-xs font-medium text-muted-foreground">
              邮箱
              <Input autoComplete="username" disabled={busy} onChange={e => setEmail(e.target.value)} type="email" value={email} />
            </label>
            <label className="flex flex-col gap-1 text-xs font-medium text-muted-foreground">
              密码
              <Input
                autoComplete="current-password"
                disabled={busy}
                onChange={e => setPassword(e.target.value)}
                type="password"
                value={password}
              />
            </label>
            {showCloud ? (
              <label className="flex flex-col gap-1 text-xs font-medium text-muted-foreground">
                云端地址
                <Input
                  autoCapitalize="off"
                  autoCorrect="off"
                  disabled={busy}
                  onChange={e => setCloudBaseUrl(e.target.value)}
                  spellCheck={false}
                  type="url"
                  value={cloudBaseUrl}
                />
              </label>
            ) : (
              <button
                className="self-start text-[0.6875rem] text-muted-foreground underline-offset-2 hover:underline"
                onClick={() => setShowCloud(true)}
                type="button"
              >
                自定义云端地址
              </button>
            )}
            {error ? (
              <p className="text-xs leading-5 text-destructive" role="alert">
                {error}
              </p>
            ) : null}
            <div className="mt-1 flex gap-2">
              <Button className="flex-1" disabled={busy} type="submit">
                {busy ? <Loader2 className="size-3.5 animate-spin" /> : <LogIn className="size-3.5" />}
                登录
              </Button>
              <Button className="flex-1" disabled={busy} onClick={() => void submit('register')} type="button" variant="secondary">
                注册
              </Button>
            </div>
          </form>
        )}
      </DialogContent>
    </Dialog>
  )
}

export { closeAccountDialog }

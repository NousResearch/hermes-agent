import { type FormEvent, useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { DesktopAccountStatus } from '@/global'
import { Loader2, LogIn, Sparkles } from '@/lib/icons'

import { ListRow, LoadingState, SectionHeading, SettingsContent } from './primitives'

const DEFAULT_CLOUD_URL = (import.meta.env?.VITE_KARI_CLOUD_URL as string | undefined)?.trim() || 'https://lotjc.com/hermes'

function notifyAccountChanged() {
  window.dispatchEvent(new Event('hermes-account-changed'))
}

export function AccountSettings() {
  const [status, setStatus] = useState<DesktopAccountStatus | null>(null)
  const [cloudBaseUrl, setCloudBaseUrl] = useState(DEFAULT_CLOUD_URL)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

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

      if (next.cloudBaseUrl) {
        setCloudBaseUrl(next.cloudBaseUrl)
      }

      if (next.email) {
        setEmail(next.email)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const login = async (event: FormEvent) => {
    event.preventDefault()
    const account = window.hermesDesktop?.account

    if (!account?.login) {
      setError('桌面账号服务不可用')

      return
    }

    setBusy(true)
    setError('')

    try {
      const result = await account.login({ cloudBaseUrl, email, password })

      if (!result.ok) {
        setError(result.error || '登录失败')

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

  const logout = async () => {
    const account = window.hermesDesktop?.account

    if (!account?.logout) {
      setError('桌面账号服务不可用')

      return
    }

    setBusy(true)
    setError('')

    try {
      await account.logout()
      notifyAccountChanged()
      setStatus({ loggedIn: false, cloudBaseUrl, cloudReachable: false, error: null })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  if (loading) {
    return <LoadingState label="加载账号状态…" />
  }

  return (
    <SettingsContent>
      <SectionHeading icon={Sparkles} title="EasyHermes 账号" />

      <div className="space-y-3">
        {status?.loggedIn ? (
          <ListRow
            action={
              <Button disabled={busy} onClick={() => void logout()} size="sm" type="button" variant="secondary">
                {busy ? <Loader2 className="size-3.5 animate-spin" /> : null}
                登出
              </Button>
            }
            description={status.email || status.cloudBaseUrl || '已连接 EasyHermes 云端账号'}
            hint={typeof status.balance === 'number' ? `余额 ${status.balance}` : undefined}
            title={status.username || status.email || '已登录'}
          />
        ) : (
          <form className="space-y-3 rounded-md border border-border/60 bg-(--ui-bg-quinary) p-3" onSubmit={login}>
            <div className="flex items-center gap-2 text-sm font-medium">
              <LogIn className="size-4 text-muted-foreground" />
              <span>登录</span>
            </div>
            <label className="flex flex-col gap-1 text-xs font-medium text-muted-foreground">
              云端地址
              <Input
                autoCapitalize="off"
                autoCorrect="off"
                disabled={busy}
                onChange={event => setCloudBaseUrl(event.target.value)}
                spellCheck={false}
                type="url"
                value={cloudBaseUrl}
              />
            </label>
            <label className="flex flex-col gap-1 text-xs font-medium text-muted-foreground">
              邮箱
              <Input
                autoComplete="username"
                disabled={busy}
                onChange={event => setEmail(event.target.value)}
                type="email"
                value={email}
              />
            </label>
            <label className="flex flex-col gap-1 text-xs font-medium text-muted-foreground">
              密码
              <Input
                autoComplete="current-password"
                disabled={busy}
                onChange={event => setPassword(event.target.value)}
                type="password"
                value={password}
              />
            </label>
            <Button disabled={busy} size="sm" type="submit">
              {busy ? <Loader2 className="size-3.5 animate-spin" /> : null}
              登录
            </Button>
          </form>
        )}

        <p className="text-xs leading-5 text-muted-foreground">
          不登录也可以使用本地聊天和工作流画布。登录后会启用 Kari 图片/视频节点、云端模型下发、消费明细和充值能力。
        </p>

        {error ? (
          <p className="text-xs leading-5 text-destructive" role="alert">
            {error}
          </p>
        ) : null}
      </div>
    </SettingsContent>
  )
}

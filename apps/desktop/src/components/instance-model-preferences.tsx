import type { ModelOptionProvider } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { confirm } from '@/store/confirm'
import {
  $instanceModels,
  bindInstanceModels,
  hasPreviousPreferences,
  importPreviousPreferences,
  refreshInstanceModels
} from '@/store/instance-model-preferences'

interface Props {
  connectionId?: string
  profile?: string
  providers: readonly ModelOptionProvider[]
  enabled?: boolean
}

export function InstanceModelPreferences({ connectionId, profile, providers, enabled = true }: Props) {
  const view = useStore($instanceModels)
  const { locale } = useI18n()
  const zh = locale.startsWith('zh')

  const failureText =
    view.error === 'HR-MODEL-002'
      ? zh
        ? '模型偏好已在其他客户端更新，请刷新后重新选择。'
        : 'Model preferences changed on another client. Refresh and choose again.'
      : zh
        ? '无法同步模型偏好，请重试。'
        : 'Could not sync model preferences. Retry.'

  const bridgeAvailable = Boolean(window.hermesDesktop?.api)
  useEffect(
    () => (enabled && bridgeAvailable ? bindInstanceModels(connectionId, profile) : undefined),
    [connectionId, profile, enabled, bridgeAvailable]
  )

  if (!enabled || !bridgeAvailable) {
    return null
  }

  const importing = async () => {
    const before = $instanceModels.get()

    if (
      await confirm({
        title: zh ? '导入旧模型偏好？' : 'Import previous model preferences?',
        description: zh
          ? '导入到当前 Hermes 实例。原数据保留，不覆盖此实例已有的选择或取消；无法唯一识别的模型保留在本机。'
          : 'Import into this Hermes instance. Originals remain; existing choices and cancellations are preserved. Ambiguous model identities stay local.'
      })
    ) {
      if (
        $instanceModels.get().scope === before.scope &&
        $instanceModels.get().snapshot?.instanceId === before.snapshot?.instanceId
      ) {
        importPreviousPreferences(providers)
      }
    }
  }

  return (
    <div className="px-3 py-1.5 text-xs text-muted-foreground" role="status">
      {view.status === 'legacy'
        ? zh
          ? '此 Hermes 不支持偏好同步，仍使用本机设置。'
          : 'This Hermes does not support preference sync. Using local settings.'
        : view.status === 'loading'
          ? zh
            ? '正在读取此 Hermes 的模型偏好…'
            : 'Loading this Hermes’s model preferences…'
          : view.status === 'error'
            ? zh
              ? '模型偏好同步暂不可用'
              : 'Model preference sync is unavailable'
            : zh
              ? '收藏和隐藏与此 Hermes 实例同步'
              : 'Favorites and visibility sync with this Hermes instance'}
      {view.error ? (
        <p role="alert">
          {failureText} ({view.error})
          <Button onClick={refreshInstanceModels} size="sm" variant="ghost">
            {zh ? '重试' : 'Retry'}
          </Button>
        </p>
      ) : null}
      {view.snapshot && hasPreviousPreferences() ? (
        <Button disabled={view.busy} onClick={() => void importing()} size="sm" variant="ghost">
          {zh ? '导入旧收藏和隐藏设置' : 'Import previous favorites and visibility'}
        </Button>
      ) : null}
    </div>
  )
}

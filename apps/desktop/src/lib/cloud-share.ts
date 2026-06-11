import { writeClipboardText } from '@/components/ui/copy-button'
import { activeGateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'

interface CloudShareResult {
  channel_id: string
  already_shared: boolean
  pushed_seq: number
}

interface CloudStatusResult {
  channel_id?: string
  configured: boolean
  shared: boolean
}

interface CloudInviteResult {
  accept_token?: string
  email?: string
  permission?: string
}

// "Share to cloud" (channels slice 4.0): promote the session to a cloud
// channel and start the gateway's background pusher. Self-contained like
// exportSession so the actions menu can call it without threading a handler
// through the sidebar. All outcomes land as plain-English toasts.
export async function shareSessionToCloud(sessionId: string): Promise<void> {
  const gateway = activeGateway()

  if (!gateway) {
    notify({ kind: 'error', title: 'Share to cloud', message: 'Not connected yet — try again in a moment.' })

    return
  }

  try {
    const result = await gateway.request<CloudShareResult>('session.cloud_share', { session_id: sessionId })

    notify({
      kind: 'success',
      title: 'Shared to cloud',
      message: result.already_shared
        ? 'This chat was already syncing to your cloud channel.'
        : 'This chat now syncs to your cloud channel.'
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)

    if (/not configured/i.test(message)) {
      notify({
        kind: 'error',
        title: "Cloud sharing isn't set up",
        message: 'Add your cloud token (HERMES_CLOUD_TOKEN) where the gateway runs, then try again.'
      })

      return
    }

    notifyError(err, 'Could not share this chat to the cloud')
  }
}

export async function copyCloudChannelId(sessionId: string): Promise<void> {
  const gateway = activeGateway()

  if (!gateway) {
    notify({ kind: 'error', title: 'Copy cloud ID', message: 'Not connected yet — try again in a moment.' })

    return
  }

  try {
    const result = await gateway.request<CloudStatusResult>('session.cloud_status', { session_id: sessionId })

    if (!result.configured) {
      notify({
        kind: 'error',
        title: "Cloud sharing isn't set up",
        message: 'Add your cloud token (HERMES_CLOUD_TOKEN) where the gateway runs, then try again.'
      })

      return
    }

    if (!result.shared || !result.channel_id) {
      notify({
        kind: 'error',
        title: 'Copy cloud ID',
        message: 'Share this chat to the cloud first.'
      })

      return
    }

    await writeClipboardText(result.channel_id)
    notify({ kind: 'success', title: 'Copied cloud ID', message: result.channel_id })
  } catch (err) {
    notifyError(err, 'Could not copy the cloud channel ID')
  }
}

export async function inviteCloudChannelMember(sessionId: string, email: string): Promise<boolean> {
  const gateway = activeGateway()
  const trimmed = email.trim()

  if (!gateway) {
    notify({ kind: 'error', title: 'Invite to cloud', message: 'Not connected yet — try again in a moment.' })

    return false
  }

  if (!trimmed) {
    notify({ kind: 'error', title: 'Invite to cloud', message: 'Email is required.' })

    return false
  }

  try {
    const result = await gateway.request<CloudInviteResult>('session.cloud_invite', {
      email: trimmed,
      permission: 'read',
      session_id: sessionId
    })

    if (result.accept_token) {
      await writeClipboardText(result.accept_token)
    }

    notify({
      kind: 'success',
      title: result.accept_token ? 'Invite token copied' : 'Cloud invite created',
      message: `Invite ready for ${result.email || trimmed}.`
    })

    return true
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)

    if (/not configured/i.test(message)) {
      notify({
        kind: 'error',
        title: "Cloud sharing isn't set up",
        message: 'Add your cloud token (HERMES_CLOUD_TOKEN) where the gateway runs, then try again.'
      })

      return false
    }

    if (/not shared/i.test(message)) {
      notify({ kind: 'error', title: 'Invite to cloud', message: 'Share this chat to the cloud first.' })

      return false
    }

    notifyError(err, 'Could not create the cloud invite')

    return false
  }
}

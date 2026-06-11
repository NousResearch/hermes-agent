import { writeClipboardText } from '@/components/ui/copy-button'
import { activeGateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'

interface CloudShareResult {
  channel_id: string
  already_shared: boolean
  pushed_seq: number
}

export interface CloudStatusResult {
  channel_id?: string
  configured: boolean
  shared: boolean
}

interface CloudInviteResult {
  accept_token?: string
  email?: string
  permission?: string
}

export type CloudChannelPermission = 'admin' | 'post' | 'read'

export interface CloudChannel {
  created_at?: string | null
  history_floor_seq?: number | null
  id: string
  is_owner?: boolean
  last_seq?: number | null
  model?: string | null
  origin_device_id?: string | null
  origin_session_key?: string | null
  source?: string | null
  status?: string | null
  title?: string | null
  updated_at?: string | null
  visibility?: string | null
  your_permission?: string | null
}

interface CloudChannelsResult {
  channels?: CloudChannel[]
  count?: number
}

interface CloudAcceptInviteResult {
  channel_id?: string
  ok?: boolean
  permission?: string
}

export interface CloudChannelMessage {
  content?: string | null
  created_at?: string | null
  finish_reason?: string | null
  id?: string
  origin_device_id?: string | null
  origin_message_id?: string | null
  origin_ts?: number | null
  role: string
  sender_account_id?: string | null
  sender_device?: string | null
  seq: number
  token_count?: number | null
  tool_calls?: unknown
  tool_name?: string | null
}

export interface CloudChannelMessagesResult {
  count?: number
  last_seq?: number
  limit?: number
  messages?: CloudChannelMessage[]
  next_seq?: number
  since_seq?: number
  truncated?: boolean
}

export interface CloudChannelParticipant {
  count: number
  device: string
}

export interface CloudChannelParticipantsResult {
  count?: number
  generated_at?: string
  host_connected?: boolean
  participants?: CloudChannelParticipant[]
}

export interface CloudChannelTailStartResult {
  channel_id: string
  subscription_id: string
}

export interface CloudChannelTailStopResult {
  ok?: boolean
  stopped?: boolean
  subscription_id?: string
}

export interface CloudChannelMember {
  account_id: string
  display_name?: string | null
  email?: string | null
  granted_via?: string | null
  joined_at?: string | null
  permission?: string | null
}

export interface CloudMembersResult {
  channel_id?: string
  count?: number
  members?: CloudChannelMember[]
  owner_account_id?: string
  your_permission?: string
}

interface CloudMemberMutationResult {
  account_id?: string
  ok?: boolean
  permission?: string
  removed?: string
}

interface CloudDeleteResult {
  deleted?: string
  ok?: boolean
  stopped?: boolean
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

export async function loadSessionCloudStatus(
  sessionId: string,
  options: { quiet?: boolean } = {}
): Promise<CloudStatusResult | null> {
  const gateway = activeGateway()
  const trimmed = sessionId.trim()

  if (!gateway) {
    if (!options.quiet) {
      notify({ kind: 'error', title: 'Cloud channel', message: 'Not connected yet — try again in a moment.' })
    }

    return null
  }

  if (!trimmed) {
    return null
  }

  try {
    return await gateway.request<CloudStatusResult>('session.cloud_status', { session_id: trimmed })
  } catch (err) {
    if (!options.quiet) {
      notifyError(err, 'Could not load cloud channel status')
    }

    return null
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

export async function loadCloudChannelMembers(sessionId: string): Promise<CloudMembersResult | null> {
  const gateway = activeGateway()

  if (!gateway) {
    notify({ kind: 'error', title: 'Cloud members', message: 'Not connected yet — try again in a moment.' })

    return null
  }

  try {
    return await gateway.request<CloudMembersResult>('session.cloud_members', { session_id: sessionId })
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)

    if (/not configured/i.test(message)) {
      notify({
        kind: 'error',
        title: "Cloud sharing isn't set up",
        message: 'Add your cloud token (HERMES_CLOUD_TOKEN) where the gateway runs, then try again.'
      })

      return null
    }

    if (/not shared/i.test(message)) {
      notify({ kind: 'error', title: 'Cloud members', message: 'Share this chat to the cloud first.' })

      return null
    }

    notifyError(err, 'Could not load cloud members')

    return null
  }
}

export async function setCloudChannelMemberPermission(
  sessionId: string,
  accountId: string,
  permission: CloudChannelPermission
): Promise<boolean> {
  const gateway = activeGateway()
  const trimmedAccount = accountId.trim()

  if (!gateway) {
    notify({ kind: 'error', title: 'Cloud members', message: 'Not connected yet — try again in a moment.' })

    return false
  }

  if (!trimmedAccount) {
    notify({ kind: 'error', title: 'Cloud members', message: 'Account ID is required.' })

    return false
  }

  try {
    const result = await gateway.request<CloudMemberMutationResult>('session.cloud_member_permission', {
      account_id: trimmedAccount,
      permission,
      session_id: sessionId
    })

    notify({
      kind: 'success',
      title: 'Cloud member updated',
      message: `Permission set to ${result.permission || permission}.`
    })

    return true
  } catch (err) {
    notifyError(err, 'Could not update cloud member')

    return false
  }
}

export async function removeCloudChannelMember(sessionId: string, accountId: string): Promise<boolean> {
  const gateway = activeGateway()
  const trimmedAccount = accountId.trim()

  if (!gateway) {
    notify({ kind: 'error', title: 'Cloud members', message: 'Not connected yet — try again in a moment.' })

    return false
  }

  if (!trimmedAccount) {
    notify({ kind: 'error', title: 'Cloud members', message: 'Account ID is required.' })

    return false
  }

  try {
    await gateway.request<CloudMemberMutationResult>('session.cloud_member_remove', {
      account_id: trimmedAccount,
      session_id: sessionId
    })

    notify({ kind: 'success', title: 'Cloud member removed', message: trimmedAccount })

    return true
  } catch (err) {
    notifyError(err, 'Could not remove cloud member')

    return false
  }
}

export async function deleteCloudChannel(sessionId: string): Promise<boolean> {
  const gateway = activeGateway()

  if (!gateway) {
    notify({ kind: 'error', title: 'Delete cloud channel', message: 'Not connected yet — try again in a moment.' })

    return false
  }

  try {
    await gateway.request<CloudDeleteResult>('session.cloud_delete', { session_id: sessionId })

    notify({
      kind: 'success',
      title: 'Cloud channel deleted',
      message: 'Cloud history and sharing were removed.'
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
      notify({ kind: 'error', title: 'Delete cloud channel', message: 'Share this chat to the cloud first.' })

      return false
    }

    notifyError(err, 'Could not delete cloud channel')

    return false
  }
}

export async function loadCloudChannels(): Promise<CloudChannel[] | null> {
  const gateway = activeGateway()

  if (!gateway) {
    notify({ kind: 'error', title: 'Cloud channels', message: 'Not connected yet - try again in a moment.' })

    return null
  }

  try {
    const result = await gateway.request<CloudChannelsResult>('cloud.channels', {})

    return result.channels ?? []
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)

    if (/not configured/i.test(message)) {
      notify({
        kind: 'error',
        title: "Cloud sharing isn't set up",
        message: 'Add your cloud token (HERMES_CLOUD_TOKEN) where the gateway runs, then try again.'
      })

      return null
    }

    notifyError(err, 'Could not load cloud channels')

    return null
  }
}

export async function loadCloudChannelMessages(
  channelId: string,
  options: { limit?: number; sinceSeq?: number } = {}
): Promise<CloudChannelMessagesResult | null> {
  const gateway = activeGateway()
  const trimmed = channelId.trim()

  if (!gateway) {
    notify({ kind: 'error', title: 'Cloud messages', message: 'Not connected yet - try again in a moment.' })

    return null
  }

  if (!trimmed) {
    notify({ kind: 'error', title: 'Cloud messages', message: 'Channel ID is required.' })

    return null
  }

  try {
    return await gateway.request<CloudChannelMessagesResult>('cloud.channel_messages', {
      channel_id: trimmed,
      limit: options.limit ?? 100,
      since_seq: options.sinceSeq ?? 0
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)

    if (/not configured/i.test(message)) {
      notify({
        kind: 'error',
        title: "Cloud sharing isn't set up",
        message: 'Add your cloud token (HERMES_CLOUD_TOKEN) where the gateway runs, then try again.'
      })

      return null
    }

    notifyError(err, 'Could not load cloud messages')

    return null
  }
}

export async function loadCloudChannelParticipants(
  channelId: string,
  options: { quiet?: boolean } = {}
): Promise<CloudChannelParticipantsResult | null> {
  const gateway = activeGateway()
  const trimmed = channelId.trim()

  if (!gateway) {
    if (!options.quiet) {
      notify({ kind: 'error', title: 'Cloud participants', message: 'Not connected yet - try again in a moment.' })
    }

    return null
  }

  if (!trimmed) {
    if (!options.quiet) {
      notify({ kind: 'error', title: 'Cloud participants', message: 'Channel ID is required.' })
    }

    return null
  }

  try {
    return await gateway.request<CloudChannelParticipantsResult>('cloud.channel_participants', {
      channel_id: trimmed
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)

    if (/not configured/i.test(message)) {
      if (!options.quiet) {
        notify({
          kind: 'error',
          title: "Cloud sharing isn't set up",
          message: 'Add your cloud token (HERMES_CLOUD_TOKEN) where the gateway runs, then try again.'
        })
      }

      return null
    }

    if (!options.quiet) {
      notifyError(err, 'Could not load cloud participants')
    }

    return null
  }
}

export async function startCloudChannelTail(
  channelId: string,
  sinceSeq = 0
): Promise<CloudChannelTailStartResult | null> {
  const gateway = activeGateway()
  const trimmed = channelId.trim()

  if (!gateway || !trimmed) {
    return null
  }

  try {
    return await gateway.request<CloudChannelTailStartResult>('cloud.channel_tail_start', {
      channel_id: trimmed,
      since_seq: sinceSeq
    })
  } catch (err) {
    notifyError(err, 'Could not start cloud live updates')

    return null
  }
}

export async function stopCloudChannelTail(subscriptionId: string): Promise<CloudChannelTailStopResult | null> {
  const gateway = activeGateway()
  const trimmed = subscriptionId.trim()

  if (!gateway || !trimmed) {
    return null
  }

  try {
    return await gateway.request<CloudChannelTailStopResult>('cloud.channel_tail_stop', {
      subscription_id: trimmed
    })
  } catch {
    return null
  }
}

export async function acceptCloudChannelInvite(token: string): Promise<CloudAcceptInviteResult | null> {
  const gateway = activeGateway()
  const trimmed = token.trim()

  if (!gateway) {
    notify({ kind: 'error', title: 'Accept cloud invite', message: 'Not connected yet - try again in a moment.' })

    return null
  }

  if (!trimmed) {
    notify({ kind: 'error', title: 'Accept cloud invite', message: 'Invite token is required.' })

    return null
  }

  try {
    const result = await gateway.request<CloudAcceptInviteResult>('cloud.accept_invite', { token: trimmed })

    notify({
      kind: 'success',
      title: 'Cloud invite accepted',
      message: result.channel_id ? `Joined ${result.channel_id}.` : 'Cloud channel joined.'
    })

    return result
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)

    if (/not configured/i.test(message)) {
      notify({
        kind: 'error',
        title: "Cloud sharing isn't set up",
        message: 'Add your cloud token (HERMES_CLOUD_TOKEN) where the gateway runs, then try again.'
      })

      return null
    }

    if (/invalid|already used|expired/i.test(message)) {
      notify({ kind: 'error', title: 'Accept cloud invite', message: 'This invite is invalid, used, or expired.' })

      return null
    }

    notifyError(err, 'Could not accept the cloud invite')

    return null
  }
}

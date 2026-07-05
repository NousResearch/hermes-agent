import type { GatewayEvent } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import {
  acceptCloudChannelInvite,
  type CloudChannel,
  type CloudChannelMessage,
  type CloudChannelParticipant,
  loadCloudChannelMessages,
  loadCloudChannelParticipants,
  loadCloudChannels,
  startCloudChannelTail,
  stopCloudChannelTail
} from '@/lib/cloud-share'
import { cn } from '@/lib/utils'
import { $gateway } from '@/store/gateway'

interface CloudChannelsDialogProps {
  onOpenChange: (open: boolean) => void
  open: boolean
}

const MESSAGE_PAGE_SIZE = 100

interface MessageCursor {
  lastSeq: number
  nextSeq: number
  truncated: boolean
}

const channelLabel = (channel: CloudChannel) => {
  const title = channel.title?.trim()

  return title || channel.id
}

const formatSeq = (seq: null | number | undefined) => (typeof seq === 'number' ? String(seq) : '0')

const messageKey = (message: CloudChannelMessage) =>
  message.id || `${message.seq}:${message.origin_device_id || ''}:${message.origin_message_id || ''}`

const messageSender = (message: CloudChannelMessage) =>
  message.sender_device || message.origin_device_id || message.sender_account_id || ''

interface CloudMessageEventPayload {
  channel_id?: string
  message?: CloudChannelMessage
  next_seq?: number
  subscription_id?: string
}

export function CloudChannelsDialog({ onOpenChange, open }: CloudChannelsDialogProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const gateway = useStore($gateway)
  const [channels, setChannels] = useState<CloudChannel[]>([])
  const [hostConnected, setHostConnected] = useState(false)
  const [inviteToken, setInviteToken] = useState('')
  const [loading, setLoading] = useState(false)
  const [messages, setMessages] = useState<CloudChannelMessage[]>([])
  const [messagesLoading, setMessagesLoading] = useState(false)
  const [messageCursor, setMessageCursor] = useState<MessageCursor | null>(null)
  const [participants, setParticipants] = useState<CloudChannelParticipant[]>([])
  const [selectedChannelId, setSelectedChannelId] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const messageCursorRef = useRef<MessageCursor | null>(null)

  useEffect(() => {
    messageCursorRef.current = messageCursor
  }, [messageCursor])

  const appendLiveMessage = useCallback((channelId: string, message: CloudChannelMessage) => {
    setMessages(current => {
      const key = messageKey(message)

      if (current.some(existing => messageKey(existing) === key)) {
        return current
      }

      return [...current, message].sort((a, b) => a.seq - b.seq)
    })
    setMessageCursor(current => {
      const base = current ?? { lastSeq: message.seq, nextSeq: message.seq, truncated: false }
      const nextSeq = Math.max(base.nextSeq, message.seq)
      const next = {
        lastSeq: Math.max(base.lastSeq, message.seq),
        nextSeq,
        truncated: base.truncated
      }
      messageCursorRef.current = next

      return next
    })
    setChannels(current =>
      current.map(channel =>
        channel.id === channelId
          ? { ...channel, last_seq: Math.max(Number(channel.last_seq ?? 0), message.seq) }
          : channel
      )
    )
  }, [])

  const loadMessages = useCallback(async (channelId: string, sinceSeq = 0, append = false) => {
    setMessagesLoading(true)

    try {
      const result = await loadCloudChannelMessages(channelId, { limit: MESSAGE_PAGE_SIZE, sinceSeq })

      if (!result) {
        return
      }

      const nextMessages = result.messages ?? []
      setMessages(current => {
        if (!append) {
          return nextMessages
        }

        const seen = new Set(current.map(messageKey))

        return [...current, ...nextMessages.filter(message => !seen.has(messageKey(message)))]
      })
      setMessageCursor({
        lastSeq: result.last_seq ?? result.next_seq ?? sinceSeq,
        nextSeq: result.next_seq ?? sinceSeq,
        truncated: Boolean(result.truncated)
      })
    } finally {
      setMessagesLoading(false)
    }
  }, [])

  const loadParticipants = useCallback(async (channelId: string, quiet = false) => {
    const result = await loadCloudChannelParticipants(channelId, { quiet })

    if (!result) {
      return
    }

    setParticipants(result.participants ?? [])
    setHostConnected(Boolean(result.host_connected))
  }, [])

  const refresh = useCallback(async () => {
    setLoading(true)

    try {
      const result = await loadCloudChannels()

      if (result) {
        setChannels(result)
      }
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (open) {
      void refresh()
    }
  }, [open, refresh])

  const acceptInvite = useCallback(async () => {
    setSubmitting(true)

    try {
      const result = await acceptCloudChannelInvite(inviteToken)

      if (result) {
        setInviteToken('')
        await refresh()

        if (result.channel_id) {
          setSelectedChannelId(result.channel_id)
          await loadMessages(result.channel_id)
          await loadParticipants(result.channel_id)
        }
      }
    } finally {
      setSubmitting(false)
    }
  }, [inviteToken, loadMessages, loadParticipants, refresh])

  const selectChannel = useCallback(
    (channel: CloudChannel) => {
      setSelectedChannelId(channel.id)
      setMessages([])
      setMessageCursor(null)
      setParticipants([])
      setHostConnected(false)
      void loadMessages(channel.id)
      void loadParticipants(channel.id)
    },
    [loadMessages, loadParticipants]
  )

  useEffect(() => {
    if (!open || !selectedChannelId) {
      return
    }

    void loadParticipants(selectedChannelId, true)
    const timer = setInterval(() => void loadParticipants(selectedChannelId, true), 5000)

    return () => clearInterval(timer)
  }, [loadParticipants, open, selectedChannelId])

  useEffect(() => {
    if (!open || !selectedChannelId || !gateway) {
      return
    }

    let active = true
    let subscriptionId: string | null = null
    const offMessage = gateway.on('cloud.channel.message', (event: GatewayEvent<CloudMessageEventPayload>) => {
      const payload = event.payload

      if (!payload?.message || payload.channel_id !== selectedChannelId) {
        return
      }

      appendLiveMessage(selectedChannelId, payload.message)
    })
    void startCloudChannelTail(selectedChannelId, messageCursorRef.current?.nextSeq ?? 0).then(result => {
      if (!result) {
        return
      }

      if (!active) {
        void stopCloudChannelTail(result.subscription_id)

        return
      }

      subscriptionId = result.subscription_id
    })

    return () => {
      active = false
      offMessage()

      if (subscriptionId) {
        void stopCloudChannelTail(subscriptionId)
      }
    }
  }, [appendLiveMessage, gateway, open, selectedChannelId])

  const selectedChannel = channels.find(channel => channel.id === selectedChannelId) ?? null
  const canLoadMore =
    Boolean(selectedChannelId) &&
    Boolean(messageCursor?.truncated || (messageCursor && messageCursor.nextSeq < messageCursor.lastSeq))

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-3xl gap-4">
        <DialogHeader>
          <DialogTitle>{s.cloudChannelsTitle}</DialogTitle>
          <DialogDescription>{s.cloudChannelsDesc}</DialogDescription>
        </DialogHeader>

        <div className="grid gap-2">
          <div className="flex min-w-0 gap-2">
            <Input
              aria-label={s.cloudInviteTokenAria}
              className="min-w-0 flex-1"
              onChange={event => setInviteToken(event.currentTarget.value)}
              onKeyDown={event => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  void acceptInvite()
                }
              }}
              placeholder={s.cloudInviteTokenPlaceholder}
              value={inviteToken}
            />
            <Button disabled={submitting || !inviteToken.trim()} onClick={() => void acceptInvite()} type="button">
              {submitting ? s.cloudInviteAccepting : s.cloudInviteAccept}
            </Button>
          </div>

          <div className="grid gap-2 md:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
            <div className="grid min-w-0 gap-2">
              <div className="flex min-h-7 items-center justify-between gap-2">
                <div className="text-xs font-medium text-(--ui-text-secondary)">{s.cloudChannelsListTitle}</div>
                <Button disabled={loading} onClick={() => void refresh()} size="sm" type="button" variant="ghost">
                  <Codicon name="refresh" />
                  {loading ? s.cloudChannelsRefreshing : s.cloudChannelsRefresh}
                </Button>
              </div>

              <div
                className={cn(
                  'max-h-80 overflow-y-auto rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background)',
                  channels.length === 0 && 'p-3'
                )}
              >
                {channels.length === 0 ? (
                  <div className="text-xs text-(--ui-text-tertiary)">
                    {loading ? s.cloudChannelsLoading : s.cloudChannelsEmpty}
                  </div>
                ) : (
                  <div className="grid gap-px p-1">
                    {channels.map(channel => {
                      const selected = channel.id === selectedChannelId

                      return (
                        <button
                          aria-pressed={selected}
                          className={cn(
                            'grid min-h-12 grid-cols-[minmax(0,1fr)_auto] gap-2 rounded-[4px] px-2 py-1.5 text-left hover:bg-(--ui-control-hover-background)',
                            selected &&
                              'bg-(--ui-control-active-background) ring-1 ring-inset ring-(--ui-stroke-tertiary)'
                          )}
                          key={channel.id}
                          onClick={() => selectChannel(channel)}
                          type="button"
                        >
                          <span className="min-w-0">
                            <span className="block truncate text-xs font-medium text-(--ui-text-primary)">
                              {channelLabel(channel)}
                            </span>
                            <span className="mt-0.5 block truncate text-[0.6875rem] text-(--ui-text-tertiary)">
                              {channel.id}
                            </span>
                          </span>
                          <span className="flex shrink-0 flex-col items-end justify-center gap-1">
                            <Badge variant={channel.is_owner ? 'default' : 'muted'}>
                              {channel.is_owner ? s.cloudChannelsOwner : channel.your_permission || s.cloudChannelsRead}
                            </Badge>
                            <span className="text-[0.6875rem] text-(--ui-text-tertiary)">
                              {s.cloudChannelsSeq(formatSeq(channel.last_seq))}
                            </span>
                          </span>
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>

            <div className="grid min-w-0 gap-2">
              <div className="flex min-h-7 items-center justify-between gap-2">
                <div className="min-w-0 truncate text-xs font-medium text-(--ui-text-secondary)">
                  {selectedChannel ? channelLabel(selectedChannel) : s.cloudMessagesTitle}
                </div>
                <Button
                  disabled={!selectedChannelId || messagesLoading}
                  onClick={() => selectedChannelId && void loadMessages(selectedChannelId)}
                  size="sm"
                  type="button"
                  variant="ghost"
                >
                  <Codicon name="refresh" />
                  {s.cloudMessagesRefresh}
                </Button>
              </div>

              {selectedChannelId && (
                <div className="flex min-h-8 min-w-0 flex-wrap items-center gap-1 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background) px-2 py-1">
                  <span className="mr-1 text-[0.6875rem] font-medium text-(--ui-text-secondary)">
                    {s.cloudParticipantsTitle}
                  </span>
                  <Badge variant={hostConnected ? 'default' : 'muted'}>
                    {hostConnected ? s.cloudHostOnline : s.cloudHostOffline}
                  </Badge>
                  {participants.length === 0 ? (
                    <span className="text-[0.6875rem] text-(--ui-text-tertiary)">
                      {s.cloudParticipantsEmpty}
                    </span>
                  ) : (
                    participants.map(participant => (
                      <Badge key={participant.device} variant="outline">
                        {s.cloudParticipantLabel(participant.device, String(participant.count))}
                      </Badge>
                    ))
                  )}
                </div>
              )}

              <div className="max-h-80 overflow-y-auto rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-background)">
                {!selectedChannelId ? (
                  <div className="p-3 text-xs text-(--ui-text-tertiary)">{s.cloudMessagesPickChannel}</div>
                ) : messages.length === 0 ? (
                  <div className="p-3 text-xs text-(--ui-text-tertiary)">
                    {messagesLoading ? s.cloudMessagesLoading : s.cloudMessagesEmpty}
                  </div>
                ) : (
                  <div className="grid divide-y divide-(--ui-stroke-tertiary)">
                    {messages.map(message => (
                      <div className="grid gap-1 px-2 py-2" key={messageKey(message)}>
                        <div className="flex min-w-0 items-center gap-1.5">
                          <Badge variant="outline">{message.role}</Badge>
                          <span className="shrink-0 text-[0.6875rem] text-(--ui-text-tertiary)">#{message.seq}</span>
                          {messageSender(message) && (
                            <span className="min-w-0 truncate text-[0.6875rem] text-(--ui-text-tertiary)">
                              {messageSender(message)}
                            </span>
                          )}
                        </div>
                        <div className="whitespace-pre-wrap break-words text-xs leading-5 text-(--ui-text-primary)">
                          {message.content || s.cloudMessagesNoContent}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {selectedChannelId && canLoadMore && (
                <Button
                  disabled={messagesLoading}
                  onClick={() => void loadMessages(selectedChannelId, messageCursor?.nextSeq ?? 0, true)}
                  type="button"
                  variant="secondary"
                >
                  {messagesLoading ? s.cloudMessagesLoadingMore : s.cloudMessagesLoadMore}
                </Button>
              )}
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {s.cloudChannelsClose}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

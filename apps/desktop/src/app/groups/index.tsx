import { useStore } from '@nanostores/react'
import { type FormEvent, useEffect, useMemo, useRef, useState } from 'react'

import { MarkdownTextContent } from '@/components/assistant-ui/markdown-text'
import { GroupApprovalBar } from '@/components/assistant-ui/tool/approval'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { SearchField } from '@/components/ui/search-field'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { $profiles, profileDisplayName, refreshProfiles } from '@/store/profile'
import { $projects, pickProjectFolder, refreshProjects } from '@/store/projects'

import { groupRoomRoute, GROUPS_ROUTE } from '../routes'

import type { GroupMessage } from './group-model'
import {
  $groupState,
  beginGroupRoomsRequest,
  cacheGroupRoom,
  cacheSentGroupMessage,
  clearCachedGroupApproval,
  reconcileGroupRooms,
  removeCachedGroupRoom
} from './group-store'
import { createGroupTransport, type GroupRequester, mentionsFromText } from './group-transport'

interface GroupsViewProps {
  navigate: (path: string) => void
  request: GroupRequester
  roomId: string | null
}

export function GroupsView({ navigate, request, roomId }: GroupsViewProps) {
  const { t } = useI18n()
  const copy = t.groups
  const state = useStore($groupState)
  const profiles = useStore($profiles)
  const projects = useStore($projects)
  const room = roomId ? state.rooms[roomId] : undefined
  const transport = useMemo(() => createGroupTransport(request), [request])
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const [profileQuery, setProfileQuery] = useState('')
  const [selectedProfiles, setSelectedProfiles] = useState<string[]>([])
  const [workspace, setWorkspace] = useState('')
  const [draft, setDraft] = useState('')
  const [error, setError] = useState('')
  const [historyCursor, setHistoryCursor] = useState<string | null>(null)
  const [hasEarlier, setHasEarlier] = useState(false)
  const [loadingEarlier, setLoadingEarlier] = useState(false)
  const transcriptRef = useRef<HTMLDivElement>(null)

  const visibleProfiles = profiles.filter(profile => {
    const query = profileQuery.trim().toLowerCase()

    return !query || profile.name.toLowerCase().includes(query) || profileDisplayName(profile).toLowerCase().includes(query)
  })

  useEffect(() => {
    if (creating) {void Promise.all([refreshProfiles(), refreshProjects()]).catch(() => undefined)}
  }, [creating])

  useEffect(() => {
    let cancelled = false
    setError('')
    const listGeneration = roomId ? null : beginGroupRoomsRequest()
    const load = roomId ? transport.getRoom(roomId) : transport.listRooms()
    void load.then(result => {
      if (cancelled) {return}

      if (listGeneration !== null && 'rooms' in result && Array.isArray(result.rooms)) {
        reconcileGroupRooms(listGeneration, result.rooms)
      }

      if ('room' in result && result.room) {
        cacheGroupRoom(result.room)
        setHistoryCursor(result.cursor ?? null)
        setHasEarlier(result.has_more === true || Boolean(result.cursor))
      }
    }).catch(() => !cancelled && setError(copy.loadFailed))

    if (roomId) {void transport.subscribe(roomId).catch(() => undefined)}

    return () => {
      cancelled = true

      if (roomId) {void transport.unsubscribe(roomId).catch(() => undefined)}
    }
  }, [copy.loadFailed, roomId, transport])

  const create = async (event: FormEvent) => {
    event.preventDefault()

    if (!name.trim() || selectedProfiles.length === 0) {return}

    const result = await transport.createRoom({
      name: name.trim(), profiles: selectedProfiles, ...(workspace ? { workspace } : {})
    })

    if (result.room) {
      cacheGroupRoom(result.room)
      setCreating(false)
      navigate(groupRoomRoute(result.room.id))
    }
  }

  const loadEarlier = async () => {
    if (!roomId || loadingEarlier || !hasEarlier) {return}
    const viewport = transcriptRef.current
    const previousHeight = viewport?.scrollHeight ?? 0
    const beforeSeq = room?.messages.find(message => message.seq !== undefined)?.seq
    setLoadingEarlier(true)

    try {
      const result = await transport.getRoom(roomId, {
        ...(beforeSeq !== undefined ? { beforeSeq } : {}),
        ...(historyCursor ? { cursor: historyCursor } : {})
      })

      if (result.room) {cacheGroupRoom(result.room)}
      setHistoryCursor(result.cursor ?? null)
      setHasEarlier(result.has_more === true || Boolean(result.cursor))
      requestAnimationFrame(() => {
        if (viewport) {viewport.scrollTop += viewport.scrollHeight - previousHeight}
      })
    } finally {
      setLoadingEarlier(false)
    }
  }

  if (!roomId) {
    return <section className="flex h-full min-h-0 flex-col overflow-hidden bg-(--ui-chat-surface-background) pt-(--titlebar-height)">
      <header className="flex items-center justify-between px-6 py-4"><h1 className="text-lg font-semibold">{copy.title}</h1><Button onClick={() => setCreating(true)}>{copy.createRoom}</Button></header>
      {error && <p className="px-6 text-sm text-destructive">{error}</p>}
      {creating && <form className="mx-6 grid max-w-xl gap-3 border-y border-(--ui-stroke-tertiary) py-4" onSubmit={event => void create(event)}>
        <label className="grid gap-1 text-sm">{copy.roomName}<Input aria-label={copy.roomName} onChange={event => setName(event.target.value)} value={name} /></label>
        <fieldset className="grid gap-2"><legend className="text-sm">{copy.profiles}</legend>
          <SearchField aria-label={copy.searchProfiles} containerClassName="w-full" onChange={setProfileQuery} placeholder={copy.searchProfiles} value={profileQuery} />
          <div className="grid max-h-52 gap-1 overflow-auto">{visibleProfiles.map(profile => {
            const checked = selectedProfiles.includes(profile.name)
            const display = profileDisplayName(profile)

            return <label className="flex items-center gap-2 py-1 text-sm" key={profile.name}><input aria-label={`${display} (${profile.name})`} checked={checked} onChange={event => setSelectedProfiles(current => event.target.checked ? [...current, profile.name] : current.filter(name => name !== profile.name))} type="checkbox" /><span>{display}</span>{display !== profile.name && <span className="text-muted-foreground">{profile.name}</span>}</label>
          })}</div>
        </fieldset>
        <label className="grid gap-1 text-sm">{copy.workspace}
          <div className="flex gap-2">
            <Select onValueChange={value => setWorkspace(value === '__none__' ? '' : value)} value={workspace || '__none__'}>
              <SelectTrigger aria-label={copy.workspace} className="flex-1"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">{copy.noWorkspace}</SelectItem>
                {projects.flatMap(project => project.folders.map(folder => <SelectItem key={`${project.id}:${folder.path}`} value={folder.path}>{project.name} — {folder.path}</SelectItem>))}
              </SelectContent>
            </Select>
            <Button onClick={() => void pickProjectFolder().then(path => path && setWorkspace(path))} type="button" variant="secondary">{copy.browseWorkspace}</Button>
          </div>
        </label>
        <div className="flex gap-2"><Button type="submit">{copy.create}</Button><Button onClick={() => setCreating(false)} type="button" variant="secondary">{copy.cancel}</Button></div>
      </form>}
      <div className="min-h-0 flex-1 overflow-auto px-6 py-3">{Object.values(state.rooms).length === 0 ? <p className="text-sm text-muted-foreground">{copy.empty}</p> : <ul className="divide-y divide-(--ui-stroke-tertiary)">{Object.values(state.rooms).map(item => <li key={item.id}><button className="flex w-full items-center justify-between py-3 text-left" onClick={() => navigate(groupRoomRoute(item.id))} type="button"><span><strong className="block">{item.name}</strong><small className="text-muted-foreground">{item.profiles.join(', ')}</small></span><span className="text-xs text-muted-foreground">{item.messages.length}</span></button></li>)}</ul>}</div>
    </section>
  }

  return <section className="flex h-full min-h-0 flex-col overflow-hidden bg-(--ui-chat-surface-background) pt-(--titlebar-height)">
    <header className="flex items-center gap-2 border-b border-(--ui-stroke-tertiary) px-6 py-3"><Button onClick={() => navigate(GROUPS_ROUTE)} size="sm" variant="ghost">{copy.back}</Button><div className="min-w-0 flex-1"><h1 className="truncate font-semibold">{room?.name ?? roomId}</h1>{room?.workspace && <p className="truncate text-xs text-muted-foreground">{copy.workspace}: {room.workspace}</p>}<div className="mt-1 flex flex-wrap gap-1">{room?.profiles.map(profile => <Button key={profile} onClick={() => void transport.interrupt(roomId, profile)} size="micro" variant="text">{profile} · {copy.stop}</Button>)}</div></div><Button onClick={() => void transport.interrupt(roomId)} size="sm" variant="secondary">{copy.stop}</Button><Button onClick={() => void transport.deleteRoom(roomId).then(() => { removeCachedGroupRoom(roomId); navigate(GROUPS_ROUTE) })} size="sm" variant="destructive">{copy.delete}</Button></header>
    {(room?.contextStatus || room?.summary) && <aside className="border-b border-(--ui-stroke-tertiary) px-6 py-3 text-sm"><strong>{copy.summary}</strong>{room.contextStatus && <p className="text-muted-foreground">{copy.compressed}</p>}{room.summary && <p className="mt-1">{room.summary}</p>}</aside>}
    <div className="min-h-0 flex-1 overflow-auto px-6 py-5" ref={transcriptRef}><div className="mx-auto flex max-w-3xl flex-col gap-5">{hasEarlier && <Button disabled={loadingEarlier} onClick={() => void loadEarlier()} size="sm" variant="textStrong">{loadingEarlier ? copy.loadingEarlier : copy.loadEarlier}</Button>}{room?.messages.map(message => <GroupMessageView copy={copy} key={message.id} message={message} respond={async (sessionId, choice) => {
      const result = await transport.respondToApproval(sessionId, choice)
      clearCachedGroupApproval(roomId, message.id)

      return result
    }} />)}</div></div>
    <form className="mx-auto flex w-full max-w-3xl gap-2 px-6 pb-5" onSubmit={event => { event.preventDefault(); const content = draft.trim();

 if (!content) {return;} setDraft(''); void transport.sendMessage(roomId, content, mentionsFromText(content, room?.profiles ?? [])).then(cacheSentGroupMessage) }}><Textarea aria-label={copy.message} onChange={event => setDraft(event.target.value)} value={draft} /><Button type="submit">{copy.send}</Button></form>
  </section>
}

function GroupMessageView({ copy, message, respond }: { copy: ReturnType<typeof useI18n>['t']['groups']; message: GroupMessage; respond: (sessionId: string, choice: 'once' | 'session' | 'always' | 'deny') => Promise<unknown> }) {
  const isUser = message.role === 'user'

  return <article className={isUser ? 'self-end max-w-[85%]' : 'w-full'}><div className="mb-1 flex items-center gap-2 text-xs text-muted-foreground"><strong className="text-foreground">{isUser ? copy.you : message.profile ?? copy.agent}</strong>{message.status === 'streaming' && <span>{copy.working}</span>}</div><div className={isUser ? 'rounded-xl bg-(--ui-bg-quaternary) px-3 py-2' : ''}><MarkdownTextContent isRunning={message.status === 'streaming'} text={message.content} />{message.parts.filter(part => part.type === 'tool-call').map(part => <ToolFallback addResult={() => undefined} args={part.args ?? {}} argsText={'argsText' in part ? part.argsText ?? '{}' : '{}'} key={part.toolCallId ?? 'tool'} result={'result' in part ? part.result : undefined} resume={() => undefined} status={message.status === 'streaming' ? { type: 'running' } : { type: 'complete' }} toolCallId={part.toolCallId ?? 'tool'} toolName={part.toolName ?? 'tool'} type="tool-call" />)}</div>{message.status === 'approval' && message.runtimeSessionId && <GroupApprovalBar onRespond={choice => respond(message.runtimeSessionId!, choice)} request={{
        allowPermanent: message.approval?.allowPermanent,
        choices: message.approval?.choices,
        command: message.approval?.command ?? message.content,
        description: message.approval?.description || copy.approval,
        sessionId: message.runtimeSessionId,
        smartDenied: message.approval?.smartDenied
      }} />}</article>
}

import { host, SESSION_AREAS, useQuery, Button, Badge } from '@hermes/plugin-sdk';
import { useState, useRef, useEffect } from 'react';
import { jsx, jsxs } from 'react/jsx-runtime';

// Host @hermes/shared LOCAL_CONNECTION_ID contract. External plugins can only
// import the plugin SDK, not @hermes/shared; do not infer local from a URL.
const LOCAL_CONNECTION_ID = 'local';
const REMOTE_VIEWER_UNSUPPORTED = 'Remote viewers are unsupported: no viewer tunnel is available. Open Hermes on the realm machine using a local connection; forwarding only the REST endpoint over SSH is not enough.';
const validId = value => typeof value === 'string' && value.length > 0 && value.length <= 256;
const validSession = session => !!session && validId(session.connectionId) && validId(session.profile)
  && [session.runtimeSessionId, session.storedSessionId].every(id => id == null || validId(id))
  && (validId(session.runtimeSessionId) || validId(session.storedSessionId));
const ownerBody = session => Object.fromEntries([
  ['runtime_session_id', session.runtimeSessionId], ['stored_session_id', session.storedSessionId]
].filter(([, value]) => validId(value)));

export function realmQueryOptions(ctx, session) {
  const enabled = validSession(session);
  return {
    queryKey: ['hermes-realms', session?.connectionId, session?.profile, session?.storedSessionId, session?.runtimeSessionId],
    enabled,
    queryFn: async () => {
      if (!enabled) throw new Error('Realm session owner is unavailable');
      return ctx.rest(`/realms?${new URLSearchParams(ownerBody(session))}`, { scope: session });
    },
    refetchInterval: 5000,
    staleTime: 3000,
    retry: 1,
    gcTime: 60000
  };
}

function ownedRealms(data, session) {
  return (Array.isArray(data?.realms) ? data.realms : []).filter(realm =>
    /^[A-Za-z0-9_-]{1,128}$/.test(realm.id || '')
    && (!session.runtimeSessionId || realm.runtime_session_id === session.runtimeSessionId)
    && (!session.storedSessionId || realm.stored_session_id === session.storedSessionId));
}

const modeLabel = mode => ({ realm: 'Realm mode · waiting for apps', host: 'Host mode · your desktop', ask: 'Ask mode · choose a desktop' })[mode];
// A VM realm is a different machine, not a different window group, so the user
// must be able to tell which one a session is on at a glance.
const KIND_NAMES = { realm: 'Realm', 'omarchy-vm': 'Omarchy VM' };
const kindName = kind => KIND_NAMES[kind] || 'Realm';
const gib = bytes => Number.isFinite(bytes) && bytes > 0 ? `${(bytes / 1073741824).toFixed(1)} GB` : null;
export function realmLabel(realm) {
  const name = kindName(realm.kind);
  const state = ['starting', 'live', 'stopping', 'stopped', 'error'].includes(realm.state) ? realm.state : 'unknown';
  const control = realm.controlled ? ' · user control' : '';
  if (realm.kind === 'omarchy-vm') {
    // Cost is what the user actually wants on hover: a guest holds its whole
    // -m figure in host RAM for as long as it runs.
    const detail = [gib(realm.stats?.memory_bytes) && `${gib(realm.stats.memory_bytes)} RAM`,
      gib(realm.stats?.disk_bytes) && `${gib(realm.stats.disk_bytes)} disk`,
      realm.network === false && 'no network'].filter(Boolean).join(' · ');
    return `${name} · ${state}${detail ? ` · ${detail}` : ''}${control}`;
  }
  const count = Number.isSafeInteger(realm.window_count) && realm.window_count >= 0 ? realm.window_count : null;
  const windows = count === null ? 'windows unknown' : `${count} ${count === 1 ? 'window' : 'windows'}`;
  return `${name} · ${windows} · ${state}${control}`;
}

export async function viewerId(session, realmId) {
  const identity = new TextEncoder().encode(JSON.stringify([session.connectionId, session.profile, realmId]));
  const digest = await crypto.subtle.digest('SHA-256', identity);
  return `realm-${Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('')}`;
}

export async function openRealmViewer(ctx, session, realm, target, isCurrent = () => true) {
  if (!validSession(session) || !ownedRealms({ realms: [realm] }, session).length) throw new Error('Realm session owner is unavailable');
  if (realm.state !== 'live') throw new Error('Realm is not live');
  if (!['watch', 'popout'].includes(target)) throw new Error('Unsupported viewer surface');
  if (session.connectionId !== LOCAL_CONNECTION_ID) throw new Error(REMOTE_VIEWER_UNSUPPORTED);
  const reply = await ctx.rest(`/realms/${encodeURIComponent(realm.id)}/watch`, { method: 'POST', body: ownerBody(session), scope: session });
  let url;
  try { url = new URL(reply.url); } catch { throw new Error('Invalid viewer URL'); }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || /[\s\\]/.test(reply.url)) throw new Error('Invalid viewer URL');
  const id = await viewerId(session, realm.id);
  if (!isCurrent()) throw new Error('Realm owner changed before viewer opened');
  const label = `${kindName(realm.kind)} · ${session.profile}`;
  const opened = target === 'watch'
    ? await host.openPreview?.({ url: reply.url, label, session })
    : await ctx.os?.openViewer?.({ id, url: reply.url, title: label.slice(0, 120), session });
  if (!opened) throw new Error('Viewer unavailable or session owner is stale');
}

export default {
  id: 'hermes-realms',
  defaultEnabled: false,
  name: 'Realms',
  description: 'Desktop viewing controls only. Enable hermes-realms under Agent plugins for each profile that should use private desktops.',
  register(ctx) {
    function StatusRow({ session }) {
      const result = useQuery(realmQueryOptions(ctx, session));
      const ownerKey = JSON.stringify(realmQueryOptions(ctx, session).queryKey);
      const currentOwner = useRef(ownerKey);
      currentOwner.current = ownerKey;
      const mounted = useRef(true);
      const pending = useRef(false);
      const [action, setAction] = useState(null);
      useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
      async function open(realm, target) {
        if (pending.current) return;
        pending.current = true;
        const current = () => mounted.current && currentOwner.current === ownerKey;
        setAction({ ownerKey, pending: true });
        try {
          await openRealmViewer(ctx, session, realm, target, current);
          if (current()) setAction(null);
        } catch {
          // Native/REST exceptions can include ticket URLs; never render/log them.
          if (current()) setAction({ ownerKey, error: 'Unable to open viewer. Refresh the realm and try again; check viewer access for remote connections.' });
        } finally { pending.current = false; }
      }
      if (!validSession(session)) return null;
      const realms = ownedRealms(result.data, session);
      if (result.isError && realms.length) return jsxs('div', { role: 'status', style: { fontSize: 12 }, children: [
        'Realm status unavailable · ',
        jsx(Button, { size: 'micro', variant: 'ghost', disabled: result.isFetching, onClick: () => void result.refetch(), children: 'Retry' })
      ] });
      const mode = result.data?.mode;
      // The two kinds have independent prerequisites: a profile with a working
      // labwc realm and no VM base image is ready, just not for the VM kind.
      const setup = result.data?.kind === 'omarchy-vm' ? result.data?.vm_setup : result.data?.setup;
      if (mode === 'realm' && setup?.ready === false) return jsx('div', {
        role: 'alert', style: { fontSize: 12 }, children: jsxs('details', { children: [
          jsx('summary', { children: `${kindName(result.data?.kind)} setup required` }),
          jsx('div', { children: setup.message })
        ] })
      });
      if (!realms.length) return modeLabel(mode) ? jsx('div', { style: { color: 'var(--ui-text-secondary)', fontSize: 12 }, children: modeLabel(mode) }) : null;
      return jsxs('div', { 'data-realms-status': '', style: { display: 'flex', flexDirection: 'column', gap: 4 }, children: [
        mode !== 'realm' && modeLabel(mode) ? jsx('span', { children: modeLabel(mode) }) : null,
        session.connectionId !== LOCAL_CONNECTION_ID ? jsx('span', { role: 'note', children: REMOTE_VIEWER_UNSUPPORTED }) : null,
        action?.ownerKey === ownerKey && action.error ? jsx('span', { role: 'alert', children: action.error }) : null,
        ...realms.map(realm => jsxs('div', { style: { display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, fontSize: 12 }, children: [
          jsx('span', { children: realmLabel(realm) }),
          jsx(Button, { size: 'micro', variant: 'ghost', disabled: session.connectionId !== LOCAL_CONNECTION_ID || realm.state !== 'live' || (action?.ownerKey === ownerKey && action.pending), onClick: () => void open(realm, 'watch'), children: 'Watch' }),
          jsx(Button, { size: 'micro', variant: 'ghost', disabled: session.connectionId !== LOCAL_CONNECTION_ID || realm.state !== 'live' || (action?.ownerKey === ownerKey && action.pending), onClick: () => void open(realm, 'popout'), children: 'Pop out' }),
          realm.error ? jsx('span', { role: 'alert', children: /https?:|ticket|token/i.test(String(realm.error)) ? 'Realm failed; check backend diagnostics.' : String(realm.error).slice(0, 240) }) : null
        ] }, realm.id))
      ] });
    }
    function RealmBadge({ session }) {
      const result = useQuery(realmQueryOptions(ctx, session));
      if (!validSession(session)) return null;
      // A failed lookup is not evidence that this session has a realm. Keep
      // ordinary rows quiet and known badges stable while polling retries.
      const realms = ownedRealms(result.data, session);
      if (!realms.length) return null;
      const label = `${result.isError ? 'Realm status unavailable · Last known: ' : ''}${realms.map(realmLabel).join('; ')}`;
      // The compact badge names the kind, because 'which machine am I on' is
      // the one thing a glance has to answer.
      const vm = realms.some(realm => realm.kind === 'omarchy-vm');
      const knownCount = !vm && realms.every(realm => Number.isSafeInteger(realm.window_count) && realm.window_count >= 0);
      const compact = vm ? 'Omarchy VM'
        : knownCount ? `Realm · ${realms.reduce((sum, realm) => sum + realm.window_count, 0)}` : 'Realm';
      return jsx(Badge, { size: 'xs', title: label, 'aria-label': label, children: compact });
    }
    ctx.register({ id: 'realm-status', area: SESSION_AREAS.statusStack, data: { render: StatusRow } });
    ctx.register({ id: 'realm-tile', area: SESSION_AREAS.tileBadge, data: { render: RealmBadge } });
    ctx.register({ id: 'realm-list', area: SESSION_AREAS.listBadge, data: { render: RealmBadge } });
  }
};

import { Button, ConfirmDialog, SegmentedControl, usePluginI18n, useQuery } from '@hermes/plugin-sdk';
import { useEffect, useRef, useState } from 'react';
import { jsx, jsxs } from 'react/jsx-runtime';

export const setupLocales = {
  en: {
    realm: 'Realm', vm: 'Omarchy VM', repair: 'Repair', install: 'Set up', start: 'Use this desktop',
    repairAction: 'Repair…', installAction: 'Set up…', vmAction: 'Set up Omarchy VM…',
    switchAction: 'Use this desktop…', reviewTitle: 'Review desktop setup', details: 'Details',
    ready: 'Ready', needsRepair: 'Needs setup or repair', needsVm: 'A disposable Omarchy desktop with its own kernel and disk.',
    checking: 'Checking setup…', started: 'Setup started', running: 'Preparing desktop…', failed: 'Setup failed',
    retry: 'Retry…', connectionLost: 'Connection interrupted. Setup may still be running.',
    reviewChanged: 'Setup requirements changed. Review the updated details before confirming.',
    error: 'Unable to prepare this desktop. Try again.', scoped: 'Desktop choice: this chat. Driver and base image: this profile. Approved system packages affect the gateway host. Nothing is installed before you confirm.'
  },
  ru: {
    realm: 'Realm', vm: 'Omarchy VM', repair: 'Исправить', install: 'Настроить', start: 'Использовать этот рабочий стол',
    repairAction: 'Исправить…', installAction: 'Настроить…', vmAction: 'Настроить Omarchy VM…',
    switchAction: 'Использовать этот рабочий стол…', reviewTitle: 'Проверка настройки рабочего стола', details: 'Подробности',
    ready: 'Готово', needsRepair: 'Требуется настройка или исправление', needsVm: 'Одноразовый рабочий стол Omarchy с отдельным ядром и диском.',
    checking: 'Проверка настройки…', started: 'Настройка запущена', running: 'Подготовка рабочего стола…', failed: 'Не удалось настроить',
    retry: 'Повторить…', connectionLost: 'Соединение прервано. Настройка может продолжаться.',
    reviewChanged: 'Требования изменились. Проверьте обновлённые сведения перед подтверждением.',
    error: 'Не удалось подготовить рабочий стол. Повторите попытку.', scoped: 'Рабочий стол — для этого чата; драйвер и базовый образ — для этого профиля. Одобренные системные пакеты устанавливаются на сервер шлюза. Установка начнётся после подтверждения.'
  },
  zh: {
    realm: 'Realm', vm: 'Omarchy VM', repair: '修复', install: '设置', start: '使用此桌面',
    repairAction: '修复…', installAction: '设置…', vmAction: '设置 Omarchy VM…',
    switchAction: '使用此桌面…', reviewTitle: '确认桌面设置', details: '详细信息',
    ready: '就绪', needsRepair: '需要设置或修复', needsVm: '拥有独立内核和磁盘的一次性 Omarchy 桌面。',
    checking: '正在检查…', started: '设置已开始', running: '正在准备桌面…', failed: '设置失败',
    retry: '重试…', connectionLost: '连接中断。设置可能仍在进行。',
    reviewChanged: '设置要求已更改。请查看更新后的详情再确认。',
    error: '无法准备此桌面。请重试。', scoped: '桌面选择仅适用于此聊天，驱动和基础镜像仅用于此配置。批准的系统软件包会安装到网关主机。在您确认之前不会安装任何内容。'
  }
};

const ownerBody = session => Object.fromEntries([
  ['runtime_session_id', session.runtimeSessionId], ['stored_session_id', session.storedSessionId]
].filter(([, value]) => typeof value === 'string' && value.length));
const running = job => job?.state === 'running';
const validReview = value => value && ['realm', 'omarchy-vm'].includes(value.kind)
  && ['repair', 'install', 'start'].includes(value.action) && typeof value.summary === 'string'
  && Array.isArray(value.details) && value.details.every(detail => typeof detail === 'string')
  && typeof value.consent === 'string' && value.consent.length > 0;

export function RealmSetupControls({ ctx, session, data, refresh }) {
  const translate = usePluginI18n('hermes-realms');
  const t = key => { const value = translate(key); return value === key ? setupLocales.en[key] : value; };
  const currentKind = data.kind || 'realm';
  const [choice, setChoice] = useState(null);
  const kind = choice || currentKind;
  const [review, setReview] = useState(null);
  const [preparing, setPreparing] = useState(false);
  const [error, setError] = useState(null);
  const [started, setStarted] = useState(null);
  const mounted = useRef(true);
  const pending = useRef(false);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const jobSeed = started || data.setup_job;
  const jobResult = useQuery({
    queryKey: ['hermes-realms-setup', session.connectionId, session.profile, session.storedSessionId, session.runtimeSessionId, jobSeed?.id],
    enabled: Boolean(jobSeed?.id) && running(jobSeed),
    queryFn: () => ctx.rest(`/realms/setup/jobs/${encodeURIComponent(jobSeed.id)}?${new URLSearchParams(ownerBody(session))}`, { scope: session }),
    refetchInterval: query => running(query.state.data || jobSeed) ? 1000 : false,
    retry: 1,
    gcTime: 60000
  });
  const job = jobResult.data || jobSeed;
  const busy = preparing || review !== null || running(job);
  const refreshed = useRef(null);
  useEffect(() => {
    if (job?.state === 'succeeded' && refreshed.current !== job.id) {
      refreshed.current = job.id;
      void refresh();
    }
  }, [job?.id, job?.state, refresh]);
  const setup = kind === 'omarchy-vm' ? data.vm_setup : data.setup;
  const ready = setup?.ready === true;
  const needsAction = !ready || kind !== currentKind || data.mode !== 'realm';

  async function prepare() {
    if (pending.current || running(job)) return;
    pending.current = true;
    setPreparing(true);
    setError(null);
    const scope = Object.freeze({ ...session });
    try {
      const value = await ctx.rest('/realms/setup/prepare', { method: 'POST', body: { ...ownerBody(scope), kind }, scope });
      if (!validReview(value) || value.kind !== kind) throw new Error(t('error'));
      if (mounted.current) setReview({ value, scope });
    } catch (failure) {
      if (mounted.current) setError(failure instanceof Error ? failure.message : t('error'));
    } finally {
      pending.current = false;
      if (mounted.current) setPreparing(false);
    }
  }

  async function confirm() {
    if (!mounted.current || pending.current || !review) return;
    pending.current = true;
    const { value, scope } = review;
    try {
      const receipt = await ctx.rest('/realms/setup/start', {
        method: 'POST', body: { ...ownerBody(scope), kind: value.kind, consent: value.consent }, scope
      });
      if (!receipt?.id || !['running', 'succeeded', 'failed'].includes(receipt.state)) throw new Error(t('error'));
      if (receipt.state === 'failed') throw new Error(receipt.message || t('failed'));
      if (mounted.current) {
        setStarted(receipt);
        setError(null);
        if (receipt.state === 'succeeded') await refresh();
      }
    } catch (failure) {
      // Electron's IPC error envelope does not preserve HTTP status fields.
      // Re-read the proposal rather than guessing from an exception string.
      let changed = false;
      if (mounted.current) {
        try {
          const next = await ctx.rest('/realms/setup/prepare', { method: 'POST', body: { ...ownerBody(scope), kind: value.kind }, scope });
          if (mounted.current && validReview(next) && next.kind === value.kind
              && JSON.stringify(next.consent) !== JSON.stringify(value.consent)) {
            setReview({ value: next, scope });
            changed = true;
          }
        } catch { /* Preserve the original failure if the connection is down. */ }
      }
      if (changed) throw new Error(t('reviewChanged'));
      throw failure;
    } finally { pending.current = false; }
  }

  const actionLabel = error || job?.state === 'failed' ? t('retry')
    : kind === 'omarchy-vm' && !ready ? t('vmAction')
    : !ready ? t('repairAction') : t('switchAction');
  const status = preparing ? t('checking') : running(job) ? (job.message || t('running'))
    : ready && kind === currentKind && data.mode === 'realm' ? t('ready')
    : kind === 'omarchy-vm' ? t('needsVm') : t('needsRepair');

  return jsxs('div', { className: 'flex flex-col gap-2 text-xs', 'data-realms-setup': '', children: [
    jsxs('div', { className: 'flex flex-wrap items-center gap-2', children: [
      jsx(SegmentedControl, { options: [{ id: 'realm', label: t('realm') }, { id: 'omarchy-vm', label: t('vm') }], value: kind, disabled: busy,
        onChange: next => { setChoice(next); setError(null); } }),
      jsx('span', { role: 'status', className: 'text-muted-foreground', children: status }),
      needsAction && !running(job) && jsx(Button, { size: 'micro', variant: 'ghost', disabled: busy,
        onClick: () => void prepare(), children: actionLabel })
    ] }),
    (error || job?.state === 'failed' || jobResult.isError) && jsx('div', { role: 'alert', children:
      error || (jobResult.isError ? t('connectionLost') : job.message || t('failed')) }),
    review && jsx(ConfirmDialog, {
      open: true, title: t('reviewTitle'), confirmLabel: t(review.value.action), busyLabel: t('checking'),
      doneLabel: t('started'), onClose: () => setReview(null), onConfirm: confirm,
      description: jsxs('span', { className: 'flex flex-col gap-3', children: [
        jsx('span', { children: review.value.summary }),
        jsx('span', { children: t('scoped') }),
        review.value.details.length > 0 && jsxs('details', { children: [
          jsx('summary', { children: t('details') }),
          ...review.value.details.map((detail, index) => jsx('span', { className: 'block break-words', children: detail }, index))
        ] })
      ] })
    })
  ] });
}

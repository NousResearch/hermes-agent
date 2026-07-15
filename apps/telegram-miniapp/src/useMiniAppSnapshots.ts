import { useCallback, useEffect, useRef, useState } from "react";
import {
  authenticateTelegram,
  fetchApprovalsSnapshot,
  fetchCapabilitiesSnapshot,
  fetchLogsSnapshot,
  fetchSessionsSnapshot,
  fetchStatusSnapshot,
  postApprovalDecision,
  MiniAppApiError,
  type ApprovalDecision,
  type CapabilityItem,
  type SnapshotMeta,
  type StatusSnapshot,
} from "./api";
import {
  approvalPreviews,
  recentLogs,
  sessionPreviews,
  type ApprovalPreview,
  type LogLine,
  type SessionPreview,
} from "./mockData";
import {
  POLL_INTERVAL_MS,
  STALE_AFTER_MS,
  STORAGE_KEYS,
  approveActionEnabled,
  createEndpointHealth,
  createEndpointStateUpdates,
  endpointKeys,
  logLineKey,
  mapServerApproval,
  mapServerLog,
  mapServerSession,
  readStoredApprovalId,
  removeStoredValue,
  updateEndpointHealth,
  writeStoredValue,
  type ApiState,
  type EndpointHealthItem,
  type EndpointKey,
  type LogLevelFilter,
} from "./appModel";
import { triggerTelegramRefreshHaptic, type TelegramRuntime } from "./telegram";

export function useMiniAppSnapshots(telegram: TelegramRuntime, apiConfigured: boolean) {
  const [snapshot, setSnapshot] = useState<StatusSnapshot | null>(null);
  const [apiState, setApiState] = useState<ApiState>("mock");
  const [selectedApprovalId, setSelectedApprovalId] = useState(() => readStoredApprovalId(approvalPreviews[0]?.id ?? ""));
  const [selectedSessionId, setSelectedSessionId] = useState(() => sessionPreviews[0]?.id ?? "");
  const [selectedLogKey, setSelectedLogKey] = useState(() => (recentLogs[0] ? logLineKey(recentLogs[0]) : ""));
  const [logLevelFilter, setLogLevelFilter] = useState<LogLevelFilter>("all");
  const [serverApprovals, setServerApprovals] = useState<ApprovalPreview[]>([]);
  const [serverSessions, setServerSessions] = useState<SessionPreview[]>([]);
  const [serverLogs, setServerLogs] = useState<LogLine[]>([]);
  const [serverCapabilities, setServerCapabilities] = useState<CapabilityItem[]>([]);
  const [approvalsVersion, setApprovalsVersion] = useState("");
  const [isActionOwner, setIsActionOwner] = useState(false);
  const [decisionError, setDecisionError] = useState("");
  const [endpointHealth, setEndpointHealth] = useState<Record<EndpointKey, EndpointHealthItem>>(() => createEndpointHealth("preview"));
  // Per-endpoint time of that endpoint's OWN last successful read, so a
  // degraded section can be dated (or marked never-loaded) truthfully instead
  // of borrowing the whole poll's success time.
  const [endpointLastOk, setEndpointLastOk] = useState<Record<EndpointKey, number | null>>(() =>
    endpointKeys.reduce((acc, key) => ({ ...acc, [key]: null }), {} as Record<EndpointKey, number | null>),
  );
  const [snapshotMeta, setSnapshotMeta] = useState<Record<"approvals" | "sessions" | "logs", SnapshotMeta | null>>({ approvals: null, sessions: null, logs: null });
  const [isRefreshing, setRefreshing] = useState(false);
  const [lastSuccessAt, setLastSuccessAt] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const refreshRequestId = useRef(0);
  const activeRefreshes = useRef(0);

  const hasServerSnapshot = lastSuccessAt !== null;
  const approvals = hasServerSnapshot ? serverApprovals : approvalPreviews;
  const sessions = hasServerSnapshot ? serverSessions : sessionPreviews;
  const logs = hasServerSnapshot ? serverLogs : recentLogs;

  useEffect(() => {
    if (approvals.length === 0) {
      if (selectedApprovalId) setSelectedApprovalId("");
      removeStoredValue(STORAGE_KEYS.selectedApprovalId);
      return;
    }

    const selectedExists = approvals.some((approval) => approval.id === selectedApprovalId);
    if (!selectedExists) {
      setSelectedApprovalId(approvals[0].id);
      return;
    }

    writeStoredValue(STORAGE_KEYS.selectedApprovalId, selectedApprovalId);
  }, [approvals, selectedApprovalId]);

  useEffect(() => {
    if (sessions.length === 0) {
      if (selectedSessionId) setSelectedSessionId("");
      return;
    }

    const selectedExists = sessions.some((session) => session.id === selectedSessionId);
    if (!selectedExists) setSelectedSessionId(sessions[0].id);
  }, [sessions, selectedSessionId]);

  useEffect(() => {
    if (logs.length === 0) {
      if (selectedLogKey) setSelectedLogKey("");
      return;
    }

    const filteredLogs = logLevelFilter === "all" ? logs : logs.filter((line) => line.level === logLevelFilter);
    const selectedExists = filteredLogs.some((line) => logLineKey(line) === selectedLogKey);
    if (!selectedExists) setSelectedLogKey(filteredLogs[0] ? logLineKey(filteredLogs[0]) : "");
  }, [logs, logLevelFilter, selectedLogKey]);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 5_000);
    return () => window.clearInterval(timer);
  }, []);

  const refreshSnapshots = useCallback(
    async ({ manual = false }: { manual?: boolean } = {}) => {
      if (!telegram.isTelegram || !telegram.initData || !apiConfigured) return;

      const requestId = refreshRequestId.current + 1;
      refreshRequestId.current = requestId;
      activeRefreshes.current += 1;
      if (manual) triggerTelegramRefreshHaptic("start");
      setRefreshing(true);
      setApiState((current) => (current === "connected" ? current : "connecting"));
      setEndpointHealth((current) => updateEndpointHealth(current, createEndpointStateUpdates("checking")));

      try {
        const [status, capabilities, approvals, sessions, logs] = await Promise.allSettled([
          fetchStatusSnapshot(),
          fetchCapabilitiesSnapshot(),
          fetchApprovalsSnapshot(),
          fetchSessionsSnapshot(),
          fetchLogsSnapshot(),
        ]);
        if (requestId !== refreshRequestId.current) return;

        // The action gate may open ONLY when every action-critical read of this
        // poll is fresh — status AND capabilities AND approvals. If any degraded,
        // drop the capability and stale approvals version so a previously-loaded
        // approve-action=true can never keep buttons live on stale data.
        const statusOk = status.status === "fulfilled";
        const actionReadsHealthy = statusOk && capabilities.status === "fulfilled" && approvals.status === "fulfilled";
        setServerCapabilities(actionReadsHealthy ? capabilities.value.items : []);
        if (!actionReadsHealthy) setApprovalsVersion("");

        // "committed" means this poll actually APPLIED the endpoint's payload to
        // displayed state, not merely that its request resolved. Section payloads
        // (approvals/sessions/logs) only commit when status also succeeds (they
        // live after the status early-return); the capability matrix is only
        // stored when actionReadsHealthy (fail-closed). Health and per-endpoint
        // last-ok both track committed state so a row never claims ok while the
        // screen shows nothing.
        const committed: Record<EndpointKey, boolean> = {
          status: statusOk,
          capabilities: actionReadsHealthy,
          approvals: statusOk && approvals.status === "fulfilled",
          sessions: statusOk && sessions.status === "fulfilled",
          logs: statusOk && logs.status === "fulfilled",
        };
        setEndpointHealth((current) =>
          updateEndpointHealth(
            current,
            endpointKeys.reduce(
              (acc, key) => ({ ...acc, [key]: committed[key] ? "ok" : "degraded" }),
              {} as Partial<Record<EndpointKey, "ok" | "degraded">>,
            ),
          ),
        );
        const pollAt = Date.now();
        setEndpointLastOk((current) =>
          endpointKeys.reduce(
            (acc, key) => ({ ...acc, [key]: committed[key] ? pollAt : current[key] }),
            {} as Record<EndpointKey, number | null>,
          ),
        );

        if (status.status !== "fulfilled") {
          setApiState("offline");
          if (manual) triggerTelegramRefreshHaptic("warning");
          return;
        }

        setSnapshot(status.value);
        if (approvals.status === "fulfilled") {
          const mappedApprovals = approvals.value.items.map(mapServerApproval);
          setServerApprovals(mappedApprovals);
          // Only keep a live approvals version when the whole action-critical
          // poll is healthy; otherwise it stays cleared (set above) so no stale
          // version can back a decision.
          if (actionReadsHealthy) setApprovalsVersion(approvals.value.snapshot_version ?? "");
          setSnapshotMeta((current) => ({ ...current, approvals: approvals.value.meta ?? null }));
          setSelectedApprovalId((current) => {
            if (mappedApprovals.some((approval) => approval.id === current)) return current;
            return mappedApprovals[0]?.id ?? "";
          });
        }
        if (sessions.status === "fulfilled") {
          const mappedSessions = sessions.value.items.map(mapServerSession);
          setServerSessions(mappedSessions);
          setSnapshotMeta((current) => ({ ...current, sessions: sessions.value.meta ?? null }));
          setSelectedSessionId((current) => {
            if (mappedSessions.some((session) => session.id === current)) return current;
            return mappedSessions[0]?.id ?? "";
          });
        }
        if (logs.status === "fulfilled") {
          const mappedLogs = logs.value.items.map(mapServerLog);
          setServerLogs(mappedLogs);
          setSnapshotMeta((current) => ({ ...current, logs: logs.value.meta ?? null }));
          setSelectedLogKey((current) => {
            if (mappedLogs.some((line) => logLineKey(line) === current)) return current;
            return mappedLogs[0] ? logLineKey(mappedLogs[0]) : "";
          });
        }
        const refreshedAt = Date.now();
        setLastSuccessAt(refreshedAt);
        setNow(refreshedAt);
        setApiState("connected");
        if (manual) triggerTelegramRefreshHaptic("success");
      } catch {
        if (requestId === refreshRequestId.current) {
          setApiState("offline");
          // Fail closed on a thrown refresh too: drop the action capability and
          // stale approvals version so buttons cannot stay live on an errored poll.
          setServerCapabilities([]);
          setApprovalsVersion("");
          setEndpointHealth((current) => updateEndpointHealth(current, createEndpointStateUpdates("degraded")));
          if (manual) triggerTelegramRefreshHaptic("warning");
        }
      } finally {
        activeRefreshes.current = Math.max(0, activeRefreshes.current - 1);
        if (activeRefreshes.current === 0) setRefreshing(false);
      }
    },
    [apiConfigured, telegram.initData, telegram.isTelegram],
  );

  useEffect(() => {
    if (!telegram.isTelegram || !telegram.initData || !apiConfigured) {
      setApiState("mock");
      setSnapshot(null);
      setServerApprovals([]);
      setServerSessions([]);
      setServerLogs([]);
      setServerCapabilities([]);
      setEndpointHealth(createEndpointHealth("preview"));
      setEndpointLastOk(endpointKeys.reduce((acc, key) => ({ ...acc, [key]: null }), {} as Record<EndpointKey, number | null>));
      setSnapshotMeta({ approvals: null, sessions: null, logs: null });
      setApprovalsVersion("");
      setIsActionOwner(false);
      setLastSuccessAt(null);
      setRefreshing(false);
      return;
    }

    let cancelled = false;
    setApiState("connecting");
    setRefreshing(true);
    setIsActionOwner(false);
    void authenticateTelegram(telegram.initData)
      .then((auth) => {
        if (cancelled) return undefined;
        // Owner eligibility comes from the server, not the mere presence of a
        // Telegram runtime.
        setIsActionOwner(Boolean(auth.is_action_owner));
        return refreshSnapshots();
      })
      .catch(() => {
        if (!cancelled) {
          setApiState("offline");
          setIsActionOwner(false);
          // Auth failed before any poll committed data: mark the endpoints
          // degraded so the sections show the outage notice instead of a blank
          // live tab with no explanation, and keep the action gate closed.
          setEndpointHealth((current) => updateEndpointHealth(current, createEndpointStateUpdates("degraded")));
          setServerCapabilities([]);
          setApprovalsVersion("");
          setRefreshing(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [apiConfigured, refreshSnapshots, telegram.initData, telegram.isTelegram]);

  useEffect(() => {
    if (!telegram.isTelegram || !telegram.initData || !apiConfigured) return undefined;
    const timer = window.setInterval(() => {
      void refreshSnapshots();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [apiConfigured, refreshSnapshots, telegram.initData, telegram.isTelegram]);

  // Effective gate: process-level capability AND server-authenticated owner AND
  // the live status payload confirming actions are enabled. Requiring the
  // status flag too means a status card that reads "safe/blocked" can never sit
  // over live action buttons — the two must agree or the gate stays closed.
  const actionsEnabled =
    approveActionEnabled(serverCapabilities) && isActionOwner && snapshot?.miniapp.actions_enabled === true;

  const actionApplication = snapshot?.miniapp.action_application ?? (snapshot?.miniapp.actions_enabled ? "record-only" : "not-wired");
  const snapshotFresh = lastSuccessAt !== null && now - lastSuccessAt <= STALE_AFTER_MS;
  const applicationCanRecord =
    actionApplication === "record-only" ||
    (actionApplication === "live" && snapshot?.miniapp.gateway_resolver_active === true);
  const canSubmitDecision = actionsEnabled && apiState === "connected" && snapshotFresh && applicationCanRecord && approvalsVersion.length > 0;
  const decisionBlockReason = canSubmitDecision
    ? ""
    : !isActionOwner
      ? "Только владелец может записывать решения."
      : !actionsEnabled
        ? "Контур решений выключен сервером."
        : apiState !== "connected"
          ? "Нет свежего соединения с Mini App API."
          : !snapshotFresh
            ? "Снимок данных устарел. Обнови очередь перед решением."
            : !applicationCanRecord
              ? "Применение gateway не подключено."
              : "Нет свежей версии очереди одобрений. Обнови снимок данных.";

  function decisionErrorMessage(error: unknown): string {
    if (error instanceof MiniAppApiError) {
      if (error.status === 401) return "Сессия Mini App истекла. Открой приложение заново или обнови данные.";
      if (error.status === 403) return "Сервер не подтвердил права владельца для этого решения.";
      if (error.status === 409) return "Очередь одобрений устарела. Я обновил снимок — проверь запрос и попробуй снова.";
      if (error.status === 429) return "Слишком много попыток. Подожди немного и попробуй снова.";
      return `Сервер отклонил решение (${error.status}). ${error.message}`.trim();
    }
    return "Не удалось записать решение. Проверь соединение и попробуй снова.";
  }

  const clearDecisionError = useCallback(() => setDecisionError(""), []);

  const submitApprovalDecision = useCallback(
    async (approvalId: string, decision: ApprovalDecision): Promise<boolean> => {
      setDecisionError("");
      if (!canSubmitDecision || !telegram.initData || !approvalsVersion) {
        setDecisionError(decisionBlockReason || "Нет свежей версии очереди одобрений. Обнови снимок данных.");
        return false;
      }
      try {
        await postApprovalDecision({
          approvalId,
          decision,
          // Idempotency key unique per approval+decision+snapshot; a retry of
          // the same decision replays the server's cached response.
          clientRequestId: `${approvalId}:${decision}:${approvalsVersion}`,
          snapshotVersion: approvalsVersion,
          initData: telegram.initData,
        });
        triggerTelegramRefreshHaptic("success");
        await refreshSnapshots();
        return true;
      } catch (error) {
        setDecisionError(decisionErrorMessage(error));
        triggerTelegramRefreshHaptic("warning");
        await refreshSnapshots();
        return false;
      }
    },
    [approvalsVersion, canSubmitDecision, decisionBlockReason, refreshSnapshots, telegram.initData],
  );

  return {
    actionsEnabled,
    canSubmitDecision,
    decisionBlockReason,
    decisionError,
    clearDecisionError,
    isActionOwner,
    submitApprovalDecision,
    snapshot,
    apiState,
    approvals,
    sessions,
    logs,
    serverCapabilities,
    endpointHealth,
    endpointLastOk,
    snapshotMeta,
    isRefreshing,
    lastSuccessAt,
    now,
    hasServerSnapshot,
    refreshSnapshots,
    selectedApprovalId,
    setSelectedApprovalId,
    selectedSessionId,
    setSelectedSessionId,
    selectedLogKey,
    setSelectedLogKey,
    logLevelFilter,
    setLogLevelFilter,
  };
}

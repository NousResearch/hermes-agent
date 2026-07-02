import { useCallback, useEffect, useRef, useState } from "react";
import {
  authenticateTelegram,
  fetchApprovalsSnapshot,
  fetchCapabilitiesSnapshot,
  fetchLogsSnapshot,
  fetchSessionsSnapshot,
  fetchStatusSnapshot,
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
  STORAGE_KEYS,
  createEndpointHealth,
  createEndpointStateUpdates,
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
  const [endpointHealth, setEndpointHealth] = useState<Record<EndpointKey, EndpointHealthItem>>(() => createEndpointHealth("preview"));
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

        setEndpointHealth((current) => updateEndpointHealth(current, {
          status: status.status === "fulfilled" ? "ok" : "degraded",
          capabilities: capabilities.status === "fulfilled" ? "ok" : "degraded",
          approvals: approvals.status === "fulfilled" ? "ok" : "degraded",
          sessions: sessions.status === "fulfilled" ? "ok" : "degraded",
          logs: logs.status === "fulfilled" ? "ok" : "degraded",
        }));

        if (status.status !== "fulfilled") {
          setApiState("offline");
          if (manual) triggerTelegramRefreshHaptic("warning");
          return;
        }

        setSnapshot(status.value);
        if (approvals.status === "fulfilled") {
          const mappedApprovals = approvals.value.items.map(mapServerApproval);
          setServerApprovals(mappedApprovals);
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
        if (capabilities.status === "fulfilled") {
          setServerCapabilities(capabilities.value.items);
        }
        const refreshedAt = Date.now();
        setLastSuccessAt(refreshedAt);
        setNow(refreshedAt);
        setApiState("connected");
        if (manual) triggerTelegramRefreshHaptic("success");
      } catch {
        if (requestId === refreshRequestId.current) {
          setApiState("offline");
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
      setSnapshotMeta({ approvals: null, sessions: null, logs: null });
      setLastSuccessAt(null);
      setRefreshing(false);
      return;
    }

    let cancelled = false;
    setApiState("connecting");
    setRefreshing(true);
    void authenticateTelegram(telegram.initData)
      .then(() => {
        if (!cancelled) return refreshSnapshots();
        return undefined;
      })
      .catch(() => {
        if (!cancelled) {
          setApiState("offline");
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

  return {
    snapshot,
    apiState,
    approvals,
    sessions,
    logs,
    serverCapabilities,
    endpointHealth,
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

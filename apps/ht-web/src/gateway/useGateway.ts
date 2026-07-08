import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import {
  JsonRpcGatewayClient,
  buildHermesWebSocketUrl,
  type ConnectionState,
  type GatewayEvent,
} from "@hermes/shared";
import {
  chatReducer,
  initialChatState,
  stateFromTranscript,
} from "./chatReducer";
import { DEFAULT_SKIN, applySkinVars, resolveSkin, type ResolvedSkin } from "./skin";
import type {
  GatewaySkin,
  SessionCreateResponse,
  SessionListItem,
  SessionListResponse,
  SessionResumeResponse,
} from "./types";

// Events the chat reducer consumes. Everything else is ignored for the MVP.
const CHAT_EVENTS = new Set([
  "message.start",
  "message.delta",
  "message.complete",
  "status.update",
  "tool.start",
  "tool.progress",
  "tool.complete",
  "clarify.request",
  "approval.request",
  "error",
]);

function readInjectedToken(): string | undefined {
  if (typeof window === "undefined") return undefined;
  const w = window as unknown as Record<string, string | undefined>;
  return w.__HT_SESSION_TOKEN__ ?? w.__HERMES_SESSION_TOKEN__;
}

function gatewayWsUrl(): string {
  const token = readInjectedToken();
  return buildHermesWebSocketUrl({
    path: "/api/ws",
    authParam: token ? ["token", token] : undefined,
  });
}

export interface UseGateway {
  connection: ConnectionState;
  skin: ResolvedSkin;
  sessionId: string | null;
  chat: ReturnType<typeof chatReducer>;
  sessions: SessionListItem[];
  submit: (text: string) => Promise<void>;
  interrupt: () => Promise<void>;
  respondClarify: (answer: string) => Promise<void>;
  respondApproval: (choice: string, all?: boolean) => Promise<void>;
  newSession: () => Promise<void>;
  resumeSession: (id: string) => Promise<void>;
  refreshSessions: () => Promise<void>;
}

export function useGateway(): UseGateway {
  const clientRef = useRef<JsonRpcGatewayClient | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("idle");
  const [skin, setSkin] = useState<ResolvedSkin>(DEFAULT_SKIN);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [chat, dispatch] = useReducer(chatReducer, initialChatState);

  // The active session id, read inside event handlers without re-subscribing.
  const activeSession = useRef<string | null>(null);
  activeSession.current = sessionId;

  const getClient = useCallback((): JsonRpcGatewayClient => {
    if (!clientRef.current) {
      clientRef.current = new JsonRpcGatewayClient({ requestIdPrefix: "ht" });
    }
    return clientRef.current;
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      const res = await getClient().request<SessionListResponse>("session.list");
      setSessions(res.sessions ?? []);
    } catch {
      // Non-fatal: sidebar just stays empty.
    }
  }, [getClient]);

  const newSession = useCallback(async () => {
    const res = await getClient().request<SessionCreateResponse>("session.create", {});
    dispatch({ type: "reset" });
    setSessionId(res.session_id);
    void refreshSessions();
  }, [getClient, refreshSessions]);

  const resumeSession = useCallback(
    async (id: string) => {
      const res = await getClient().request<SessionResumeResponse>("session.resume", {
        session_id: id,
      });
      setSessionId(res.session_id);
      // Replace chat state with the resumed transcript.
      const seeded = stateFromTranscript(res.messages ?? []);
      dispatch({ type: "reset" });
      for (const m of seeded.messages) {
        if (m.role === "user") {
          dispatch({ type: "userSubmitted", text: m.text });
        } else {
          dispatch({ type: "event", name: "message.start", payload: undefined });
          dispatch({ type: "event", name: "message.complete", payload: { text: m.text } });
        }
      }
    },
    [getClient],
  );

  // Connect once on mount; apply skin from gateway.ready / skin.changed.
  useEffect(() => {
    const client = getClient();
    const offState = client.onState(setConnection);

    const applySkinEvent = (payload: GatewaySkin | { skin?: GatewaySkin } | undefined) => {
      const raw = (payload && "skin" in payload ? payload.skin : payload) as
        | GatewaySkin
        | undefined;
      const resolved = resolveSkin(raw);
      setSkin(resolved);
      applySkinVars(resolved);
    };

    const offReady = client.on("gateway.ready", (e: GatewayEvent) => {
      applySkinEvent(e.payload as { skin?: GatewaySkin });
    });
    const offSkin = client.on("skin.changed", (e: GatewayEvent) => {
      applySkinEvent(e.payload as GatewaySkin);
    });

    const offChat = client.onAny((e: GatewayEvent) => {
      if (!CHAT_EVENTS.has(e.type)) return;
      // Filter events by the active session (gateway can drive several).
      if (e.session_id && activeSession.current && e.session_id !== activeSession.current) {
        return;
      }
      dispatch({ type: "event", name: e.type, payload: e.payload });
    });

    let cancelled = false;
    client
      .connect(gatewayWsUrl())
      .then(async () => {
        if (cancelled) return;
        await refreshSessions();
        await newSession();
      })
      .catch(() => {
        // onState already surfaced 'error'; the UI shows a retry affordance.
      });

    return () => {
      cancelled = true;
      offState();
      offReady();
      offSkin();
      offChat();
      client.close();
      clientRef.current = null;
    };
    // Mount-only: getClient/refreshSessions/newSession are stable via useCallback.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submit = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || !activeSession.current) return;
      dispatch({ type: "userSubmitted", text: trimmed });
      await getClient().request("prompt.submit", {
        session_id: activeSession.current,
        text: trimmed,
      });
    },
    [getClient],
  );

  const interrupt = useCallback(async () => {
    if (!activeSession.current) return;
    await getClient().request("session.interrupt", { session_id: activeSession.current });
  }, [getClient]);

  const respondClarify = useCallback(
    async (answer: string) => {
      const req = chat.clarify;
      if (!req || !activeSession.current) return;
      dispatch({ type: "event", name: "clarify.resolved", payload: {} });
      await getClient().request("clarify.respond", {
        session_id: activeSession.current,
        request_id: req.requestId,
        answer,
      });
    },
    [getClient, chat.clarify],
  );

  const respondApproval = useCallback(
    async (choice: string, all = false) => {
      if (!activeSession.current) return;
      dispatch({ type: "event", name: "approval.resolved", payload: {} });
      await getClient().request("approval.respond", {
        session_id: activeSession.current,
        choice,
        all,
      });
    },
    [getClient],
  );

  return {
    connection,
    skin,
    sessionId,
    chat,
    sessions,
    submit,
    interrupt,
    respondClarify,
    respondApproval,
    newSession,
    resumeSession,
    refreshSessions,
  };
}

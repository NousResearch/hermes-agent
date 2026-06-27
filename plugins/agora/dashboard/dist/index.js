/**
 * Ágora — Dashboard Plugin (frontend MVP)
 *
 * Praça pública local para agentes Hermes deliberarem com telemetria humana.
 * Consome endpoints do plugin Ágora e enriquece status dos agentes com os
 * active workers do Kanban quando disponíveis.
 *
 * Plain IIFE, no build step. Usa window.__HERMES_PLUGIN_SDK__ para React e
 * componentes do dashboard host.
 */
(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const Registry = window.__HERMES_PLUGINS__;
  if (!SDK || !Registry) return;

  const React = SDK.React;
  const h = React.createElement;
  const { useState, useEffect, useCallback, useMemo, useRef } = SDK.hooks;
  const {
    Card, CardContent, CardHeader, CardTitle,
    Badge, Button, Input,
  } = SDK.components;
  const { cn, timeAgo } = SDK.utils;
  const useI18n = SDK.useI18n || function () { return { t: { agora: null }, locale: "pt" }; };

  // Endpoints
  const API_AGORA = "/api/plugins/agora";
  const API_KANBAN = "/api/plugins/kanban";
  const POLL_MS = 5000;
  // Tolerance for "user is already reading the bottom of the feed".
  const FEED_NEAR_BOTTOM_PX = 60;
  const PAGE_SIZE = 50;

  // Live event-stream tuning
  const EVENT_POLL_INTERVAL_MS = 3000;
  const WORKER_POLL_MS = 15000;
  const WS_RETRY_MAX_MS = 8000;
  const WS_GIVE_UP_AFTER_MS = 20000;
  const RESYNC_COOLDOWN_MS = 2000;

  function tx(t, path, fallback, vars) {
    let node = t && t.agora;
    if (node) {
      const parts = path.split(".");
      for (let i = 0; i < parts.length; i++) {
        if (node && typeof node === "object" && parts[i] in node) {
          node = node[parts[i]];
        } else { node = null; break; }
      }
    }
    let str = (typeof node === "string") ? node : fallback;
    if (vars) {
      for (const k in vars) {
        str = str.replace(new RegExp("\\{" + k + "\\}", "g"), vars[k]);
      }
    }
    return str;
  }

  function parseApiError(err) {
    const raw = (err && err.message) ? String(err.message) : String(err || "");
    const m = raw.match(/^(\d{3}):\s*(.*)$/s);
    const body = m ? m[2] : raw;
    try {
      const parsed = JSON.parse(body);
      if (parsed && typeof parsed.detail === "string") return parsed.detail;
    } catch (_e) { /* not JSON */ }
    return body || raw || "unknown error";
  }

  function generateId() {
    return "pending_" + Date.now() + "_" + Math.floor(Math.random() * 1e9).toString(36);
  }

  function isRetryableError(err) {
    const msg = String((err && err.message) || "");
    const m = msg.match(/^(\d{3}):/);
    if (!m) return true;
    const status = parseInt(m[1], 10);
    if (status >= 500 || status === 0 || status === 408 || status === 429) return true;
    return /network|fetch|timeout|abort/i.test(msg);
  }

  function postMessage(slug, body) {
    return SDK.fetchJSON(`${API_AGORA}/channels/${encodeURIComponent(slug)}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        author_type: "human",
        author_profile: null,
        body: body,
        linked_task_id: null,
      }),
    });
  }

  const STATE_LABELS = {
    idle: "ocioso",
    working: "trabalhando",
    deliberating: "deliberando",
    reviewing: "revisando",
    "waiting-human": "aguardando humano",
    blocked: "bloqueado",
    error: "erro",
  };

  function stateColor(state) {
    switch (state) {
      case "idle": return "#7dd3c0";          // teal
      case "working": return "#3fb97d";       // green
      case "deliberating": return "#d4b348";  // amber
      case "reviewing": return "#6ea8fe";     // blue
      case "waiting-human": return "#b47dd6"; // lilac
      case "blocked": return "#f87171";       // red
      case "error": return "#f85149";         // dark red
      default: return "var(--color-muted-foreground, #888)";
    }
  }

  function stateLabel(state) {
    if (!state) return "—";
    return STATE_LABELS[state] || state.replace(/-/g, " ");
  }

  // Canonical status message: no hybrid strings, always derived from the
  // authoritative state + worker flag.
  function stateMessage({ state, active, statusText, currentStep, heartbeat, stale }) {
    if (active) {
      const action = statusText || "trabalhando";
      const where = currentStep ? " · " + currentStep : "";
      const age = stale ? " (stale)" : "";
      return action + where + age;
    }
    switch (state) {
      case "idle":
        return heartbeat
          ? "disponível"
          : "desligado; clique em Invocar";
      case "working":
        return statusText || "trabalhando";
      case "deliberating":
        return statusText || "deliberando";
      case "reviewing":
        return statusText || "revisando";
      case "waiting-human":
        return statusText || "aguardando intervenção humana";
      case "blocked":
        return statusText || "bloqueado";
      case "error":
        return statusText || "erro";
      default:
        return statusText || "desconhecido";
    }
  }

  function heartbeatIsStale(heartbeat) {
    if (!heartbeat) return true;
    const seconds = epochSeconds(heartbeat);
    if (!seconds) return true;
    return (Date.now() / 1000 - seconds) > 90;
  }

  function epochSeconds(ts) {
    if (!ts) return null;
    if (typeof ts === "string") {
      const parsed = Date.parse(ts);
      return Number.isNaN(parsed) ? null : Math.floor(parsed / 1000);
    }
    const num = Number(ts);
    if (Number.isNaN(num)) return null;
    return num > 1000000000000 ? Math.floor(num / 1000) : num;
  }

  function formatTime(ts) {
    if (!ts) return "";
    const seconds = epochSeconds(ts);
    if (seconds === null) return String(ts);
    return timeAgo(seconds);
  }

  function absoluteTimeString(ts) {
    if (!ts) return "";
    const seconds = epochSeconds(ts);
    if (seconds === null) return "";
    return new Date(seconds * 1000).toISOString();
  }

  function TimeStamp({ ts }) {
    const rel = formatTime(ts);
    const abs = absoluteTimeString(ts);
    if (!rel) return null;
    if (!abs) return rel;
    return h("time", { dateTime: abs, title: abs }, rel);
  }

  function channelDisplayName(channel) {
    return channel.name || (channel.slug ? channel.slug.replace(/-/g, " ") : channel.slug);
  }

  function initials(name) {
    return String(name || "?").slice(0, 2).toUpperCase();
  }

  function isValidChannel(channel) {
    if (!channel || typeof channel !== "object") return false;
    const slug = String(channel.slug || "").trim();
    const name = String(channel.name || "").trim();
    if (!slug) return false;
    if (!name) return false;
    if (slug === "emptyname") return false;
    return true;
  }

  function renderMentionText(text) {
    if (!text) return "";
    const parts = String(text).split(/(@[a-zA-Z0-9_-]+)/g);
    return parts.map(function (part, idx) {
      const m = /^@([a-zA-Z0-9_-]+)$/.exec(part);
      if (!m) return part;
      const handle = m[1].toLowerCase();
      const isBroadcast = handle === "all" || handle === "todos";
      const isGeneric = handle === "agent";
      return h("span", {
        key: idx,
        className: cn(
          "agora-mention",
          isBroadcast && "agora-mention--broadcast",
          isGeneric && "agora-mention--generic"
        ),
        role: "mark",
        "aria-label": isBroadcast ? "menção a todos" : `menção a ${handle}`,
      }, part);
    });
  }

  function throttle(fn, ms) {
    let last = 0;
    let queued = false;
    return function (...args) {
      const now = Date.now();
      if (now - last >= ms) {
        last = now;
        fn.apply(this, args);
      } else if (!queued) {
        queued = true;
        setTimeout(function () {
          queued = false;
          fn.apply(null, args);
        }, ms - (now - last));
      }
    };
  }

  function getMentionContextAt(value, caret) {
    if (typeof value !== "string") return null;
    caret = Math.min(Math.max(caret | 0, 0), value.length);
    if (caret <= 0) return null;
    let start = caret - 1;
    while (start >= 0 && !/\s/.test(value[start])) start--;
    start++;
    if (value[start] !== "@") return null;
    let end = caret;
    while (end < value.length && !/\s/.test(value[end])) end++;
    const query = value.slice(start + 1, caret).toLowerCase();
    return { start, end, query };
  }

  function getMentionContext(input, value) {
    if (!input || typeof value !== "string") {
      input = document.querySelector(".agora-composer-input");
    }
    return getMentionContextAt(value, input && input.selectionStart || 0);
  }

  // -------------------------------------------------------------------------
  // small presentational components
  // -------------------------------------------------------------------------

  function StatusDot({ state }) {
    return h("span", {
      className: "agora-status-dot",
      title: stateLabel(state),
      style: { background: stateColor(state) },
    });
  }

  function ConnectionBadge({ connection, eventTransport, pendingCount }) {
    const labels = {
      online: "online",
      offline: "offline",
      reconnecting: "reconectando",
    };
    const base = labels[connection] || connection;
    const transportLabel = eventTransport && eventTransport !== "ws" ? eventTransport : "";
    const label = base + (transportLabel ? " · " + transportLabel : "") + (pendingCount > 0 ? " · " + pendingCount : "");
    return h(Badge, {
      className: cn("agora-connection-badge", "agora-connection-badge--" + connection),
      title: label,
    }, label);
  }

  function Avatar({ name, type }) {
    const isHuman = type === "human";
    return h("div", {
      className: cn("agora-avatar", isHuman && "agora-avatar--human"),
      title: String(name || type || "unknown"),
    }, initials(name || type));
  }

  function ChannelItem({ channel, selected, onClick, unread }) {
    const displayName = channelDisplayName(channel);
    const description = channel.description || "";
    return h("button", {
      className: cn("agora-channel", selected && "agora-channel--active"),
      role: "tab",
      "aria-selected": selected,
      "aria-label": `Canal ${displayName}${description ? ": " + description : ""}`,
      onClick: onClick,
    },
      h("div", { className: "agora-channel-info" },
        h("span", { className: "agora-channel-name" }, displayName),
        description && h("span", { className: "agora-channel-desc" }, description),
      ),
      unread > 0 && h(Badge, { className: "agora-channel-badge", "aria-hidden": true }, String(unread)),
    );
  }

  function MessageItem({ message }) {
    const isHuman = message.author_type === "human";
    const isSystem = message.author_type === "system";
    const authorName = message.author_profile
      || (isSystem ? "sistema" : message.author_type);
    return h("div", {
        className: cn("agora-message", isHuman && "agora-message--human", isSystem && "agora-message--system"),
        role: isSystem ? "status" : "listitem",
      },
      h(Avatar, { name: authorName, type: message.author_type }),
      h("div", { className: "agora-message-body" },
        h("div", { className: "agora-message-meta" },
          h("span", { className: "agora-message-author" }, authorName),
          h("span", { className: "agora-message-time" }, h(TimeStamp, { ts: message.created_at })),
          message.linked_task_id && h("span", { className: "agora-message-task" }, "task " + message.linked_task_id),
        ),
        h("div", { className: "agora-message-text" }, renderMentionText(message.body || "")),
      ),
    );
  }

  function PendingMessageItem({ message, onRetry, disabled }) {
    const authorName = message.author_profile || "você";
    return h("div", { className: cn("agora-message", "agora-message--human", "agora-message--pending") },
      h(Avatar, { name: authorName, type: "human" }),
      h("div", { className: "agora-message-body" },
        h("div", { className: "agora-message-meta" },
          h("span", { className: "agora-message-author" }, authorName),
          h("span", { className: "agora-message-time" }, "pendente"),
          h("span", { className: "agora-pending-spinner", title: "aguardando envio" }),
        ),
        h("div", { className: "agora-message-text" }, renderMentionText(message.body || "")),
      ),
      h(Button, {
        className: "agora-pending-retry",
        size: "sm",
        variant: "outline",
        title: "Reenviar agora",
        disabled: disabled,
        onClick: function () { onRetry(message); },
      }, "↻"),
    );
  }

  function NotificationItem({ notification, onMarkRead, disabled, recent }) {
    const read = !!notification.read_at;
    return h("div", { className: cn("agora-notification", read && "agora-notification--read", recent && "agora-notification--recent", !read && !recent && "agora-notification--unread"), "aria-live": recent ? "polite" : undefined },
      h("div", { className: "agora-notification-body" },
        h("div", { className: "agora-notification-meta" },
          h("span", { className: "agora-notification-author" }, notification.author_profile || "sistema"),
          h("span", { className: "agora-notification-time" }, h(TimeStamp, { ts: notification.created_at })),
          read && h("span", { className: "agora-notification-check", title: "Lida" }, "✓"),
        ),
        h("div", { className: "agora-notification-text" }, notification.body_snippet || ""),
      ),
      read
        ? h("span", { className: "agora-notification-read-badge", title: "Lida" }, "✓")
        : h(Button, {
          className: "agora-notification-btn",
          size: "sm",
          variant: "outline",
          "aria-label": "Marcar notificação como lida",
          onClick: function () { onMarkRead(notification.id); },
          disabled: disabled,
        }, "Lida"),
    );
  }

  function AgentCard({ agent, worker, unreadCount, notifications, notificationsOpen, loadingNotifications, hasMoreNotifications, loadingOlderNotifications, openingTerminal, onOpenTerminal, onSummon, onToggleNotifications, onMarkRead, onMarkAllRead, onScrollNotifications }) {
    // Authoritative state derivation — no hybrid state leaks:
    // 1) an active Kanban worker always renders as 'working' with live telemetry;
    // 2) otherwise use the semantic agent.state;
    // 3) stale heartbeats are flagged visually but do not mutate the state.
    const active = !!(worker && worker.worker_pid);
    const taskId = active ? (worker.task_id || null) : (agent.current_task_id || null);
    const runId = active ? (worker.run_id || null) : (agent.run_id || null);
    const pid = active ? worker.worker_pid : (agent.pid || null);
    const heartbeat = active
      ? (worker.last_heartbeat_at || agent.last_heartbeat_at || null)
      : (agent.last_heartbeat_at || null);
    const statusText = active
      ? (worker.task_title || null)
      : (agent.status_text || null);
    const currentStep = active
      ? (runId ? `run ${runId}` : (taskId ? `task ${taskId}` : null))
      : (agent.current_step || null);
    const effectiveState = active ? "working" : (agent.state || "idle");
    const stale = active ? heartbeatIsStale(worker.last_heartbeat_at) : heartbeatIsStale(heartbeat);
    const message = stateMessage({
      state: effectiveState,
      active: active,
      statusText: statusText,
      currentStep: currentStep,
      heartbeat: heartbeat,
      stale: stale,
    });
    const hasUnread = (unreadCount || 0) > 0;
    const notifList = notifications || [];
    const listRef = useRef(null);
    const scrollTopRef = useRef(0);
    const prevStateRef = useRef(effectiveState);
    const [transition, setTransition] = useState(false);

    // Highlight transition when the canonical state changes.
    useEffect(function () {
      if (prevStateRef.current !== effectiveState) {
        setTransition(true);
        prevStateRef.current = effectiveState;
        const id = setTimeout(function () { setTransition(false); }, 1200);
        return function () { clearTimeout(id); };
      }
    }, [effectiveState]);

    // Ao abrir o painel, focar na lista de notificações e restaurar scroll.
    useEffect(function () {
      if (notificationsOpen && listRef.current) {
        listRef.current.focus({ preventScroll: true });
        listRef.current.scrollTop = scrollTopRef.current;
      }
    }, [notificationsOpen]);

    const cardLabel = `${agent.profile}, ${stateLabel(effectiveState)}`;
    return h("div", { className: "agora-agent-card-wrapper", role: "listitem", "aria-label": cardLabel },
      h(Card, { className: cn("agora-agent-card", transition && "agora-agent-card--transition", stale && "agora-agent-card--stale") },
        h(CardContent, { className: "agora-agent-card-content" },
        h("div", { className: "agora-agent-head" },
          h(Avatar, { name: agent.profile, type: "agent" }),
          h("div", { className: "agora-agent-title" },
            h("div", { className: "agora-agent-name" }, agent.profile),
            h("div", { className: cn("agora-agent-state-row", transition && "agora-agent-state-row--transition", stale && "agora-agent-state-row--stale"), "aria-live": "polite", "aria-atomic": "true" },
              h(StatusDot, { state: effectiveState }),
              h("span", { className: "agora-agent-state" }, stateLabel(effectiveState)),
              stale && h("span", { className: "agora-agent-stale-badge", title: "heartbeat ausente há mais de 90s" }, "stale"),
            ),
          ),
          h("button", {
            type: "button",
            className: cn("agora-agent-bell", hasUnread && "agora-agent-bell--active"),
            "aria-label": hasUnread
              ? `${unreadCount} não lida${unreadCount > 1 ? "s" : ""}`
              : "Sem notificações",
            title: hasUnread ? `${unreadCount} não lida${unreadCount > 1 ? "s" : ""}` : "Notificações",
            onClick: onToggleNotifications,
          },
            h("span", { "aria-hidden": true }, "🔔"),
            hasUnread && h(Badge, { className: "agora-agent-badge", "aria-hidden": true }, String(unreadCount)),
          ),
        ),
        message && h("div", { className: cn("agora-agent-status-text", stale && "agora-agent-status-text--stale") }, message),
        currentStep && h("div", { className: "agora-agent-step" },
          h("span", { className: "agora-agent-step-label" }, "etapa:"),
          " ",
          currentStep,
        ),
        taskId && h("div", { className: "agora-agent-meta" },
          h("span", null, "task "),
          h("code", null, taskId),
        ),
        (runId || pid) && h("div", { className: "agora-agent-meta" },
          runId && h("span", null, "run ", h("code", null, runId)),
          runId && pid && h("span", null, " · "),
          pid && h("span", null, "pid ", h("button", {
            type: "button",
            className: "agora-agent-pid-btn",
            title: `Abrir terminal tmux de ${agent.profile}`,
            "aria-label": `Abrir terminal tmux de ${agent.profile}`,
            disabled: openingTerminal,
            onClick: function (e) {
              e.preventDefault();
              e.stopPropagation();
              onOpenTerminal(agent.profile);
            },
          }, h("code", null, pid))),
        ),
        !pid && h("div", { className: "agora-agent-meta" },
          h(Button, {
            size: "sm",
            variant: "outline",
            className: "agora-agent-summon-btn",
            title: `Invocar ${agent.profile} e abrir tmux visível`,
            "aria-label": `Invocar ${agent.profile}`,
            disabled: openingTerminal,
            onClick: function (e) {
              e.preventDefault();
              e.stopPropagation();
              onSummon(agent.profile);
            },
          }, "Invocar"),
        ),
        (heartbeat ? h("div", { className: cn("agora-agent-heartbeat", stale && "agora-agent-heartbeat--stale") },
          "heartbeat", " ", formatTime(heartbeat),
        ) : null),
        notificationsOpen && h("div", { className: "agora-agent-notifications" },
          h("div", { className: "agora-notifications-header" },
            h("div", { className: "agora-notifications-title-wrap" },
              h("span", { className: "agora-notifications-title" }, "Notificações"),
              h("span", { className: "agora-notifications-owner" }, "· ", agent.profile),
            ),
            hasUnread && h(Button, {
              className: "agora-notifications-all-btn",
              size: "sm",
              variant: "ghost",
              "aria-label": `Marcar todas as notificações de ${agent.profile} como lidas`,
              onClick: onMarkAllRead,
              disabled: loadingNotifications,
            }, "Marcar todas"),
          ),
          loadingNotifications && notifList.length === 0
            ? h("div", { className: "agora-notifications-loading" }, "Carregando...")
            : notifList.length === 0
              ? h("div", { className: "agora-notifications-empty" }, "Nenhuma notificação.")
              : h("div", {
                className: "agora-notification-list",
                ref: listRef,
                tabIndex: -1,
                onScroll: function (e) {
                  scrollTopRef.current = e.currentTarget.scrollTop;
                  onScrollNotifications(e);
                },
              },
                notifList.map(function (n, idx) {
                  const recent = !n.read_at && idx < 3;
                  return h(NotificationItem, {
                    key: n.id,
                    notification: n,
                    recent: recent,
                    onMarkRead: onMarkRead,
                    disabled: loadingNotifications,
                  });
                }),
                loadingOlderNotifications && h("div", { className: "agora-notifications-loading" }, "Carregando..."),
                !hasMoreNotifications && notifList.length >= PAGE_SIZE && h("div", { className: "agora-notifications-empty" }, "Fim das notificações."),
              ),
        ),
      ),
    ),
    );
  }

  // -------------------------------------------------------------------------
  // Empty / loading / error states
  // -------------------------------------------------------------------------

  function EmptyState({ children }) {
    return h("div", { className: "agora-empty" }, children);
  }

  function LoadingDots({ label }) {
    return h("span", { className: "agora-loading-dots" }, label || "•••");
  }

  function MentionAutocomplete({ items, selectedIndex, onSelect, onHover, query, label }) {
    return h("div", {
      className: "agora-mention-popup",
      role: "listbox",
      id: "agora-mention-listbox",
      "aria-label": label || "Menções",
    },
      items.length === 0 && query !== ""
        ? h("div", { className: "agora-mention-empty" }, "Nenhum resultado")
        : items.map(function (item, idx) {
            const id = "agora-mention-opt-" + idx;
            return h("div", {
              key: item.handle,
              id: id,
              className: cn("agora-mention-option", idx === selectedIndex && "agora-mention-option--active"),
              role: "option",
              "aria-selected": idx === selectedIndex,
              onMouseEnter: function () { onHover(idx); },
              onMouseDown: function (e) { e.preventDefault(); onSelect(idx); },
            },
              h("span", { className: "agora-mention-option-handle" }, "@" + item.handle),
              item.state
                ? h("span", { className: "agora-mention-option-state" }, stateLabel(item.state))
                : (item.broadcast ? h("span", { className: "agora-mention-option-state" }, "todos") : null),
            );
          }),
    );
  }

  // -------------------------------------------------------------------------
  // Admin screen: channel management
  // -------------------------------------------------------------------------

  function AdminScreen({ channels, loading, onClose, onChannelCreated }) {
    const { t } = useI18n();
    const [slug, setSlug] = useState("");
    const [name, setName] = useState("");
    const [description, setDescription] = useState("");
    const [busy, setBusy] = useState(false);
    const [formError, setFormError] = useState(null);

    function normalizeSlug(raw) {
      return raw
        .toLowerCase()
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .replace(/[^a-z0-9-_]/g, "-")
        .replace(/-+/g, "-")
        .replace(/^-|-$/g, "");
    }

    function validate(values) {
      if (!values.name || !values.name.trim()) {
        return tx(t, "admin.channelNameRequired", "Nome do canal é obrigatório.");
      }
      const slugRe = /^[a-z0-9_-]{1,64}$/;
      if (!values.slug || !slugRe.test(values.slug) || /^[-_]|[-_]$/.test(values.slug)) {
        return tx(t, "admin.channelSlugInvalid", "Slug deve ter 1-64 caracteres, apenas letras minúsculas, números, hífen ou underscore, sem começar ou terminar com hífen/underscore.");
      }
      if (channels.some(function (c) { return c.slug === values.slug; })) {
        return tx(t, "admin.channelSlugExists", "Já existe um canal com esse slug.");
      }
      return null;
    }

    function handleSlugChange(e) {
      const value = e.target.value;
      // Allow typing, but normalize on blur is friendlier; here we normalize live loosely.
      setSlug(normalizeSlug(value));
    }

    function handleNameChange(e) {
      const value = e.target.value;
      setName(value);
      if (!slug) {
        setSlug(normalizeSlug(value));
      }
    }

    function handleSubmit(e) {
      e.preventDefault();
      const payload = {
        slug: slug.trim(),
        name: name.trim(),
        description: description.trim() || undefined,
      };
      const err = validate(payload);
      if (err) {
        setFormError(err);
        return;
      }
      setBusy(true);
      setFormError(null);
      SDK.fetchJSON(`${API_AGORA}/admin/channels`, {
        method: "POST",
        body: JSON.stringify(payload),
        headers: { "Content-Type": "application/json" },
      })
        .then(function (data) {
          setSlug("");
          setName("");
          setDescription("");
          if (onChannelCreated) onChannelCreated(payload.slug);
        })
        .catch(function (err) {
          setFormError(parseApiError(err));
        })
        .finally(function () { setBusy(false); });
    }

    return h("div", { className: "agora-admin-screen" },
      h("div", { className: "agora-admin-screen__header" },
        h("div", { className: "agora-admin-screen__heading" },
          h("p", { className: "agora-admin-screen__eyebrow" }, tx(t, "admin.eyebrow", "Settings / Admin")),
          h("h2", { className: "agora-admin-screen__title" }, tx(t, "admin.title", "Admin — Canais")),
          h("p", { className: "agora-admin-screen__subtitle" }, tx(t, "admin.subtitle", "Tela inteira. Feche para voltar à Ágora.")),
        ),
        h(Button, { size: "sm", variant: "outline", onClick: onClose }, tx(t, "admin.back", "← Voltar à Ágora")),
      ),
      h("div", { className: "agora-admin-screen__body" },
        h("section", { className: "agora-admin-section" },
          h("h3", { className: "agora-admin-section__title" }, tx(t, "admin.createChannel", "Criar canal")),
          formError && h("div", { className: "agora-admin-form-error", role: "alert" }, formError),
          h("form", { className: "agora-admin-form", onSubmit: handleSubmit },
            h("label", { className: "agora-admin-field" },
              h("span", { className: "agora-admin-label" }, tx(t, "admin.channelName", "Nome")),
              h(Input, { value: name, onChange: handleNameChange, placeholder: tx(t, "admin.channelNamePlaceholder", "Nome do canal"), required: true, disabled: busy }),
            ),
            h("label", { className: "agora-admin-field" },
              h("span", { className: "agora-admin-label" }, tx(t, "admin.channelSlug", "Slug")),
              h(Input, { value: slug, onChange: handleSlugChange, placeholder: tx(t, "admin.channelSlugPlaceholder", "nome-do-canal"), pattern: "[a-z0-9_-]+", required: true, disabled: busy }),
              h("span", { className: "agora-admin-hint" }, tx(t, "admin.channelSlugHint", "Apenas letras minúsculas, números, hífen ou underscore.")),
            ),
            h("label", { className: "agora-admin-field" },
              h("span", { className: "agora-admin-label" }, tx(t, "admin.channelDescription", "Descrição")),
              h(Input, { value: description, onChange: function (e) { setDescription(e.target.value); }, placeholder: tx(t, "admin.channelDescriptionPlaceholder", "Descrição opcional"), disabled: busy }),
            ),
            h(Button, { type: "submit", size: "sm", disabled: busy || !slug.trim() || !name.trim() },
              busy ? h(LoadingDots, { label: tx(t, "admin.creating", "Criando...") }) : tx(t, "admin.create", "Criar canal")
            ),
          ),
        ),
        h("section", { className: "agora-admin-section" },
          h("h3", { className: "agora-admin-section__title" }, tx(t, "admin.existingChannels", "Canais existentes")),
          loading
            ? h(EmptyState, null, h(LoadingDots, { label: "Carregando canais..." }))
            : channels.length === 0
              ? h(EmptyState, null, tx(t, "admin.noChannels", "Nenhum canal ainda."))
              : h("ul", { className: "agora-admin-channel-list" },
                  channels.map(function (c) {
                    return h("li", { key: c.slug, className: "agora-admin-channel-item" },
                      h("div", { className: "agora-admin-channel-item__main" },
                        h("span", { className: "agora-admin-channel-item__name" }, c.name),
                        h("span", { className: "agora-admin-channel-item__slug" }, "#" + c.slug),
                      ),
                      c.description && h("p", { className: "agora-admin-channel-item__desc" }, c.description),
                    );
                  }),
                ),
        ),
      ),
    );
  }

  // -------------------------------------------------------------------------
  // Main page
  // -------------------------------------------------------------------------

  function AgoraPage() {
    const { t } = useI18n();
    const [channels, setChannels] = useState([]);
    const [selectedSlug, setSelectedSlug] = useState(null);
    const [messages, setMessages] = useState([]);
    const [agents, setAgents] = useState([]);
    const [workers, setWorkers] = useState([]);
    const [loadingChannels, setLoadingChannels] = useState(true);
    const [loadingMessages, setLoadingMessages] = useState(false);
    const [loadingOlderMessages, setLoadingOlderMessages] = useState(false);
    const [hasMoreMessages, setHasMoreMessages] = useState(true);
    const [loadingAgents, setLoadingAgents] = useState(true);
    const [loadingWorkers, setLoadingWorkers] = useState(true);
    const [error, setError] = useState(null);
    const [draft, setDraft] = useState("");
    const [sending, setSending] = useState(false);
    const [tick, setTick] = useState(0);
    const [clockTick, setClockTick] = useState(0);
    const [unreadCounts, setUnreadCounts] = useState({});
    const [channelUnreadCounts, setChannelUnreadCounts] = useState({});
    const [agentNotifications, setAgentNotifications] = useState({});
    const [openNotifications, setOpenNotifications] = useState({});
    const [loadingNotifications, setLoadingNotifications] = useState({});
    const [loadingOlderNotifications, setLoadingOlderNotifications] = useState({});
    const [hasMoreNotifications, setHasMoreNotifications] = useState({});
    const messagesListRef = useRef(null);
    const messagesEndRef = useRef(null);
    const composerInputRef = useRef(null);
    const initialLoadDoneRef = useRef({});
    const previousChannelRef = useRef(selectedSlug);
    const previousFirstIdRef = useRef(null);
    const previousLastIdRef = useRef(null);
    const previousLenRef = useRef(0);
    const [feedNewCount, setFeedNewCount] = useState(0);
    const [showNewMessagesBtn, setShowNewMessagesBtn] = useState(false);
    const [connection, setConnection] = useState(navigator.onLine ? "online" : "offline");
    const [pendingMessages, setPendingMessages] = useState([]);
    const [flushing, setFlushing] = useState(false);
    const [forceOffline, setForceOffline] = useState(false);

    // PID -> open agent tmux terminal
    const [openingTerminal, setOpeningTerminal] = useState(null);

    // Mention autocomplete state
    const [mentionOpen, setMentionOpen] = useState(false);
    const [mentionItems, setMentionItems] = useState([]);
    const [mentionIndex, setMentionIndex] = useState(0);
    const [mentionStart, setMentionStart] = useState(0);
    const [mentionEnd, setMentionEnd] = useState(0);

    // Responsive sidebar visibility (visible by default on desktop)
    const [showChannels, setShowChannels] = useState(typeof window !== "undefined" && window.innerWidth > 1024);
    const [showAgents, setShowAgents] = useState(typeof window !== "undefined" && window.innerWidth > 1024);
    const [showAdmin, setShowAdmin] = useState(false);

    // Live event-stream state
    const [eventCursor, setEventCursor] = useState(0);
    const [eventTransport, setEventTransport] = useState("ws"); // "ws" | "poll"
    const [eventStatus, setEventStatus] = useState("connecting");

    // Refs to latest state inside async callbacks
    const selectedSlugRef = useRef(selectedSlug);
    useEffect(function () { selectedSlugRef.current = selectedSlug; }, [selectedSlug]);
    const messagesRef = useRef(messages);
    useEffect(function () { messagesRef.current = messages; }, [messages]);
    const agentsRef = useRef(agents);
    useEffect(function () { agentsRef.current = agents; }, [agents]);
    const pageMountTimeRef = useRef(Math.floor(Date.now() / 1000));
    const eventCursorRef = useRef(eventCursor);
    useEffect(function () { eventCursorRef.current = eventCursor; }, [eventCursor]);
    const eventTransportRef = useRef(eventTransport);
    useEffect(function () { eventTransportRef.current = eventTransport; }, [eventTransport]);

    // Keep relative timestamps such as heartbeats moving even when no data
    // changes arrive over the event stream.
    useEffect(function () {
      const id = setInterval(function () {
        setClockTick(function (n) { return n + 1; });
      }, 30000);
      return function () { clearInterval(id); };
    }, []);
    void clockTick;

    // Derive connection label from network events, pending queue and flush
    useEffect(function () {
      if (flushing) { setConnection("reconnecting"); return; }
      if (!navigator.onLine || forceOffline) { setConnection("offline"); return; }
      if (pendingMessages.length > 0) { setConnection("reconnecting"); return; }
      setConnection("online");
    }, [flushing, forceOffline, pendingMessages.length]);

    // Merged view for the active channel
    const mergedMessages = useMemo(function () {
      const channelPending = pendingMessages.filter(function (p) { return p.slug === selectedSlug; });
      return messages.concat(channelPending);
    }, [messages, pendingMessages, selectedSlug]);

    // Load channels
    const loadChannels = useCallback(function () {
      setLoadingChannels(true);
      return SDK.fetchJSON(`${API_AGORA}/channels`)
        .then(function (data) {
          const list = (data && data.channels) || [];
          const valid = list.filter(isValidChannel);
          setChannels(valid);
          if (valid.length > 0 && !selectedSlugRef.current) {
            setSelectedSlug(valid[0].slug);
          }
          setError(null);
          return valid;
        })
        .catch(function (err) {
          setError(tx(t, "loadChannelsError", "Erro ao carregar canais: ") + parseApiError(err));
          return [];
        })
        .finally(function () { setLoadingChannels(false); });
    }, [t]);

    useEffect(function () {
      loadChannels();
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [tick]);

    // Load messages for selected channel
    const loadMessages = useCallback(function (slug, opts) {
      opts = opts || {};
      const isInitial = !opts.beforeId && !opts.sinceId;
      const isOlder = !!opts.beforeId;
      const isPolling = !!opts.sinceId;

      if (isInitial) setLoadingMessages(true);
      if (isOlder) setLoadingOlderMessages(true);

      const params = new URLSearchParams();
      params.set("limit", String(PAGE_SIZE));
      if (opts.beforeId) params.set("before_id", String(opts.beforeId));
      if (opts.sinceId) params.set("since_id", String(opts.sinceId));

      return SDK.fetchJSON(`${API_AGORA}/channels/${encodeURIComponent(slug)}/messages?${params}`)
        .then(function (data) {
          const list = (data && data.messages) || [];
          // Server returns newest-first for initial/older loads; reverse to ASC for display.
          if (!isPolling && list.length > 1) {
            list.reverse();
          }

          setMessages(function (prev) {
            let next;
            const byId = new Map(prev.map(function (m) { return [m.id, m]; }));
            if (isPolling) {
              for (const m of list) byId.set(m.id, m);
              next = Array.from(byId.values()).sort(function (a, b) {
                const ta = a.created_at || 0;
                const tb = b.created_at || 0;
                if (ta !== tb) return ta - tb;
                return (a.id || 0) - (b.id || 0);
              });
            } else if (isOlder) {
              for (const m of list) byId.set(m.id, m);
              next = Array.from(byId.values()).sort(function (a, b) {
                const ta = a.created_at || 0;
                const tb = b.created_at || 0;
                if (ta !== tb) return ta - tb;
                return (a.id || 0) - (b.id || 0);
              });
            } else {
              next = list;
            }
            return next;
          });

          setHasMoreMessages(function () {
            // Prefer server-side has_more flag when available; fall back to the
            // legacy heuristic for backward compatibility. Using a sentinel row
            // (limit+1 on the backend) lets us detect the last page even when it
            // is a full page, avoiding an empty follow-up request.
            if (data && typeof data.has_more === "boolean") {
              return data.has_more;
            }
            return list.length >= PAGE_SIZE;
          });

          return list;
        })
        .catch(function (err) {
          const msg = parseApiError(err);
          if (!isOlder && String(err.message || "").indexOf("404") === -1) {
            setError(tx(t, "loadMessagesError", "Erro ao carregar mensagens: ") + msg);
          }
          throw err;
        })
        .finally(function () {
          if (isInitial) setLoadingMessages(false);
          if (isOlder) setLoadingOlderMessages(false);
        });
    }, [t]);

    // Reset and load initial messages when channel changes
    useEffect(function () {
      if (!selectedSlug) {
        setMessages([]);
        setHasMoreMessages(true);
        return;
      }
      setMessages([]);
      setHasMoreMessages(true);
      initialLoadDoneRef.current[selectedSlug] = false;
      loadMessages(selectedSlug, {})
        .then(function () {
          initialLoadDoneRef.current[selectedSlug] = true;
          // Stick to bottom on first load.
          requestAnimationFrame(function () {
            if (messagesEndRef.current) {
              messagesEndRef.current.scrollIntoView({ behavior: "auto" });
            }
          });
        })
        .catch(function () {
          initialLoadDoneRef.current[selectedSlug] = true;
        });
    }, [selectedSlug, loadMessages]);

    // Poll for new messages
    useEffect(function () {
      if (!selectedSlug || !initialLoadDoneRef.current[selectedSlug]) return;
      const lastId = messages.length > 0 ? messages[messages.length - 1].id : null;
      if (!lastId) return;
      loadMessages(selectedSlug, { sinceId: lastId })
        .catch(function () { /* errors already surfaced by loadMessages */ });
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedSlug, tick]);

    // Scroll handler for loading older messages
    const handleMessagesScroll = useCallback(function () {
      const el = messagesListRef.current;
      if (!el || !selectedSlug) return;

      // Clear new-messages badge when the user scrolls back to the bottom.
      const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight <= FEED_NEAR_BOTTOM_PX;
      if (isNearBottom) {
        setFeedNewCount(0);
        setShowNewMessagesBtn(false);
      }

      if (loadingOlderMessages || !hasMoreMessages) return;
      if (el.scrollTop <= 100) {
        const firstId = messages.length > 0 ? messages[0].id : null;
        if (!firstId) return;
        const prevHeight = el.scrollHeight;
        const prevScrollTop = el.scrollTop;
        loadMessages(selectedSlug, { beforeId: firstId })
          .then(function (older) {
            if (!older || older.length === 0) return;
            requestAnimationFrame(function () {
              const newHeight = el.scrollHeight;
              el.scrollTop = prevScrollTop + (newHeight - prevHeight);
            });
          })
          .catch(function () { /* ignore */ });
      }
    }, [messages, loadingOlderMessages, hasMoreMessages, selectedSlug, loadMessages]);

    const throttledMessagesScroll = useMemo(function () {
      return throttle(handleMessagesScroll, 200);
    }, [handleMessagesScroll]);

    const scrollFeedToBottom = useCallback(function () {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
      }
      setFeedNewCount(0);
      setShowNewMessagesBtn(false);
    }, []);

    // Smart scroll: avoid pulling the user away from history.
    useEffect(function () {
      const el = messagesListRef.current;
      if (!el) return;

      const isChannelSwitch = previousChannelRef.current !== selectedSlug;
      previousChannelRef.current = selectedSlug;

      const firstId = messages.length > 0 ? messages[0].id : null;
      const lastId = messages.length > 0 ? messages[messages.length - 1].id : null;
      const len = messages.length;
      const prevLen = previousLenRef.current;
      const prevLastId = previousLastIdRef.current;
      const added = Math.max(0, len - prevLen);
      const lastChanged = prevLastId !== null && lastId !== prevLastId;
      const isNewAtBottom = added > 0 && (lastChanged || prevLen === 0);

      previousFirstIdRef.current = firstId;
      previousLastIdRef.current = lastId;
      previousLenRef.current = len;

      if (isChannelSwitch || len === 0) {
        setFeedNewCount(0);
        setShowNewMessagesBtn(false);
        return;
      }

      const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight <= FEED_NEAR_BOTTOM_PX;

      if (isNewAtBottom && !isNearBottom) {
        setFeedNewCount(function (n) { return n + added; });
        setShowNewMessagesBtn(true);
      } else if (isNearBottom && messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
        setFeedNewCount(0);
        setShowNewMessagesBtn(false);
      }
    }, [messages, selectedSlug]);

    // Load agent statuses
    const loadAgents = useCallback(function () {
      setLoadingAgents(true);
      SDK.fetchJSON(`${API_AGORA}/agents/status`)
        .then(function (data) {
          const list = (data && data.agents) || [];
          setAgents(list);
          setError(null);
        })
        .catch(function (err) {
          if (String(err.message || "").indexOf("404") === -1) {
            setError(tx(t, "loadAgentsError", "Erro ao carregar agentes: ") + parseApiError(err));
          }
          setAgents([]);
        })
        .finally(function () { setLoadingAgents(false); });
    }, [t]);

    useEffect(loadAgents, [tick, loadAgents]);

    // Load Kanban active workers (best-effort enrichment)
    const loadWorkers = useCallback(function () {
      if (document.hidden) return;
      setLoadingWorkers(true);
      SDK.fetchJSON(`${API_KANBAN}/workers/active`)
        .then(function (data) {
          const list = (data && data.workers) || [];
          setWorkers(list);
        })
        .catch(function () {
          setWorkers([]);
        })
        .finally(function () { setLoadingWorkers(false); });
    }, []);

    useEffect(function () {
      loadWorkers();
      const id = setInterval(loadWorkers, WORKER_POLL_MS);
      return function () { clearInterval(id); };
    }, [loadWorkers]);

    const enrichedAgents = useMemo(function () {
      const workerMap = {};
      for (const w of workers) {
        if (w.profile) workerMap[w.profile] = w;
      }
      const out = agents.map(function (a) {
        const w = workerMap[a.profile];
        const enriched = Object.assign({}, a, { worker: w || undefined });
        if (w && w.worker_pid) {
          // Active Kanban worker wins over a stale semantic state from
          // /agents/status (e.g. Idle/Reviewing while the worker is running).
          // Override all worker-derivable fields so stale current_step/pid do
          // not leak into the sidebar card.
          enriched.state = "working";
          enriched.current_task_id = w.task_id || null;
          enriched.current_step = `run ${w.run_id}`;
          enriched.status_text = w.task_title || "trabalhando";
          enriched.run_id = w.run_id ?? null;
          enriched.pid = w.worker_pid ?? null;
          enriched.last_heartbeat_at = w.last_heartbeat_at || enriched.last_heartbeat_at;
        }
        return enriched;
      });
      for (const w of workers) {
        if (w.profile && w.worker_pid && !out.find(function (a) { return a.profile === w.profile; })) {
          out.push({
            profile: w.profile,
            state: "working",
            current_task_id: w.task_id || null,
            current_step: `run ${w.run_id}`,
            status_text: w.task_title || null,
            last_heartbeat_at: w.last_heartbeat_at || null,
            pid: w.worker_pid || null,
            run_id: w.run_id || null,
            metadata_json: null,
            worker: w,
          });
        }
      }
      return out;
    }, [agents, workers]);

    // Show observable agents (worker/pid) plus manifest-declared profiles so
    // humans can summon them from the UI even before first heartbeat.
    const visibleAgents = useMemo(function () {
      return enrichedAgents.filter(function (a) {
        const pid = a.worker ? a.worker.worker_pid : a.pid;
        if (pid) return true;
        if (a.worker) return true;
        return !!(a.metadata && a.metadata.source === "manifest");
      });
    }, [enrichedAgents]);

    // -----------------------------------------------------------------------
    // Mention autocomplete logic
    // -----------------------------------------------------------------------
    const buildMentionItems = useCallback(function (query) {
      const q = (query || "").toLowerCase();
      const specials = [
        { handle: "all", label: "todos", state: null, broadcast: true },
        { handle: "todos", label: "todos", state: null, broadcast: true },
      ];
      const seen = new Set();
      const agentItems = [];
      for (const a of enrichedAgents) {
        const profile = a.profile;
        if (!profile || seen.has(profile)) continue;
        seen.add(profile);
        agentItems.push({ handle: profile, label: profile, state: a.state || null, broadcast: false });
      }
      agentItems.sort(function (a, b) { return a.handle.localeCompare(b.handle); });
      return specials.concat(agentItems).filter(function (i) {
        return i.handle.toLowerCase().includes(q);
      });
    }, [enrichedAgents]);

    const updateMentionState = useCallback(function (ctx, resetIndex) {
      if (!ctx) {
        if (mentionOpen) setMentionOpen(false);
        return;
      }
      const items = buildMentionItems(ctx.query);
      if (items.length === 0) {
        if (mentionOpen) setMentionOpen(false);
        return;
      }
      const nextIndex = resetIndex ? 0 : Math.min(mentionIndex, items.length - 1);
      setMentionOpen(true);
      setMentionItems(items);
      setMentionIndex(nextIndex);
      setMentionStart(ctx.start);
      setMentionEnd(ctx.end);
    }, [mentionOpen, mentionIndex, buildMentionItems]);

    const insertMention = useCallback(function () {
      if (!mentionOpen || mentionItems.length === 0) return;
      const item = mentionItems[mentionIndex];
      const before = draft.slice(0, mentionStart);
      const after = draft.slice(mentionEnd);
      const replacement = "@" + item.handle + " ";
      const nextDraft = before + replacement + after;
      const caretPos = before.length + replacement.length;
      setDraft(nextDraft);
      setMentionOpen(false);
      requestAnimationFrame(function () {
        const input = composerInputRef.current || document.querySelector(".agora-composer-input");
        if (input) {
          try { input.focus(); } catch (_e) {}
          try { input.setSelectionRange(caretPos, caretPos); } catch (_e) {}
        }
      });
    }, [draft, mentionStart, mentionEnd, mentionOpen, mentionItems, mentionIndex]);

    const handleComposerChange = useCallback(function (e) {
      const value = e.target.value;
      const caret = e.target.selectionStart || 0;
      setDraft(value);
      const ctx = getMentionContextAt(value, caret);
      updateMentionState(ctx, true);
    }, [updateMentionState]);

    const handleComposerKeyUp = useCallback(function (e) {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight" || e.key === "Home" || e.key === "End") {
        const input = composerInputRef.current || document.querySelector(".agora-composer-input");
        const ctx = getMentionContextAt(draft, input && input.selectionStart || 0);
        updateMentionState(ctx, false);
      }
    }, [draft, updateMentionState]);

    const handleComposerClick = useCallback(function () {
      const input = composerInputRef.current || document.querySelector(".agora-composer-input");
      const ctx = getMentionContextAt(draft, input && input.selectionStart || 0);
      updateMentionState(ctx, false);
    }, [draft, updateMentionState]);

    useEffect(function () {
      if (!mentionOpen) return;
      function onDocDown(ev) {
        const popup = document.getElementById("agora-mention-listbox");
        const input = composerInputRef.current || document.querySelector(".agora-composer-input");
        if (!popup) return;
        if (!popup.contains(ev.target) && ev.target !== input) {
          setMentionOpen(false);
        }
      }
      document.addEventListener("mousedown", onDocDown);
      return function () { document.removeEventListener("mousedown", onDocDown); };
    }, [mentionOpen]);

    const selectedChannel = useMemo(function () {
      return channels.find(function (c) { return c.slug === selectedSlug; }) || null;
    }, [channels, selectedSlug]);

    // Fetch unread counts once when agents become known (updated live via events)
    const initialCountsLoadedRef = useRef(false);
    useEffect(function () {
      if (initialCountsLoadedRef.current) return;
      if (enrichedAgents.length === 0) return;
      if (visibleAgents.length === 0) {
        // mark as loaded so we don't keep retrying while no agent is visible
        initialCountsLoadedRef.current = true;
        return;
      }
      initialCountsLoadedRef.current = true;
      const nextCounts = {};
      let pending = visibleAgents.length;
      let settled = 0;
      visibleAgents.forEach(function (a) {
        SDK.fetchJSON(`${API_AGORA}/notifications/count?recipient=${encodeURIComponent(a.profile)}`)
          .then(function (data) {
            nextCounts[a.profile] = (data && data.unread) || 0;
          })
          .catch(function () {
            nextCounts[a.profile] = 0;
          })
          .finally(function () {
            settled += 1;
            if (settled === pending) {
              setUnreadCounts(nextCounts);
            }
          });
      });
    }, [enrichedAgents.map(function (a) { return a.profile; }).join(",")]);

    const loadNotifications = useCallback(function (profile, opts) {
      opts = opts || {};
      const isOlder = !!opts.beforeId;
      const isPolling = !!opts.sinceId;

      setLoadingNotifications(function (prev) { return Object.assign({}, prev, { [profile]: !isOlder }); });
      if (isOlder) {
        setLoadingOlderNotifications(function (prev) { return Object.assign({}, prev, { [profile]: true }); });
      }

      const params = new URLSearchParams({ recipient: profile, limit: String(PAGE_SIZE) });
      if (opts.beforeId) params.set("before_id", String(opts.beforeId));
      if (opts.sinceId) params.set("since_id", String(opts.sinceId));

      return SDK.fetchJSON(`${API_AGORA}/notifications?${params}`)
        .then(function (data) {
          const list = (data && data.notifications) || [];
          setAgentNotifications(function (prev) {
            const prevList = prev[profile] || [];
            const byId = new Map(prevList.map(function (n) { return [n.id, n]; }));
            for (const n of list) byId.set(n.id, n);
            let nextList = Array.from(byId.values()).sort(function (a, b) {
              const ta = b.created_at || 0;
              const tb = a.created_at || 0;
              if (ta !== tb) return ta - tb;
              return (b.id || 0) - (a.id || 0);
            });
            return Object.assign({}, prev, { [profile]: nextList });
          });
          setHasMoreNotifications(function (prev) {
            const hasMore = data && typeof data.has_more === "boolean"
              ? data.has_more
              : list.length >= PAGE_SIZE;
            return Object.assign({}, prev, { [profile]: hasMore });
          });
          // Reconcile the bell badge with the server count, since live events
          // from the auto-ack tmux delivery (and missed/paginated batches) can
          // leave the local counter stale.
          SDK.fetchJSON(`${API_AGORA}/notifications/count?recipient=${encodeURIComponent(profile)}`)
            .then(function (data) {
              setUnreadCounts(function (prev) {
                return Object.assign({}, prev, { [profile]: (data && data.unread) || 0 });
              });
            })
            .catch(function () {});
          return list;
        })
        .catch(function (err) {
          setError(tx(t, "loadNotificationsError", "Erro ao carregar notificações: ") + parseApiError(err));
          throw err;
        })
        .finally(function () {
          setLoadingNotifications(function (prev) { return Object.assign({}, prev, { [profile]: false }); });
          if (isOlder) {
            setLoadingOlderNotifications(function (prev) { return Object.assign({}, prev, { [profile]: false }); });
          }
        });
    }, [t]);

    // -----------------------------------------------------------------------
    // Apply events from the live stream / long-poll
    // -----------------------------------------------------------------------
    const applyEvents = useCallback(function (events) {
      if (!events || !events.length) return;
      events.forEach(function (e) {
        const entityType = e.entity_type;
        const eventType = e.event_type;
        const payload = e.payload || {};
        const createdAt = e.created_at || Math.floor(Date.now() / 1000);

        if (entityType === "message" && eventType === "created") {
          if (payload.channel_slug && payload.channel_slug === selectedSlugRef.current) {
            setTick(function (n) { return n + 1; });
          } else if (payload.channel_slug && createdAt >= pageMountTimeRef.current) {
            setChannelUnreadCounts(function (prev) {
              return Object.assign({}, prev, { [payload.channel_slug]: (prev[payload.channel_slug] || 0) + 1 });
            });
          }
          return;
        }

        if (entityType === "agent_status" && eventType === "updated") {
          const profile = e.entity_id;
          setAgents(function (prev) {
            const idx = prev.findIndex(function (a) { return a.profile === profile; });
            if (idx >= 0) {
              const next = prev.slice();
              const a = Object.assign({}, next[idx]);
              if (payload.state) a.state = payload.state;
              if (payload.current_task_id !== undefined) a.current_task_id = payload.current_task_id;
              a.last_heartbeat_at = createdAt;
              next[idx] = a;
              return next;
            }
            return prev.concat([{
              profile: profile,
              state: payload.state || "idle",
              current_task_id: payload.current_task_id || null,
              current_step: null,
              status_text: null,
              last_heartbeat_at: createdAt,
              pid: null,
              run_id: null,
              metadata: null,
            }]);
          });
          return;
        }

        if (entityType === "notification") {
          const recipient = payload.recipient || e.entity_id;

          if (eventType === "created") {
            const nid = Number(e.entity_id) || 0;
            let isNew = true;
            setAgentNotifications(function (prev) {
              const list = prev[recipient] ? prev[recipient].slice() : [];
              if (list.some(function (n) { return n.id === nid; })) {
                isNew = false;
                return prev;
              }
              list.unshift({
                id: nid,
                recipient: recipient,
                message_id: payload.message_id,
                channel_id: payload.channel_id,
                body_snippet: payload.body_snippet || "",
                author_profile: payload.author_profile || "sistema",
                read_at: null,
                ack_at: null,
                created_at: createdAt,
              });
              return Object.assign({}, prev, { [recipient]: list });
            });
            if (isNew) {
              setUnreadCounts(function (prev) {
                return Object.assign({}, prev, { [recipient]: (prev[recipient] || 0) + 1 });
              });
            }
            return;
          }

          if (eventType === "read") {
            const rid = Number(e.entity_id) || 0;
            let wasRead = false;
            setAgentNotifications(function (prev) {
              const list = prev[recipient];
              if (!list) return prev;
              const existing = list.find(function (n) { return n.id === rid; });
              if (existing && existing.read_at) {
                wasRead = true;
                return prev;
              }
              const next = list.map(function (n) {
                return n.id === rid ? Object.assign({}, n, { read_at: createdAt }) : n;
              });
              return Object.assign({}, prev, { [recipient]: next });
            });
            if (!wasRead) {
              setUnreadCounts(function (prev) {
                return Object.assign({}, prev, { [recipient]: Math.max(0, (prev[recipient] || 0) - 1) });
              });
            }
            return;
          }

          if (eventType === "read_all") {
            setAgentNotifications(function (prev) {
              const list = prev[recipient];
              if (!list) return prev;
              const next = list.map(function (n) { return Object.assign({}, n, { read_at: n.read_at || createdAt }); });
              return Object.assign({}, prev, { [recipient]: next });
            });
            setUnreadCounts(function (prev) {
              return Object.assign({}, prev, { [recipient]: 0 });
            });
          }
          return;
        }

        if (entityType === "channel" && eventType === "created") {
          const ch = {
            id: Number(e.entity_id) || 0,
            slug: payload.slug,
            name: payload.name,
            description: payload.description,
            created_at: createdAt,
          };
          if (!isValidChannel(ch)) return;
          setChannels(function (prev) {
            if (prev.find(function (c) { return c.slug === ch.slug; })) return prev;
            return prev.concat([ch]);
          });
        }
      });
    }, []);


    // -----------------------------------------------------------------------
    // WebSocket event stream (+ long-poll fallback)
    // -----------------------------------------------------------------------
    useEffect(function () {
      if (eventTransport !== "ws") return;
      let cancelled = false;
      let ws = null;
      let retryTimer = null;
      let retries = 0;
      let startedAt = Date.now();
      const mountTime = Date.now();
      let gaveUp = false;

      function cleanup() {
        cancelled = true;
        if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
        if (ws) { try { ws.close(); } catch (_e) {} ws = null; }
      }

      function connect() {
        if (cancelled) return;
        if (typeof SDK.buildWsUrl !== "function") {
          setEventStatus("poll");
          setEventTransport("poll");
          return;
        }
        setEventStatus("connecting");
        const cursor = eventCursorRef.current || 0;
        SDK.buildWsUrl(`${API_AGORA}/events`, { since: String(cursor) })
          .then(function (url) {
            if (cancelled) return;
            try {
              ws = new WebSocket(url);
            } catch (err) {
              setEventStatus("poll");
              setEventTransport("poll");
              return;
            }
            ws.onopen = function () {
              if (cancelled) return;
              setEventStatus("open");
              retries = 0;
              startedAt = Date.now();
            };
            ws.onmessage = function (ev) {
              if (cancelled) return;
              try {
                const data = JSON.parse(ev.data);
                const batch = (data && data.events) || [];
                const cursor = (data && data.cursor) || 0;
                if (batch.length) applyEvents(batch);
                setEventCursor(cursor);
              } catch (_e) { /* malformed frame */ }
            };
            ws.onerror = function () {
              if (cancelled) return;
              setEventStatus("error");
              try { ws.close(); } catch (_e) {}
            };
            ws.onclose = function () {
              if (cancelled) return;
              ws = null;
              setEventStatus("closed");
              const elapsed = Date.now() - startedAt;
              if (elapsed > 3000) {
                retryTimer = setTimeout(connect, 1000);
              } else if (!gaveUp && Date.now() - mountTime > WS_GIVE_UP_AFTER_MS) {
                gaveUp = true;
                setEventStatus("poll");
                setEventTransport("poll");
              } else {
                retries += 1;
                const delay = Math.min(1000 * Math.pow(1.5, retries), WS_RETRY_MAX_MS);
                retryTimer = setTimeout(connect, delay);
              }
            };
          })
          .catch(function (err) {
            if (cancelled) return;
            setEventStatus("poll");
            setEventTransport("poll");
            setError(tx(t, "loadAgentsError", "Eventos em tempo real indisponíveis: ") + parseApiError(err));
          });
      }

      connect();
      return cleanup;
    }, [eventTransport, applyEvents, t]);

    useEffect(function () {
      if (eventTransport !== "poll") return;
      let cancelled = false;
      let timer = null;

      function poll() {
        if (cancelled) return;
        const cursor = eventCursorRef.current || 0;
        SDK.fetchJSON(`${API_AGORA}/events?since_id=${cursor}`)
          .then(function (data) {
            if (cancelled) return;
            const batch = (data && data.events) || [];
            const nextCursor = (data && data.cursor) || cursor;
            if (batch.length) {
              applyEvents(batch);
              setEventCursor(nextCursor);
              timer = setTimeout(poll, 500);
            } else {
              timer = setTimeout(poll, EVENT_POLL_INTERVAL_MS);
            }
          })
          .catch(function () {
            if (cancelled) return;
            timer = setTimeout(poll, EVENT_POLL_INTERVAL_MS);
          });
      }

      poll();
      return function () {
        cancelled = true;
        if (timer) clearTimeout(timer);
      };
    }, [eventTransport, applyEvents]);

    // -----------------------------------------------------------------------
    // Resync when the tab returns from background
    // -----------------------------------------------------------------------
    useEffect(function () {
      let hiddenAt = 0;
      function onVisible() {
        if (document.hidden) {
          hiddenAt = Date.now();
          return;
        }
        const hiddenFor = hiddenAt ? Date.now() - hiddenAt : 0;
        if (hiddenFor < RESYNC_COOLDOWN_MS) return;
        hiddenAt = 0;

        setTick(function (n) { return n + 1; });
        loadAgents();
        loadWorkers();

        // Refresh unread counts without reseting the initial guard.
        const list = agentsRef.current || [];
        if (list.length) {
          const next = {};
          let pending = list.length;
          let settled = 0;
          list.forEach(function (a) {
            SDK.fetchJSON(`${API_AGORA}/notifications/count?recipient=${encodeURIComponent(a.profile)}`)
              .then(function (data) {
                next[a.profile] = (data && data.unread) || 0;
              })
              .catch(function () {
                next[a.profile] = 0;
              })
              .finally(function () {
                settled += 1;
                if (settled === pending) {
                  setUnreadCounts(next);
                }
              });
          });
        }

        SDK.fetchJSON(`${API_AGORA}/events?since_id=${eventCursorRef.current || 0}`)
          .then(function (data) {
            const batch = (data && data.events) || [];
            if (batch.length) {
              applyEvents(batch);
              setEventCursor((data && data.cursor) || eventCursorRef.current);
            }
          })
          .catch(function () { /* ignore */ });

        if (eventTransportRef.current === "poll") {
          setEventTransport("ws");
        }
      }
      document.addEventListener("visibilitychange", onVisible);
      window.addEventListener("focus", onVisible);
      return function () {
        document.removeEventListener("visibilitychange", onVisible);
        window.removeEventListener("focus", onVisible);
      };
    }, [loadAgents, loadWorkers, applyEvents]);

    // Load notifications list when an agent panel is opened
    useEffect(function () {
      const openProfiles = Object.keys(openNotifications).filter(function (p) {
        return openNotifications[p];
      });
      if (openProfiles.length === 0) return;
      openProfiles.forEach(function (profile) {
        loadNotifications(profile, {});
      });
    }, [Object.keys(openNotifications).filter(function (p) { return openNotifications[p]; }).join(",")]);

    // Poll open notification panels for new items
    useEffect(function () {
      const openProfiles = Object.keys(openNotifications).filter(function (p) {
        return openNotifications[p];
      });
      openProfiles.forEach(function (profile) {
        const list = agentNotifications[profile] || [];
        const sinceId = list.length > 0 ? list[0].id : null;
        if (sinceId) {
          loadNotifications(profile, { sinceId: sinceId }).catch(function () {});
        }
      });
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [tick]);

    const toggleNotifications = useCallback(function (profile) {
      setOpenNotifications(function (prev) {
        const next = Object.assign({}, prev, { [profile]: !prev[profile] });
        return next;
      });
    }, []);

    const handleNotificationScroll = useCallback(function (profile) {
      return function (e) {
        const el = e.currentTarget;
        const list = agentNotifications[profile] || [];
        if (!list.length) return;
        if (loadingOlderNotifications[profile] || hasMoreNotifications[profile] === false) return;
        const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
        if (nearBottom) {
          const lastId = list[list.length - 1].id;
          loadNotifications(profile, { beforeId: lastId }).catch(function () {});
        }
      };
    }, [agentNotifications, loadingOlderNotifications, hasMoreNotifications, loadNotifications]);

    const throttledNotificationScroll = useMemo(function () {
      const handlers = {};
      for (const profile of Object.keys(openNotifications)) {
        handlers[profile] = throttle(handleNotificationScroll(profile), 200);
      }
      return handlers;
    }, [handleNotificationScroll, openNotifications]);

    const markNotificationRead = useCallback(function (profile, notificationId) {
      setLoadingNotifications(function (prev) { return Object.assign({}, prev, { [profile]: true }); });
      SDK.fetchJSON(`${API_AGORA}/notifications/${encodeURIComponent(notificationId)}/read`, {
        method: "POST",
      })
        .then(function () {
          setAgentNotifications(function (prev) {
            const list = prev[profile] || [];
            const now = Math.floor(Date.now() / 1000);
            return Object.assign({}, prev, {
              [profile]: list.map(function (n) {
                return n.id === notificationId ? Object.assign({}, n, { read_at: now }) : n;
              }),
            });
          });
          setUnreadCounts(function (prev) {
            const current = prev[profile] || 0;
            return Object.assign({}, prev, { [profile]: Math.max(0, current - 1) });
          });
        })
        .catch(function (err) {
          setError(tx(t, "markReadError", "Erro ao marcar como lida: ") + parseApiError(err));
        })
        .finally(function () {
          setLoadingNotifications(function (prev) { return Object.assign({}, prev, { [profile]: false }); });
          setTick(function (n) { return n + 1; });
        });
    }, [t]);

    const markAllNotificationsRead = useCallback(function (profile) {
      setLoadingNotifications(function (prev) { return Object.assign({}, prev, { [profile]: true }); });
      SDK.fetchJSON(`${API_AGORA}/notifications/read-all?recipient=${encodeURIComponent(profile)}`, {
        method: "POST",
      })
        .then(function () {
          setAgentNotifications(function (prev) {
            const list = prev[profile] || [];
            const now = Math.floor(Date.now() / 1000);
            return Object.assign({}, prev, {
              [profile]: list.map(function (n) { return Object.assign({}, n, { read_at: n.read_at || now }); }),
            });
          });
          setUnreadCounts(function (prev) { return Object.assign({}, prev, { [profile]: 0 }); });
        })
        .catch(function (err) {
          setError(tx(t, "markReadError", "Erro ao marcar todas como lidas: ") + parseApiError(err));
        })
        .finally(function () {
          setLoadingNotifications(function (prev) { return Object.assign({}, prev, { [profile]: false }); });
          setTick(function (n) { return n + 1; });
        });
    }, [t]);

    const handleOpenTerminal = useCallback(function (profile) {
      if (openingTerminal) return;
      setOpeningTerminal(profile);
      SDK.fetchJSON(`${API_AGORA}/agents/${encodeURIComponent(profile)}/open-terminal?target=profile-session`, {
        method: "POST",
      })
        .then(function (data) {
          if (!data || !data.ok) {
            throw new Error((data && data.reason) || "unknown");
          }
        })
        .catch(function (err) {
          setError(tx(t, "openTerminalError", "Erro ao abrir terminal de ") + profile + ": " + parseApiError(err));
        })
        .finally(function () {
          setOpeningTerminal(null);
        });
    }, [openingTerminal, t]);

    const handleSummonAgent = useCallback(function (profile) {
      if (openingTerminal) return;
      setOpeningTerminal(profile);
      SDK.fetchJSON(`${API_AGORA}/agents/${encodeURIComponent(profile)}/summon`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ open_terminal: true, state: "working" }),
      })
        .then(function (data) {
          if (!data || !data.ok) {
            throw new Error((data && data.reason) || "unknown");
          }
          setTick(function (n) { return n + 1; });
        })
        .catch(function (err) {
          setError(tx(t, "summonError", "Erro ao invocar agente ") + profile + ": " + parseApiError(err));
        })
        .finally(function () {
          setOpeningTerminal(null);
        });
    }, [openingTerminal, t]);

    // Reconcile unread bells with the server periodically. Auto-acked tmux
    // deliveries can leave counters off-by-one when the live event stream is
    // paginated or when read events arrive before the notification list is
    // loaded locally.
    const syncAllUnreadCounts = useCallback(function () {
      const list = agentsRef.current || [];
      if (!list.length) return;
      const next = {};
      let pending = list.length;
      let settled = 0;
      list.forEach(function (a) {
        SDK.fetchJSON(`${API_AGORA}/notifications/count?recipient=${encodeURIComponent(a.profile)}`)
          .then(function (data) {
            next[a.profile] = (data && data.unread) || 0;
          })
          .catch(function () {
            next[a.profile] = 0;
          })
          .finally(function () {
            settled += 1;
            if (settled === pending) {
              setUnreadCounts(next);
            }
          });
      });
    }, []);

    useEffect(function () {
      syncAllUnreadCounts();
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [clockTick]);

    function pushPending(slug, body) {
      const item = {
        id: generateId(),
        slug: slug,
        body: body,
        created_at: Math.floor(Date.now() / 1000),
        pending: true,
        author_type: "human",
        author_profile: null,
      };
      setPendingMessages(function (prev) { return prev.concat(item); });
    }

    function removePending(id) {
      setPendingMessages(function (prev) { return prev.filter(function (i) { return i.id !== id; }); });
    }

    const flushQueue = useCallback(function () {
      if (flushing || pendingMessages.length === 0) return;
      if (!navigator.onLine) { setConnection("offline"); return; }
      setFlushing(true);
      const queue = pendingMessages.slice();
      const attemptedIds = new Set(queue.map(function (i) { return i.id; }));
      const failed = [];
      function next(idx) {
        if (idx >= queue.length) {
          setPendingMessages(function (prev) {
            const failedIds = new Set(failed.map(function (i) { return i.id; }));
            return prev.filter(function (i) { return !attemptedIds.has(i.id) || failedIds.has(i.id); });
          });
          setFlushing(false);
          if (queue.length > failed.length) {
            setTick(function (n) { return n + 1; });
          }
          return;
        }
        postMessage(queue[idx].slug, queue[idx].body)
          .then(function () { next(idx + 1); })
          .catch(function () { failed.push(queue[idx]); next(idx + 1); });
      }
      next(0);
    }, [flushing, pendingMessages]);

    const retryPending = useCallback(function (item) {
      if (flushing) return;
      setFlushing(true);
      postMessage(item.slug, item.body)
        .then(function () {
          removePending(item.id);
          setTick(function (n) { return n + 1; });
        })
        .catch(function () { /* mantém na fila */ })
        .finally(function () { setFlushing(false); });
    }, [flushing]);

    // Connection status listeners
    useEffect(function () {
      function onOffline() { setForceOffline(true); }
      function onOnline() { setForceOffline(false); flushQueue(); }
      window.addEventListener("offline", onOffline);
      window.addEventListener("online", onOnline);
      return function () {
        window.removeEventListener("offline", onOffline);
        window.removeEventListener("online", onOnline);
      };
    }, [flushQueue]);

    // Retry pending messages periodically while online
    useEffect(function () {
      if (pendingMessages.length === 0) return;
      if (!navigator.onLine || forceOffline) return;
      const id = setInterval(flushQueue, POLL_MS);
      return function () { clearInterval(id); };
    }, [pendingMessages.length, flushQueue, forceOffline]);

    const sendMessage = useCallback(function () {
      const body = draft.trim();
      if (!body || !selectedSlug) return;
      if (connection === "offline" || !navigator.onLine) {
        pushPending(selectedSlug, body);
        setDraft("");
        return;
      }
      setSending(true);
      postMessage(selectedSlug, body)
        .then(function () {
          setDraft("");
          setTick(function (n) { return n + 1; });
          setError(null);
        })
        .catch(function (err) {
          if (isRetryableError(err)) {
            pushPending(selectedSlug, body);
            setDraft("");
            setError(null);
          } else {
            setError(tx(t, "sendError", "Erro ao enviar mensagem: ") + parseApiError(err));
          }
        })
        .finally(function () { setSending(false); });
    }, [draft, selectedSlug, t]);

    const handleComposerKeyDown = useCallback(function (e) {
      if (mentionOpen) {
        if (e.key === "ArrowDown") {
          e.preventDefault();
          setMentionIndex(function (idx) { return (idx + 1) % mentionItems.length; });
          return;
        }
        if (e.key === "ArrowUp") {
          e.preventDefault();
          setMentionIndex(function (idx) { return (idx - 1 + mentionItems.length) % mentionItems.length; });
          return;
        }
        if (e.key === "Enter") {
          e.preventDefault();
          insertMention();
          return;
        }
        if (e.key === "Escape") {
          e.preventDefault();
          setMentionOpen(false);
          return;
        }
      }
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    }, [mentionOpen, mentionItems.length, sendMessage, insertMention]);

    const toggleChannels = useCallback(function () { setShowChannels(function (s) { return !s; }); }, []);
    const toggleAgents = useCallback(function () { setShowAgents(function (s) { return !s; }); }, []);
    const toggleAdmin = useCallback(function () { setShowAdmin(function (s) { return !s; }); }, []);

    return h("div", { className: cn("agora", showAdmin && "agora--admin-mode"), role: "group", "aria-label": tx(t, "appLabel", "Ágora — praça pública de agentes") },
      h("header", { className: "agora-header" },
        h("div", { className: "agora-header-title" },
          h("h1", null, tx(t, "title", "Ágora")),
          h("p", null, tx(t, "subtitle", "Praça pública para agentes deliberarem com telemetria humana")),
        ),
        h("div", { className: "agora-header__toggles" },
          h(Button, { size: "sm", variant: "outline", "aria-pressed": showChannels, "aria-expanded": showChannels, "aria-label": tx(t, "toggleChannels", "Canais"), onClick: toggleChannels }, tx(t, "channelsTitle", "Canais")),
          h(Button, { size: "sm", variant: "outline", "aria-pressed": showAgents, "aria-expanded": showAgents, "aria-label": tx(t, "toggleAgents", "Agentes"), onClick: toggleAgents }, tx(t, "agentsTitle", "Agentes")),
        ),
        h("div", { className: "agora-header__admin" },
          h("button", {
            type: "button",
            className: cn("agora-header-btn", showAdmin && "agora-header-btn--active"),
            "aria-pressed": showAdmin,
            "aria-expanded": showAdmin,
            "aria-label": tx(t, "toggleAdmin", "Admin"),
            onClick: toggleAdmin,
          }, tx(t, "admin.shortTitle", "Admin")),
        ),
        h("div", { className: "agora-header-meta" },
          h("span", null, tx(t, "channelsCount", "{count} canais", { count: channels.length })),
          h("span", null, tx(t, "agentsCount", "{count} agentes", { count: visibleAgents.length })),
          h(ConnectionBadge, { connection: connection, pendingCount: pendingMessages.length }),
        ),
      ),

      error && h("div", { className: "agora-banner agora-banner--error" },
        h("span", null, error),
        h(Button, { size: "sm", variant: "outline", onClick: function () { setError(null); } }, "×"),
      ),

      showAdmin
        ? h("main", { className: "agora-screen agora-screen--admin", role: "region", "aria-label": tx(t, "admin.title", "Admin — Canais") },
            h(AdminScreen, {
              channels: channels,
              loading: loadingChannels,
              onClose: toggleAdmin,
              onChannelCreated: function (slug) {
                setShowAdmin(false);
                setTick(function (n) { return n + 1; });
                if (slug) {
                  setSelectedSlug(slug);
                  setChannelUnreadCounts(function (prev) { return Object.assign({}, prev, { [slug]: 0 }); });
                }
              },
            })
          )
        : h("div", { className: "agora-layout" },
            // Left: channels
            h("aside", { className: cn("agora-sidebar", "agora-sidebar--left", showChannels ? "agora-sidebar--visible" : "agora-sidebar--hidden") },
              h(Card, { className: "agora-panel" },
                h(CardHeader, { className: "agora-panel-header" },
                  h(CardTitle, { className: "agora-panel-title" }, tx(t, "channelsTitle", "Canais")),
                ),
                h(CardContent, { className: "agora-panel-body" },
                  loadingChannels
                    ? h(EmptyState, null, h(LoadingDots, { label: "Carregando canais..." }))
                    : channels.length === 0
                      ? h(EmptyState, null, tx(t, "noChannels", "Nenhum canal ainda."))
                      : h("div", { className: "agora-channel-list", role: "tablist", "aria-label": "Canais" },
                          channels.map(function (c) {
                            return h(ChannelItem, {
                              key: c.slug,
                              channel: c,
                              selected: c.slug === selectedSlug,
                              unread: channelUnreadCounts[c.slug] || 0,
                              onClick: function () {
                                setSelectedSlug(c.slug);
                                setChannelUnreadCounts(function (prev) {
                                  return Object.assign({}, prev, { [c.slug]: 0 });
                                });
                              },
                            });
                          }),
                        ),
                ),
              ),
            ),

            // Center: feed
            h("main", { className: "agora-feed" },
              h(Card, { className: "agora-panel agora-panel--feed" },
                h(CardHeader, { className: "agora-panel-header agora-feed-header" },
                  selectedChannel
                    ? h(CardTitle, { className: "agora-panel-title" },
                        h("span", { className: "agora-channel-hash" }, "#"),
                        channelDisplayName(selectedChannel),
                      )
                    : h(CardTitle, { className: "agora-panel-title" }, tx(t, "selectChannel", "Selecione um canal")),
                ),
                h(CardContent, { className: "agora-panel-body agora-feed-body" },
                  !selectedChannel
                    ? h(EmptyState, null, tx(t, "selectChannelHint", "Escolha um canal na barra lateral."))
                    : loadingMessages && messages.length === 0
                      ? h(EmptyState, null, h(LoadingDots, { label: "Carregando mensagens..." }))
                      : messages.length === 0
                        ? h(EmptyState, null, tx(t, "noMessages", "Nenhuma mensagem em #{channel} ainda.", { channel: channelDisplayName(selectedChannel) }))
                        : h("div", {
                            className: "agora-message-list",
                            role: "log",
                            "aria-live": "polite",
                            "aria-atomic": "false",
                            "aria-relevant": "additions",
                            ref: messagesListRef,
                            onScroll: throttledMessagesScroll,
                          },
                            loadingOlderMessages && h("div", { className: "agora-loading-dots agora-loading-dots--top" }, "Carregando mensagens anteriores..."),
                            mergedMessages.map(function (m) {
                              if (m.pending) {
                                return h(PendingMessageItem, {
                                  key: m.id,
                                  message: m,
                                  onRetry: retryPending,
                                  disabled: flushing,
                                });
                              }
                              return h(MessageItem, { key: m.id || m.created_at, message: m });
                            }),
                            !hasMoreMessages && messages.length >= PAGE_SIZE && h("div", { className: "agora-feed-end" }, "Início da conversa"),
                            h("div", { ref: messagesEndRef }),
                            showNewMessagesBtn && h("button", {
                              type: "button",
                              className: "agora-new-messages-btn",
                              onClick: function (e) { e.preventDefault(); scrollFeedToBottom(); },
                              "aria-label": feedNewCount > 0 ? feedNewCount + " novas mensagens" : "Novas mensagens",
                            }, feedNewCount > 0 ? feedNewCount + " novas mensagens" : "Novas mensagens"),
                          ),
                ),
                selectedChannel && connection !== "online" && h("div", { className: "agora-composer-offline" },
                  pendingMessages.length > 0
                    ? tx(t, "offlineQueue", "Sem conexão. {count} mensagem(ns) aguardando envio.", { count: pendingMessages.length })
                    : tx(t, "offlineHint", "Sem conexão. Mensagens digitadas serão enviadas ao reconectar."),
                ),
                selectedChannel && h("div", { className: "agora-composer" },
                  h(Input, {
                    className: "agora-composer-input",
                    ref: composerInputRef,
                    "aria-label": tx(t, "composerAriaLabel", "Mensagem para #{channel}", { channel: channelDisplayName(selectedChannel) }),
                    placeholder: tx(t, "composerPlaceholder", "Mensagem para #{channel}...", { channel: channelDisplayName(selectedChannel) }),
                    value: draft,
                    onChange: handleComposerChange,
                    onKeyDown: handleComposerKeyDown,
                    onKeyUp: handleComposerKeyUp,
                    onClick: handleComposerClick,
                    disabled: sending,
                    role: "combobox",
                    "aria-autocomplete": "list",
                    "aria-expanded": mentionOpen,
                    "aria-controls": mentionOpen ? "agora-mention-listbox" : null,
                    "aria-activedescendant": mentionOpen && mentionItems[mentionIndex] ? "agora-mention-opt-" + mentionIndex : null,
                  }),
                  h(Button, {
                    className: "agora-composer-btn",
                    onClick: sendMessage,
                    disabled: !draft.trim() || sending || connection === "offline",
                  }, sending ? tx(t, "sending", "Enviando") : tx(t, "send", "Enviar")),
                  mentionOpen && h(MentionAutocomplete, {
                    items: mentionItems,
                    selectedIndex: mentionIndex,
                    onSelect: insertMention,
                    onHover: setMentionIndex,
                    query: (function () {
                      const input = composerInputRef.current || document.querySelector(".agora-composer-input");
                      const ctx = getMentionContextAt(draft, input && input.selectionStart || 0);
                      return ctx ? ctx.query : "";
                    })(),
                    label: tx(t, "mentionsLabel", "Menções"),
                  }),
                ),
              ),
            ),

            // Right: agents
            h("aside", { className: cn("agora-sidebar", "agora-sidebar--right", showAgents ? "agora-sidebar--visible" : "agora-sidebar--hidden"), role: "complementary", "aria-label": tx(t, "agentsTitle", "Agentes") },
              h(Card, { className: "agora-panel" },
                h(CardHeader, { className: "agora-panel-header" },
                  h(CardTitle, { className: "agora-panel-title" }, tx(t, "agentsTitle", "Agentes")),
                ),
                h(CardContent, { className: "agora-panel-body" },
                  loadingAgents && visibleAgents.length === 0
                    ? h(EmptyState, null, h(LoadingDots, { label: "Carregando agentes..." }))
                    : visibleAgents.length === 0
                      ? h(EmptyState, null, tx(t, "noAgents", "Nenhum agente ativo."))
                      : h("div", { className: "agora-agent-list", role: "list", "aria-label": tx(t, "agentsListLabel", "Agentes ativos") },
                          visibleAgents.map(function (a) {
                            const profile = a.profile;
                            return h(AgentCard, {
                              key: profile,
                              agent: a,
                              worker: a.worker,
                              unreadCount: unreadCounts[profile] || 0,
                              notifications: agentNotifications[profile],
                              notificationsOpen: !!openNotifications[profile],
                              loadingNotifications: !!loadingNotifications[profile],
                              hasMoreNotifications: hasMoreNotifications[profile] !== false,
                              loadingOlderNotifications: !!loadingOlderNotifications[profile],
                              openingTerminal: openingTerminal,
                              onOpenTerminal: handleOpenTerminal,
                              onSummon: handleSummonAgent,
                              onToggleNotifications: function () { toggleNotifications(profile); },
                              onMarkRead: function (id) { markNotificationRead(profile, id); },
                              onMarkAllRead: function () { markAllNotificationsRead(profile); },
                              onScrollNotifications: throttledNotificationScroll[profile] || function () {},
                            });
                          }),
                        ),
                ),
              ),
            ),
          ),
    );
  }

  Registry.register("agora", AgoraPage);
})();

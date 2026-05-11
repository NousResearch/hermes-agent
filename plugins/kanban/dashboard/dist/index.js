/**
 * Hermes Kanban — Dashboard Plugin
 *
 * Board view for the multi-agent collaboration board backed by
 * ~/.hermes/kanban.db. Calls the plugin's backend at /api/plugins/kanban/
 * and tails task_events over a WebSocket for live updates.
 *
 * Plain IIFE, no build step. Uses window.__HERMES_PLUGIN_SDK__ for React +
 * shadcn primitives; HTML5 drag-and-drop for card movement on desktop and
 * a pointer-based fallback for touch.
 */
(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK) return;

  const { React } = SDK;
  const h = React.createElement;
  const {
    Card, CardContent,
    Badge, Button, Input, Label, Select, SelectOption,
  } = SDK.components;
  const { useState, useEffect, useCallback, useMemo, useRef } = SDK.hooks;
  const { cn, timeAgo } = SDK.utils;

  // useI18n is a hook each component calls locally. Older host dashboards
  // may not expose it yet; fall back to a shim so the bundle still renders
  // English against an older host SDK. English fallback strings live
  // alongside each call site (passed as the third arg of tx()).
  const useI18n = SDK.useI18n || function () { return { t: { kanban: null }, locale: "en" }; };

  // Resolve a translation by dotted path under the kanban namespace
  // (e.g. "columnLabels.triage"); fall back to the English string passed in.
  function tx(t, path, fallback, vars) {
    let node = t && t.kanban;
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

  // Order matches BOARD_COLUMNS in plugin_api.py.
  const COLUMN_ORDER = ["triage", "todo", "ready", "running", "blocked", "done"];
  // English fallback dictionaries — used when the i18n catalog is missing
  // a key, and as defaults for the get*() helpers below so callers running
  // outside any React component (where there's no `t`) still get sane text.
  const FALLBACK_COLUMN_LABEL = {
    triage: "Triage",
    todo: "Todo",
    ready: "Ready",
    running: "In Progress",
    blocked: "Blocked",
    done: "Done",
    archived: "Archived",
  };
  const FALLBACK_COLUMN_HELP = {
    triage: "Raw ideas — a specifier will flesh out the spec",
    todo: "Waiting on dependencies or unassigned",
    ready: "Assigned and waiting for a dispatcher tick",
    running: "Claimed by a worker — in-flight",
    blocked: "Worker asked for human input",
    done: "Completed",
    archived: "Archived",
  };
  const FALLBACK_DESTRUCTIVE = {
    done: "Mark this task as done? The worker's claim is released and dependent children become ready.",
    archived: "Archive this task? It disappears from the default board view.",
    blocked: "Mark this task as blocked? The worker's claim is released.",
  };
  const FALLBACK_DIAGNOSTIC_EVENT_LABELS = {
    completion_blocked_hallucination: "⚠ Completion blocked — phantom card ids",
    suspected_hallucinated_references: "⚠ Prose referenced phantom card ids",
  };
  const DIAGNOSTIC_EVENT_KIND_KEYS = {
    completion_blocked_hallucination: "completionBlockedHallucination",
    suspected_hallucinated_references: "suspectedHallucinatedReferences",
  };
  const DESTRUCTIVE_KEYS = {
    done: "confirmDone",
    archived: "confirmArchive",
    blocked: "confirmBlocked",
  };

  function getColumnLabel(t, status) {
    return tx(t, "columnLabels." + status, FALLBACK_COLUMN_LABEL[status] || status);
  }
  function getColumnHelp(t, status) {
    return tx(t, "columnHelp." + status, FALLBACK_COLUMN_HELP[status] || "");
  }
  function getDestructiveConfirm(t, status) {
    const key = DESTRUCTIVE_KEYS[status];
    if (!key) return null;
    return tx(t, key, FALLBACK_DESTRUCTIVE[status]);
  }
  function getDiagnosticEventLabel(t, kind) {
    const key = DIAGNOSTIC_EVENT_KIND_KEYS[kind];
    if (!key) return null;
    return tx(t, key, FALLBACK_DIAGNOSTIC_EVENT_LABELS[kind]);
  }

  const COLUMN_DOT = {
    triage: "hermes-kanban-dot-triage",
    todo: "hermes-kanban-dot-todo",
    ready: "hermes-kanban-dot-ready",
    running: "hermes-kanban-dot-running",
    blocked: "hermes-kanban-dot-blocked",
    done: "hermes-kanban-dot-done",
    archived: "hermes-kanban-dot-archived",
  };

  function isDiagnosticEvent(kind) {
    return Object.prototype.hasOwnProperty.call(FALLBACK_DIAGNOSTIC_EVENT_LABELS, kind);
  }

  function phantomIdsFromEvent(ev) {
    if (!ev || !ev.payload) return [];
    const p = ev.payload;
    return p.phantom_cards || p.phantom_refs || [];
  }

  // Takes an optional `t` so the prompt/alert text is localised. Callers
  // outside React components can pass null and fall through to English.
  function withCompletionSummary(patch, count, t) {
    if (!patch || patch.status !== "done") return patch;
    const label = count && count > 1 ? `${count} selected task(s)` : "this task";
    const value = window.prompt(
      tx(t, "completionSummary",
        "Completion summary for {label}. This is stored as the task result.",
        { label: label }),
      "",
    );
    if (value === null) return null;
    const summary = value.trim();
    if (!summary) {
      window.alert(tx(t, "completionSummaryRequired",
        "Completion summary is required before marking a task done."));
      return null;
    }
    return Object.assign({}, patch, { result: summary, summary });
  }

  const API = "/api/plugins/kanban";
  const MIME_TASK = "text/x-hermes-task";

  // Docs link — surfaced as a `?` icon next to the board switcher and as
  // `title=` hints on unlabelled controls. Kept in one place so rebrands or
  // path changes are a single edit.
  const DOCS_URL = "https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban";
  const DOCS_TUTORIAL_URL = "https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban-tutorial";

  // localStorage key for the user's selected board. Independent of the
  // CLI's on-disk ``<root>/kanban/current`` pointer so browser users
  // can inspect any board without shifting the CLI's active board out
  // from under a terminal they left open.
  const LS_BOARD_KEY = "hermes.kanban.selectedBoard";

  function readSelectedBoard() {
    try {
      const v = window.localStorage.getItem(LS_BOARD_KEY);
      return (v || "").trim() || null;
    } catch (_e) { return null; }
  }

  function writeSelectedBoard(slug) {
    try {
      // Persist the user's dashboard-side board pin even for "default".
      // Previously this stripped "default" to keep localStorage empty,
      // but the fetch layer read that absence as "no opinion" and fell
      // through to the server-side ``current`` file — which the board
      // switcher also writes. Result: selecting the default tab after
      // creating a new board with "switch" checked showed the new
      // board's (wrong) data because the URL omitted ``?board=`` and
      // the backend happily returned whichever board was "current".
      // Persisting every selection keeps the dashboard's board opinion
      // independent of the CLI's active board, which was the original
      // design intent. Regression: #20879.
      if (slug) window.localStorage.setItem(LS_BOARD_KEY, slug);
      else window.localStorage.removeItem(LS_BOARD_KEY);
    } catch (_e) { /* ignore quota / private mode */ }
  }

  function withBoard(url, board) {
    // Always append ?board=<slug> when we have one picked — including
    // "default". Omitting the param would fall through to the backend's
    // resolution chain (env var → ``current`` file → default), which
    // means the dashboard's tab selection gets silently overridden by
    // whatever board the CLI or "switch" checkbox last activated.
    // Regression: #20879.
    if (!board) return url;
    const sep = url.indexOf("?") >= 0 ? "&" : "?";
    return `${url}${sep}board=${encodeURIComponent(board)}`;
  }

  // The SDK's Select component fires ``onValueChange(value)`` directly
  // (it's a shadcn-style popup, not a native <select>). Older plugin
  // code calls ``onChange({target: {value}})`` which silently never
  // fires. This helper wires both signatures so a setter works with
  // either API — use it as:
  //
  //   h(Select, {..., ...selectChangeHandler(setState), ...})
  function selectChangeHandler(setter) {
    return {
      onValueChange: function (v) { setter(v == null ? "" : v); },
      onChange: function (e) {
        const v = e && e.target ? e.target.value : e;
        setter(v == null ? "" : v);
      },
    };
  }

  // -------------------------------------------------------------------------
  // Minimal safe markdown renderer.
  //
  // Recognises a small subset (headings, bold, italic, inline code, fenced
  // code, links, bullet lists, paragraphs). HTML escaping first, then
  // inline replacements against the escaped string — no raw HTML from the
  // user is ever executed.
  // -------------------------------------------------------------------------

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }
  function renderInline(esc) {
    // Fenced code has already been extracted before this runs; process
    // inline replacements on the escaped string.
    return esc
      // inline code
      .replace(/`([^`\n]+)`/g, (_m, c) => `<code>${c}</code>`)
      // bold
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      // italic
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      // safe links — only http(s) and mailto
      .replace(
        /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)/g,
        (_m, text, href) =>
          `<a href="${href}" target="_blank" rel="noopener noreferrer">${text}</a>`,
      );
  }
  function renderMarkdown(src) {
    if (!src) return "";
    // Split out fenced code blocks first so their contents aren't mangled.
    const blocks = [];
    let working = String(src).replace(/```([\s\S]*?)```/g, (_m, code) => {
      blocks.push(code);
      return `\u0000CODE${blocks.length - 1}\u0000`;
    });
    const escaped = escapeHtml(working);
    const lines = escaped.split(/\r?\n/);
    const out = [];
    let inList = false;
    for (const raw of lines) {
      const line = raw;
      const bullet = /^\s*[-*]\s+(.*)$/.exec(line);
      const heading = /^(#{1,4})\s+(.*)$/.exec(line);
      if (bullet) {
        if (!inList) { out.push("<ul>"); inList = true; }
        out.push(`<li>${renderInline(bullet[1])}</li>`);
        continue;
      }
      if (inList) { out.push("</ul>"); inList = false; }
      if (heading) {
        const level = heading[1].length;
        out.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      } else if (line.trim() === "") {
        out.push("");
      } else {
        out.push(`<p>${renderInline(line)}</p>`);
      }
    }
    if (inList) out.push("</ul>");
    let html = out.join("\n");
    // Re-insert fenced code blocks.
    html = html.replace(/\u0000CODE(\d+)\u0000/g, (_m, i) =>
      `<pre class="hermes-kanban-md-code"><code>${escapeHtml(blocks[Number(i)])}</code></pre>`,
    );
    return html;
  }

  function MarkdownBlock(props) {
    const enabled = props.enabled !== false;
    if (!enabled) {
      return h("pre", { className: "hermes-kanban-pre" }, props.source || "");
    }
    return h("div", {
      className: "hermes-kanban-md",
      dangerouslySetInnerHTML: { __html: renderMarkdown(props.source || "") },
    });
  }

  // -------------------------------------------------------------------------
  // Touch drag-drop helper.
  //
  // HTML5 DnD is desktop-only. On touch devices we attach a pointerdown
  // handler that simulates a drag proxy and fires a custom event on the
  // column under the finger when released. Columns listen for both the
  // standard `drop` event and our `hermes-kanban:drop` event.
  // -------------------------------------------------------------------------

  function attachTouchDrag(el, taskId) {
    if (!el) return;
    function onDown(e) {
      if (e.pointerType !== "touch") return;
      e.preventDefault();
      const proxy = el.cloneNode(true);
      proxy.classList.add("hermes-kanban-touch-proxy");
      document.body.appendChild(proxy);
      let lastTarget = null;

      function move(ev) {
        proxy.style.left = `${ev.clientX - proxy.offsetWidth / 2}px`;
        proxy.style.top = `${ev.clientY - 24}px`;
        proxy.style.display = "none";
        const under = document.elementFromPoint(ev.clientX, ev.clientY);
        proxy.style.display = "";
        const col = under && under.closest && under.closest("[data-kanban-column]");
        if (col !== lastTarget) {
          if (lastTarget) lastTarget.classList.remove("hermes-kanban-column--drop");
          if (col) col.classList.add("hermes-kanban-column--drop");
          lastTarget = col;
        }
      }
      function up() {
        document.removeEventListener("pointermove", move);
        document.removeEventListener("pointerup", up);
        document.removeEventListener("pointercancel", up);
        if (lastTarget) {
          lastTarget.classList.remove("hermes-kanban-column--drop");
          const status = lastTarget.getAttribute("data-kanban-column");
          lastTarget.dispatchEvent(new CustomEvent("hermes-kanban:drop", {
            detail: { taskId, status },
            bubbles: true,
          }));
        }
        proxy.remove();
      }
      // Kick off proxy at the pointer origin.
      proxy.style.position = "fixed";
      proxy.style.pointerEvents = "none";
      proxy.style.opacity = "0.85";
      proxy.style.zIndex = "9999";
      proxy.style.width = `${el.offsetWidth}px`;
      proxy.style.left = `${e.clientX - el.offsetWidth / 2}px`;
      proxy.style.top = `${e.clientY - 24}px`;
      document.addEventListener("pointermove", move);
      document.addEventListener("pointerup", up);
      document.addEventListener("pointercancel", up);
    }
    el.addEventListener("pointerdown", onDown);
    return function () { el.removeEventListener("pointerdown", onDown); };
  }

  // -------------------------------------------------------------------------
  // Error boundary
  // -------------------------------------------------------------------------

  // Wrap the boundary's fallback in a tiny function component so we can
  // call useI18n() — class components can't use hooks directly.
  function ErrorBoundaryFallback(props) {
    const { t } = useI18n();
    return h(Card, null,
      h(CardContent, { className: "p-6 text-sm" },
        h("div", { className: "text-destructive font-semibold mb-1" },
          tx(t, "renderingError", "Kanban tab hit a rendering error")),
        h("div", { className: "text-muted-foreground text-xs mb-3" },
          props.message),
        h(Button, {
          onClick: props.onReset,
          size: "sm",
        }, tx(t, "reloadView", "Reload view")),
      ),
    );
  }

  class ErrorBoundary extends React.Component {
    constructor(props) { super(props); this.state = { error: null }; }
    static getDerivedStateFromError(error) { return { error }; }
    componentDidCatch(error, info) {
      // eslint-disable-next-line no-console
      console.error("Kanban plugin crashed:", error, info);
    }
    render() {
      if (this.state.error) {
        return h(ErrorBoundaryFallback, {
          message: String(this.state.error && this.state.error.message || this.state.error),
          onReset: () => this.setState({ error: null }),
        });
      }
      return this.props.children;
    }
  }

  // -------------------------------------------------------------------------
  // Root page
  // -------------------------------------------------------------------------

  function KanbanPage() {
    const { t } = useI18n();
    const [board, setBoard] = useState(() => readSelectedBoard() || "default");
    const [boardList, setBoardList] = useState([]);      // [{slug, name, counts, ...}]
    const [showNewBoard, setShowNewBoard] = useState(false);

    const [kanbanBoard, setKanbanBoard] = useState(null);  // the grid data
    // Alias so the rest of the function can keep using `board` semantically
    // for the grid data (card columns + tenants + assignees) without
    // colliding with the selected-board slug above. History: the old
    // component had `const [board, setBoard]` for the grid data. We
    // renamed the grid data to `kanbanBoard` so the more useful name
    // (`board`) belongs to the selected slug.
    const boardData = kanbanBoard;
    const setBoardData = setKanbanBoard;
    const [config, setConfig] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const [tenantFilter, setTenantFilter] = useState("");
    const [assigneeFilter, setAssigneeFilter] = useState("");
    const [includeArchived, setIncludeArchived] = useState(false);
    const [search, setSearch] = useState("");
    const [laneByProfile, setLaneByProfile] = useState(true);
    const [configApplied, setConfigApplied] = useState(false);

    const [selectedTaskId, setSelectedTaskId] = useState(null);
    const [selectedIds, setSelectedIds] = useState(() => new Set());
    const [lastSelectedId, setLastSelectedId] = useState(null);
    const [failedIds, setFailedIds] = useState(() => new Set());
    const [draggingTaskId, setDraggingTaskId] = useState(null);
    const handleDragStart = useCallback(function (taskId) { setDraggingTaskId(taskId); }, []);
    const handleDragEnd = useCallback(function () { setDraggingTaskId(null); }, []);
    // Per-task event counter incremented whenever the WS stream reports
    // a new event for that task id. TaskDrawer useEffect-depends on its
    // own task's counter so it reloads itself on live events instead of
    // showing stale data.
    const [taskEventTick, setTaskEventTick] = useState({});

    const cursorRef = useRef(0);
    const reloadTimerRef = useRef(null);
    const wsRef = useRef(null);
    const wsBackoffRef = useRef(1000);
    const wsClosedRef = useRef(false);

    // --- load config once ---------------------------------------------------
    useEffect(function () {
      SDK.fetchJSON(withBoard(`${API}/config`, board))
        .then(function (c) {
          setConfig(c);
          if (!configApplied) {
            if (c.default_tenant) setTenantFilter(c.default_tenant);
            if (typeof c.lane_by_profile === "boolean") setLaneByProfile(c.lane_by_profile);
            if (typeof c.include_archived_by_default === "boolean") setIncludeArchived(c.include_archived_by_default);
            setConfigApplied(true);
          }
        })
        .catch(function () { setConfig({ render_markdown: true }); });
    }, []);  // eslint-disable-line react-hooks/exhaustive-deps

    // --- fetch full board ---------------------------------------------------
    const loadBoard = useCallback(() => {
      const qs = new URLSearchParams();
      if (tenantFilter) qs.set("tenant", tenantFilter);
      if (includeArchived) qs.set("include_archived", "true");
      const url = qs.toString() ? `${API}/board?${qs}` : `${API}/board`;
      return SDK.fetchJSON(withBoard(url, board))
        .then(function (data) {
          setBoardData(data);
          cursorRef.current = data.latest_event_id || 0;
          setError(null);
        })
        .catch(function (err) {
          setError(String(err && err.message ? err.message : err));
        })
        .finally(function () { setLoading(false); });
    }, [tenantFilter, includeArchived, board]);

    // --- load list of boards for the switcher ------------------------------
    const loadBoardList = useCallback(function () {
      return SDK.fetchJSON(withBoard(`${API}/boards`, board))
        .then(function (data) {
          const boards = (data && data.boards) || [];
          setBoardList(boards);
          // If the stored slug isn't in the list any longer (board was
          // deleted in the CLI while dashboard was open), fall back to
          // default so the UI doesn't hang on a 404.
          if (board !== "default" && !boards.find(function (b) { return b.slug === board; })) {
            setBoard("default");
            writeSelectedBoard("default");
          }
        })
        .catch(function () { /* non-fatal */ });
    }, [board]);

    useEffect(function () { loadBoardList(); }, [loadBoardList]);

    const scheduleReload = useCallback(function () {
      if (reloadTimerRef.current) return;
      reloadTimerRef.current = setTimeout(function () {
        reloadTimerRef.current = null;
        loadBoard();
      }, 250);
    }, [loadBoard]);

    useEffect(function () {
      loadBoard();
      return function () {
        if (reloadTimerRef.current) {
          clearTimeout(reloadTimerRef.current);
          reloadTimerRef.current = null;
        }
      };
    }, [loadBoard]);

    // --- WebSocket ---------------------------------------------------------
    useEffect(function () {
      if (!boardData) return undefined;
      wsClosedRef.current = false;
      function openWs() {
        if (wsClosedRef.current) return;
        const token = window.__HERMES_SESSION_TOKEN__ || "";
        const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
        const qsParams = {
          since: String(cursorRef.current || 0),
          token: token,
        };
        // Pin the WS stream to the currently-selected board so events
        // from other boards don't bleed in. Includes "default" so the
        // dashboard's own board pin always wins over the server-side
        // ``current`` file — same rationale as ``withBoard()`` above.
        // Regression: #20879.
        if (board) qsParams.board = board;
        const qs = new URLSearchParams(qsParams);
        const url = `${proto}//${window.location.host}${API}/events?${qs}`;
        let ws;
        try { ws = new WebSocket(url); } catch (_e) { return; }
        wsRef.current = ws;
        ws.onopen = function () { wsBackoffRef.current = 1000; };
        ws.onmessage = function (ev) {
          try {
            const msg = JSON.parse(ev.data);
            if (msg && Array.isArray(msg.events) && msg.events.length > 0) {
              cursorRef.current = msg.cursor || cursorRef.current;
              // Stamp per-task signal so the TaskDrawer can reload itself.
              setTaskEventTick(function (prev) {
                const next = Object.assign({}, prev);
                for (const e of msg.events) {
                  if (e && e.task_id) next[e.task_id] = (next[e.task_id] || 0) + 1;
                }
                return next;
              });
              scheduleReload();
            }
          } catch (_e) { /* ignore */ }
        };
        ws.onclose = function (ev) {
          if (wsClosedRef.current) return;
          if (ev && ev.code === 1008) {
            setError(tx(t, "wsAuthFailed",
              "WebSocket auth failed — reload the page to refresh the session token."));
            return;
          }
          const delay = Math.min(wsBackoffRef.current, 30000);
          wsBackoffRef.current = Math.min(wsBackoffRef.current * 2, 30000);
          setTimeout(openWs, delay);
        };
      }
      openWs();
      return function () {
        wsClosedRef.current = true;
        try { wsRef.current && wsRef.current.close(); } catch (_e) { /* noop */ }
      };
    }, [!!boardData, board, scheduleReload]);

    // --- filtering ----------------------------------------------------------
    const filteredBoard = useMemo(function () {
      if (!boardData) return null;
      const q = search.trim().toLowerCase();
      const filterTask = function (t) {
        if (tenantFilter && t.tenant !== tenantFilter) return false;
        if (assigneeFilter && t.assignee !== assigneeFilter) return false;
        if (q) {
          const hay = `${t.id} ${t.title || ""} ${t.body || ""} ${t.result || ""} ${t.latest_summary || ""} ${t.assignee || ""} ${t.tenant || ""}`.toLowerCase();
          if (hay.indexOf(q) === -1) return false;
        }
        return true;
      };
      return Object.assign({}, boardData, {
        columns: boardData.columns.map(function (col) {
          return Object.assign({}, col, { tasks: col.tasks.filter(filterTask) });
        }),
      });
    }, [boardData, tenantFilter, assigneeFilter, search]);

    // --- actions ------------------------------------------------------------
    const moveTask = useCallback(function (taskId, newStatus) {
      const confirmMsg = getDestructiveConfirm(t, newStatus);
      if (confirmMsg && !window.confirm(confirmMsg)) return;
      const patch = withCompletionSummary({ status: newStatus }, 1, t);
      if (!patch) return;
      setBoardData(function (b) {
        if (!b) return b;
        let moved = null;
        const columns = b.columns.map(function (col) {
          const next = col.tasks.filter(function (t) {
            if (t.id === taskId) { moved = Object.assign({}, t, { status: newStatus }); return false; }
            return true;
          });
          return Object.assign({}, col, { tasks: next });
        });
        if (moved) {
          const dest = columns.find(function (c) { return c.name === newStatus; });
          if (dest) dest.tasks = [moved].concat(dest.tasks);
        }
        return Object.assign({}, b, { columns });
      });
      SDK.fetchJSON(withBoard(`${API}/tasks/${encodeURIComponent(taskId)}`, board), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      }).catch(function (err) {
        setError(tx(t, "moveFailed", "Move failed: ") + (err.message || err));
        loadBoard();
      });
    }, [loadBoard, board, t]);

    const clearSelected = useCallback(function () {
      setSelectedIds(new Set());
      setLastSelectedId(null);
      setFailedIds(new Set());
    }, []);
    const moveSelected = useCallback(function (newStatus) {
      const confirmMsg = DESTRUCTIVE_TRANSITIONS[newStatus];
      if (confirmMsg && !window.confirm(confirmMsg)) return;
      if (selectedIds.size === 0) return;
      const patch = withCompletionSummary({ status: newStatus }, selectedIds.size);
      if (!patch) return;
      const ids = Array.from(selectedIds);
      // Optimistic UI: remove selected from all columns and prepend to target.
      setBoardData(function (b) {
        if (!b) return b;
        const moved = [];
        const columns = b.columns.map(function (col) {
          const kept = [];
          for (const t of col.tasks) {
            if (selectedIds.has(t.id)) moved.push(Object.assign({}, t, { status: newStatus }));
            else kept.push(t);
          }
          return Object.assign({}, col, { tasks: kept });
        });
        const dest = columns.find(function (c) { return c.name === newStatus; });
        if (dest) dest.tasks = moved.concat(dest.tasks);
        return Object.assign({}, b, { columns });
      });
      SDK.fetchJSON(withBoard(`${API}/tasks/bulk`, board), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(Object.assign({ ids }, patch)),
      }).then(function (res) {
        const failed = (res.results || []).filter(function (r) { return !r.ok; });
        if (failed.length > 0) {
          setError(`Bulk move: ${failed.length} of ${res.results.length} failed`);
          setFailedIds(new Set(failed.map(function (f) { return f.id; })));
        } else {
          setFailedIds(new Set());
        }
        setSelectedIds(new Set());
        setLastSelectedId(null);
        loadBoard();
      }).catch(function (err) {
        setError(`Move failed: ${err.message || err}`);
        setFailedIds(new Set(selectedIds));
        loadBoard();
      });
    }, [selectedIds, loadBoard, board]);

    const createTask = useCallback(function (body) {
      return SDK.fetchJSON(withBoard(`${API}/tasks`, board), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then(function (res) {
        // Surface dispatcher-presence warnings (e.g. "no gateway is
        // running") via the existing error banner channel. Not fatal —
        // the task was created successfully — but the user should know
        // their ready task will sit idle until the gateway is up.
        if (res && res.warning) {
          setError(tx(t, "taskCreatedWarning", "Task created, but: ") + res.warning);
        }
        loadBoard();
        loadBoardList();  // refresh counts in the switcher
        return res;
      });
    }, [loadBoard, loadBoardList, board, t]);

    const toggleSelected = useCallback(function (id, additive) {
      setSelectedIds(function (prev) {
        const next = new Set(additive ? prev : []);
        if (prev.has(id)) next.delete(id);
        else next.add(id);
        return next;
      });
      setLastSelectedId(id);
      setFailedIds(function (prev) {
        if (prev.has(id)) {
          const next = new Set(prev);
          next.delete(id);
          return next;
        }
        return prev;
      });
    }, []);

    const toggleRange = useCallback(function (toId) {
      // Build flat visible task order from filteredBoard columns.
      setSelectedIds(function (prev) {
        const next = new Set(prev);
        if (!filteredBoard || !filteredBoard.columns) return next;
        const order = [];
        for (const col of filteredBoard.columns) {
          for (const t of col.tasks || []) order.push(t.id);
        }
        const anchor = lastSelectedId;
        if (!anchor || anchor === toId) {
          next.add(toId);
          return next;
        }
        const aIdx = order.indexOf(anchor);
        const bIdx = order.indexOf(toId);
        if (aIdx === -1 || bIdx === -1) {
          next.add(toId);
          return next;
        }
        const lo = Math.min(aIdx, bIdx);
        const hi = Math.max(aIdx, bIdx);
        for (let i = lo; i <= hi; i++) next.add(order[i]);
        return next;
      });
      setLastSelectedId(toId);
    }, [filteredBoard, lastSelectedId]);

    const selectAllVisible = useCallback(function () {
      if (!filteredBoard || !filteredBoard.columns) return;
      const next = new Set();
      for (const col of filteredBoard.columns) {
        for (const t of col.tasks || []) next.add(t.id);
      }
      setSelectedIds(next);
      if (next.size > 0) {
        const first = Array.from(next)[0];
        setLastSelectedId(first);
      }
    }, [filteredBoard]);

    const selectAllInColumn = useCallback(function (columnName) {
      if (!filteredBoard || !filteredBoard.columns) return;
      const col = filteredBoard.columns.find(function (c) { return c.name === columnName; });
      if (!col) return;
      const allSelected = col.tasks && col.tasks.length > 0 && col.tasks.every(function (t) { return selectedIds.has(t.id); });
      const next = new Set(selectedIds);
      if (allSelected) {
        for (const t of col.tasks || []) next.delete(t.id);
      } else {
        for (const t of col.tasks || []) next.add(t.id);
      }
      setSelectedIds(next);
      if (col.tasks && col.tasks.length > 0) setLastSelectedId(col.tasks[0].id);
    }, [filteredBoard, selectedIds]);

    const applyBulk = useCallback(function (patch, confirmMsg) {
      if (selectedIds.size === 0) return;
      if (confirmMsg && !window.confirm(confirmMsg)) return;
      const finalPatch = withCompletionSummary(patch, selectedIds.size, t);
      if (!finalPatch) return;
      const body = Object.assign({ ids: Array.from(selectedIds) }, finalPatch);
      // Optimistic UI for status moves (same pattern as moveSelected).
      if (finalPatch.status) {
        setBoardData(function (b) {
          if (!b) return b;
          const moved = [];
          const columns = b.columns.map(function (col) {
            const kept = [];
            for (const t of col.tasks) {
              if (selectedIds.has(t.id)) moved.push(Object.assign({}, t, { status: finalPatch.status }));
              else kept.push(t);
            }
            return Object.assign({}, col, { tasks: kept });
          });
          const dest = columns.find(function (c) { return c.name === finalPatch.status; });
          if (dest) dest.tasks = moved.concat(dest.tasks);
          return Object.assign({}, b, { columns });
        });
      }
      SDK.fetchJSON(withBoard(`${API}/tasks/bulk`, board), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
        .then(function (res) {
          const failed = (res.results || []).filter(function (r) { return !r.ok; });
          if (failed.length > 0) {
            setError(tx(t, "bulkFailed", "Bulk: ") +
              `${failed.length} of ${res.results.length} failed: ` +
              failed.slice(0, 3).map(function (f) { return `${f.id} (${f.error})`; }).join("; "));
            setFailedIds(new Set(failed.map(function (f) { return f.id; })));
          } else {
            setFailedIds(new Set());
          }
          setSelectedIds(new Set());
          setLastSelectedId(null);
          loadBoard();
        })
        .catch(function (e) {
          setError(String(e.message || e));
          setFailedIds(new Set(selectedIds));
          loadBoard();
        });
    }, [selectedIds, loadBoard, board, t]);

    // --- board switching ----------------------------------------------------
    const switchBoard = useCallback(function (nextSlug) {
      if (!nextSlug || nextSlug === board) return;
      // Optimistic UI: clear the current grid + show loading, reset the
      // event cursor so the WS reopens aligned to the new board's
      // latest_event_id on the next loadBoard.
      setBoardData(null);
      cursorRef.current = 0;
      setLoading(true);
      setBoard(nextSlug);
      writeSelectedBoard(nextSlug);
      // Reset filters so stale search/tenant/assignee don't persist across boards.
      setSearch("");
      setTenantFilter("");
      setAssigneeFilter("");
      setIncludeArchived(false);
      clearSelected();
    }, [board, clearSelected]);

    const createNewBoard = useCallback(function (payload) {
      return SDK.fetchJSON(`${API}/boards`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }).then(function (res) {
        loadBoardList();
        const slug = res && res.board && res.board.slug;
        if (slug && payload.switch) switchBoard(slug);
        return res;
      });
    }, [loadBoardList, switchBoard, board]);

    const deleteBoard = useCallback(function (slug) {
      if (!slug || slug === "default") return Promise.resolve();
      return SDK.fetchJSON(`${API}/boards/${encodeURIComponent(slug)}`, {
        method: "DELETE",
      }).then(function () {
        loadBoardList();
        if (board === slug) switchBoard("default");
      });
    }, [board, loadBoardList, switchBoard]);

    // --- render -------------------------------------------------------------
    if (loading && !boardData) {
      return h("div", { className: "p-8 text-sm text-muted-foreground" },
        tx(t, "loading", "Loading Kanban board…"));
    }
    if (error && !boardData) {
      return h(Card, null,
        h(CardContent, { className: "p-6" },
          h("div", { className: "text-sm text-destructive" },
            tx(t, "loadFailed", "Failed to load Kanban board: "), error),
          h("div", { className: "text-xs text-muted-foreground mt-2" },
            tx(t, "loadFailedHint",
              "The backend auto-creates kanban.db on first read. If this persists, check the dashboard logs.")),
        ),
      );
    }
    if (!filteredBoard) return null;

    const renderMd = !config || config.render_markdown !== false;

    return h(ErrorBoundary, null,
      h("div", { className: "hermes-kanban flex flex-col gap-4" },
        h(BoardSwitcher, {
          board: board,
          boardList: boardList,
          onSwitch: switchBoard,
          onNewClick: function () { setShowNewBoard(true); },
          onDeleteBoard: deleteBoard,
        }),
        showNewBoard ? h(NewBoardDialog, {
          onCancel: function () { setShowNewBoard(false); },
          onCreate: function (payload) {
            return createNewBoard(payload).then(function () { setShowNewBoard(false); });
          },
        }) : null,
        h(AttentionStrip, {
          boardData,
          onOpen: setSelectedTaskId,
        }),
        h(BoardToolbar, {
          board: boardData,
          tenantFilter, setTenantFilter,
          assigneeFilter, setAssigneeFilter,
          includeArchived, setIncludeArchived,
          laneByProfile, setLaneByProfile,
          search, setSearch,
          onNudgeDispatch: function () {
            SDK.fetchJSON(withBoard(`${API}/dispatch?max=8`, board), { method: "POST" })
              .then(loadBoard)
              .catch(function (e) { setError(String(e.message || e)); });
          },
          onRefresh: loadBoard,
        }),
        selectedIds.size > 0 ? h(BulkActionBar, {
          count: selectedIds.size,
          assignees: (boardData && boardData.assignees) || [],
          onApply: applyBulk,
          onClear: clearSelected,
          onSelectAllVisible: selectAllVisible,
        }) : null,
        error ? h("div", { className: "text-xs text-destructive px-2" }, error) : null,
        h(BoardColumns, {
          board: filteredBoard,
          laneByProfile,
          selectedIds,
          failedIds,
          draggingTaskId,
          onDragStart: handleDragStart,
          onDragEnd: handleDragEnd,
          toggleSelected,
          toggleRange,
          selectAllInColumn,
          onMove: moveTask,
          onMoveSelected: moveSelected,
          onOpen: setSelectedTaskId,
          onCreate: createTask,
          allTasks: boardData.columns.reduce(function (acc, c) { return acc.concat(c.tasks); }, []),
        }),
        selectedTaskId ? h(TaskDrawer, {
          taskId: selectedTaskId,
          boardSlug: board,
          onClose: function () { setSelectedTaskId(null); },
          onRefresh: loadBoard,
          renderMarkdown: renderMd,
          allTasks: boardData.columns.reduce(function (acc, c) { return acc.concat(c.tasks); }, []),
          assignees: (boardData && boardData.assignees) || [],
          eventTick: taskEventTick[selectedTaskId] || 0,
        }) : null,
      ),
    );
  }

  // -------------------------------------------------------------------------
  // Attention strip — surfaces every task with active diagnostics,
  // severity-marked (warning/error/critical). Collapsed by default; click
  // Show to expand into per-task rows with Open buttons. Dismissible
  // per session via state flag.
  // -------------------------------------------------------------------------

  function collectDiagTasks(boardData) {
    if (!boardData || !boardData.columns) return [];
    const out = [];
    for (const col of boardData.columns) {
      for (const t of col.tasks || []) {
        if (t.diagnostics && t.diagnostics.length > 0) out.push(t);
        else if (t.warnings && t.warnings.count > 0) out.push(t);
      }
    }
    // Sort: highest severity first (critical > error > warning), then by
    // most recent latest_at.
    const sevIdx = function (s) {
      if (s === "critical") return 3;
      if (s === "error") return 2;
      if (s === "warning") return 1;
      return 0;
    };
    out.sort(function (a, b) {
      const aSev = sevIdx((a.warnings && a.warnings.highest_severity) || "warning");
      const bSev = sevIdx((b.warnings && b.warnings.highest_severity) || "warning");
      if (aSev !== bSev) return bSev - aSev;
      const aLa = (a.warnings && a.warnings.latest_at) || 0;
      const bLa = (b.warnings && b.warnings.latest_at) || 0;
      return bLa - aLa;
    });
    return out;
  }

  function AttentionStrip(props) {
    const { t } = useI18n();
    const [expanded, setExpanded] = useState(false);
    const [dismissed, setDismissed] = useState(false);
    const diagTasks = useMemo(
      function () { return collectDiagTasks(props.boardData); },
      [props.boardData]
    );
    if (dismissed || diagTasks.length === 0) return null;
    // Pick the highest severity present so we can colour the strip.
    let topSev = "warning";
    for (const td of diagTasks) {
      const s = (td.warnings && td.warnings.highest_severity) || "warning";
      if (s === "critical") { topSev = "critical"; break; }
      if (s === "error" && topSev !== "critical") topSev = "error";
    }
    return h("div", {
      className: cn(
        "hermes-kanban-attention",
        "hermes-kanban-attention--" + topSev,
      ),
    },
      h("div", { className: "hermes-kanban-attention-bar" },
        h("span", { className: "hermes-kanban-attention-icon" },
          topSev === "critical" ? "!!!" : topSev === "error" ? "!!" : "⚠"),
        h("span", { className: "hermes-kanban-attention-text" },
          diagTasks.length === 1
            ? tx(t, "taskNeedsAttention", "1 task needs attention")
            : tx(t, "tasksNeedAttention", "{n} tasks need attention",
                { n: diagTasks.length }),
        ),
        h("button", {
          className: "hermes-kanban-attention-toggle",
          onClick: function () { setExpanded(function (x) { return !x; }); },
          type: "button",
        }, expanded ? tx(t, "hide", "Hide") : tx(t, "show", "Show")),
        h("button", {
          className: "hermes-kanban-attention-dismiss",
          onClick: function () { setDismissed(true); },
          title: "Hide until next page reload",
          type: "button",
        }, "\u2715"),
      ),
      expanded
        ? h("div", { className: "hermes-kanban-attention-list" },
            diagTasks.map(function (task) {
              const sev = (task.warnings && task.warnings.highest_severity) || "warning";
              const kinds = task.warnings && task.warnings.kinds ? Object.keys(task.warnings.kinds) : [];
              return h("div", {
                key: task.id,
                className: cn(
                  "hermes-kanban-attention-row",
                  "hermes-kanban-attention-row--" + sev,
                ),
              },
                h("span", { className: "hermes-kanban-attention-row-sev" },
                  sev === "critical" ? "!!!" : sev === "error" ? "!!" : "⚠"),
                h("span", { className: "hermes-kanban-attention-row-id" }, task.id),
                h("span", { className: "hermes-kanban-attention-row-title" },
                  task.title || tx(t, "untitled", "(untitled)")),
                h("span", { className: "hermes-kanban-attention-row-meta" },
                  task.assignee ? "@" + task.assignee : tx(t, "unassigned", "unassigned"),
                  " \u00b7 ",
                  kinds.length > 0 ? kinds.join(", ") : tx(t, "diagnostic", "diagnostic"),
                ),
                h("button", {
                  className: "hermes-kanban-attention-row-btn",
                  onClick: function () { props.onOpen(task.id); },
                  type: "button",
                }, tx(t, "open", "Open")),
              );
            }),
          )
        : null,
    );
  }

  // -------------------------------------------------------------------------
  // Diagnostics section — generic renderer for a task's active distress
  // signals. Each diagnostic carries its own title, detail, data payload,
  // and a list of structured actions; the section renders them uniformly
  // regardless of kind. Replaces the hallucination-specific
  // ``RecoveryPopover`` from the previous iteration.
  //
  // Action kinds supported today:
  //   reclaim   → POST /tasks/:id/reclaim
  //   reassign  → POST /tasks/:id/reassign (with profile picker)
  //   unblock   → PATCH /tasks/:id  body: {status: "ready"}
  //   comment   → scroll to the comment input at the bottom of the drawer
  //   cli_hint  → copy payload.command to clipboard
  //   open_docs → open payload.url in a new tab
  // Unknown kinds are rendered as a disabled informational row so the
  // server can add new action kinds without breaking the UI.
  // -------------------------------------------------------------------------

  function DiagnosticActionButton(props) {
    const { t } = useI18n();
    const { action, onExec, busy, extra } = props;
    const label = (action.suggested ? "\u2606 " : "") + action.label;
    const cls = cn(
      "hermes-kanban-diag-action-btn",
      action.suggested ? "hermes-kanban-diag-action-btn--suggested" : "",
    );
    if (action.kind === "reclaim" || action.kind === "reassign" ||
        action.kind === "unblock") {
      return h("button", {
        className: cls,
        disabled: busy || (extra && extra.disabled),
        onClick: function () { onExec(action); },
        type: "button",
      }, label);
    }
    if (action.kind === "cli_hint") {
      return h("button", {
        className: cls,
        disabled: busy,
        onClick: function () { onExec(action); },
        type: "button",
        title: tx(t, "copyCommand", "Copy command to clipboard"),
      }, (extra && extra.copied) ? tx(t, "copied", "Copied") : label);
    }
    if (action.kind === "comment") {
      return h("button", {
        className: cls,
        onClick: function () { onExec(action); },
        type: "button",
      }, label);
    }
    if (action.kind === "open_docs") {
      return h("a", {
        className: cls,
        href: (action.payload && action.payload.url) || "#",
        target: "_blank",
        rel: "noreferrer",
      }, label);
    }
    // Unknown kind — render informational, non-interactive.
    return h("span", { className: cls + " hermes-kanban-diag-action-btn--unknown" },
      label);
  }

  function DiagnosticCard(props) {
    const { t } = useI18n();
    const { diag, task, boardSlug, assignees, onRefresh } = props;
    const [busy, setBusy] = useState(false);
    const [msg, setMsg] = useState(null);
    const [copiedKey, setCopiedKey] = useState(null);
    const [reassignProfile, setReassignProfile] = useState(task.assignee || "");

    const execAction = function (action) {
      if (busy) return;
      if (action.kind === "cli_hint") {
        const cmd = (action.payload && action.payload.command) || action.label;
        const fallback = function () { window.prompt("Copy this command:", cmd); };
        try {
          const p = navigator.clipboard && navigator.clipboard.writeText(cmd);
          if (p && p.then) {
            p.then(function () {
              setCopiedKey(action.label);
              setTimeout(function () { setCopiedKey(null); }, 2000);
            }).catch(fallback);
          } else {
            fallback();
          }
        } catch (_) {
          fallback();
        }
        return;
      }
      if (action.kind === "comment") {
        // Scroll the comment input into view; the drawer already has one
        // at the bottom. Focus it so the operator can start typing.
        const ta = document.querySelector(".hermes-kanban-drawer-comment-row input, .hermes-kanban-drawer-comment-row textarea");
        if (ta) {
          ta.scrollIntoView({ behavior: "smooth", block: "nearest" });
          ta.focus();
        }
        return;
      }
      if (action.kind === "unblock") {
        setBusy(true); setMsg(null);
        const url = withBoard(`${API}/tasks/${encodeURIComponent(task.id)}`, boardSlug);
        SDK.fetchJSON(url, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ status: "ready" }),
        }).then(function () {
          setMsg({ ok: true, text: tx(t, "unblockedMessage",
            "Unblocked {id}. Task is ready for the next tick.", { id: task.id }) });
          if (onRefresh) onRefresh();
        }).catch(function (err) {
          setMsg({ ok: false, text: tx(t, "unblockFailed", "Unblock failed: ") + (err.message || err) });
        }).then(function () { setBusy(false); });
        return;
      }
      if (action.kind === "reclaim") {
        setBusy(true); setMsg(null);
        const url = withBoard(`${API}/tasks/${encodeURIComponent(task.id)}/reclaim`, boardSlug);
        SDK.fetchJSON(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reason: `recovery action for ${diag.kind}` }),
        }).then(function () {
          setMsg({ ok: true, text: tx(t, "reclaimedMessage",
            "Reclaimed {id}. Task is back to ready.", { id: task.id }) });
          if (onRefresh) onRefresh();
        }).catch(function (err) {
          setMsg({ ok: false, text: tx(t, "reclaimFailed", "Reclaim failed: ") + (err.message || err) });
        }).then(function () { setBusy(false); });
        return;
      }
      if (action.kind === "reassign") {
        if (!reassignProfile) {
          setMsg({ ok: false, text: tx(t, "pickProfileFirst", "Pick a profile first.") });
          return;
        }
        setBusy(true); setMsg(null);
        const url = withBoard(`${API}/tasks/${encodeURIComponent(task.id)}/reassign`, boardSlug);
        const body = {
          profile: reassignProfile || null,
          reclaim_first: !!(action.payload && action.payload.reclaim_first),
          reason: `recovery action for ${diag.kind}`,
        };
        SDK.fetchJSON(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }).then(function () {
          setMsg({
            ok: true,
            text: tx(t, "reassignedMessage", "Reassigned {id} to {profile}.",
              { id: task.id, profile: reassignProfile }),
          });
          if (onRefresh) onRefresh();
        }).catch(function (err) {
          setMsg({ ok: false, text: tx(t, "reassignFailed", "Reassign failed: ") + (err.message || err) });
        }).then(function () { setBusy(false); });
        return;
      }
    };

    // Pull out the reassign action so we can render its picker inline.
    const reassignAction = (diag.actions || []).find(function (a) {
      return a.kind === "reassign";
    });

    const sevClass = "hermes-kanban-diag--" + (diag.severity || "warning");
    return h("div", { className: cn("hermes-kanban-diag", sevClass) },
      h("div", { className: "hermes-kanban-diag-header" },
        h("span", { className: "hermes-kanban-diag-sev" },
          diag.severity === "critical" ? "!!!" :
          diag.severity === "error" ? "!!" : "\u26a0"),
        h("span", { className: "hermes-kanban-diag-title" },
          diag.title),
      ),
      h("div", { className: "hermes-kanban-diag-detail" },
        diag.detail),
      diag.data && Object.keys(diag.data).length > 0
        ? h("div", { className: "hermes-kanban-diag-data" },
            Object.keys(diag.data).map(function (k) {
              const v = diag.data[k];
              if (Array.isArray(v) && v.length > 0 && typeof v[0] === "string" &&
                  v[0].indexOf("t_") === 0) {
                // Task-id list — render as chips.
                return h("div", { key: k, className: "hermes-kanban-diag-data-row" },
                  h("span", { className: "hermes-kanban-diag-data-key" }, k + ":"),
                  v.map(function (x) {
                    return h("code", {
                      key: x, className: "hermes-kanban-event-phantom-chip",
                    }, x);
                  }),
                );
              }
              return h("div", { key: k, className: "hermes-kanban-diag-data-row" },
                h("span", { className: "hermes-kanban-diag-data-key" }, k + ":"),
                h("span", { className: "hermes-kanban-diag-data-val" },
                  Array.isArray(v) ? v.join(", ") : String(v)),
              );
            }),
          )
        : null,
      // Inline reassign picker — only shown when the diagnostic offers
      // a reassign action. Profile list comes from the board payload.
      reassignAction
        ? h("div", { className: "hermes-kanban-diag-reassign-row" },
            h("span", { className: "hermes-kanban-diag-reassign-label" },
              tx(t, "reassignTo", "Reassign to:")),
            h("select", {
              className: "hermes-kanban-recovery-select",
              value: reassignProfile,
              onChange: function (e) { setReassignProfile(e.target.value); },
            },
              h("option", { value: "" }, "(unassigned)"),
              (assignees || []).map(function (a) {
                return h("option", { key: a, value: a }, a);
              }),
            ),
          )
        : null,
      h("div", { className: "hermes-kanban-diag-actions" },
        (diag.actions || []).map(function (a, i) {
          return h(DiagnosticActionButton, {
            key: a.kind + i,
            action: a,
            onExec: execAction,
            busy: busy,
            extra: {
              copied: copiedKey === a.label,
              disabled: (a.kind === "reassign" && !reassignProfile),
            },
          });
        }),
      ),
      msg
        ? h("div", {
            className: cn(
              "hermes-kanban-diag-msg",
              msg.ok ? "hermes-kanban-diag-msg--ok" : "hermes-kanban-diag-msg--err",
            ),
          }, msg.text)
        : null,
    );
  }

  function DiagnosticsSection(props) {
    const { t } = useI18n();
    const diags = props.diagnostics || [];
    const hasOpenDiags = diags.length > 0;
    const [open, setOpen] = useState(hasOpenDiags);
    useEffect(function () {
      if (hasOpenDiags) setOpen(true);
    }, [hasOpenDiags]);
    if (!hasOpenDiags && !props.alwaysVisible) {
      // Nothing active. Collapse the section entirely rather than showing
      // an empty "Recovery" header — keeps clean tasks visually clean.
      return null;
    }
    return h("div", { className: "hermes-kanban-section" },
      h("div", { className: "hermes-kanban-section-head-row" },
        h("span", { className: "hermes-kanban-section-head" },
          hasOpenDiags
            ? h("span", { className: "hermes-kanban-section-head-warning" },
                `\u26a0 ${tx(t, "diagnostics", "Diagnostics")} (${diags.length})`)
            : tx(t, "diagnostics", "Diagnostics"),
        ),
        h("button", {
          className: "hermes-kanban-section-toggle",
          onClick: function () { setOpen(function (x) { return !x; }); },
          type: "button",
        }, open ? tx(t, "hide", "Hide") : tx(t, "show", "Show")),
      ),
      open
        ? h("div", { className: "hermes-kanban-diag-list" },
            diags.map(function (d, i) {
              return h(DiagnosticCard, {
                key: props.task.id + ":" + d.kind + i,
                diag: d,
                task: props.task,
                boardSlug: props.boardSlug,
                assignees: props.assignees,
                onRefresh: props.onRefresh,
              });
            }),
          )
        : null,
    );
  }

    // -------------------------------------------------------------------------
  // Board switcher (multi-project)
  // -------------------------------------------------------------------------

  // Small `?` affordance next to the board controls. Opens the kanban docs
  // page in a new tab so users can look up what any of the widgets mean
  // without losing the current board view.
  function DocsLink() {
    return h("a", {
      href: DOCS_URL,
      target: "_blank",
      rel: "noopener noreferrer",
      className: "hermes-kanban-docs-link",
      title: "Open Hermes Kanban docs in a new tab",
      "aria-label": "Hermes Kanban documentation",
    }, "?");
  }

  function BoardSwitcher(props) {
    const { t } = useI18n();
    const list = props.boardList || [];
    const current = list.find(function (b) { return b.slug === props.board; });
    const currentName = current && current.name ? current.name : props.board;
    const currentTotal = current ? current.total : 0;
    const hasMultipleBoards = list.length > 1;

    // Hide entirely when only the default board exists AND it's empty —
    // single-project users never see boards UI unless they ask for it.
    // We show the [+ New board] affordance as soon as any board has a
    // task (so the user can discover multi-project before they need it)
    // OR when any non-default board exists.
    const totalAcrossAllBoards = list.reduce(function (n, b) { return n + (b.total || 0); }, 0);
    const shouldShow = hasMultipleBoards || totalAcrossAllBoards > 0;
    if (!shouldShow) {
      return h("div", {
        className: "hermes-kanban-boardswitcher-compact",
        title: tx(t, "boardSwitcherHint", "Boards let you separate unrelated streams of work"),
      },
        h(Button, {
          onClick: props.onNewClick,
          size: "sm",
          className: "h-7 text-xs",
        }, tx(t, "newBoard", "+ New board")),
        h(DocsLink, null),
      );
    }

    return h("div", { className: "hermes-kanban-boardswitcher" },
      h("div", { className: "hermes-kanban-boardswitcher-inner" },
        h("div", { className: "flex flex-col gap-0.5" },
          h("div", { className: "text-[11px] uppercase tracking-wider text-muted-foreground" },
            tx(t, "board", "Board")),
          h("div", { className: "flex items-center gap-2" },
            h(Select, Object.assign({
              value: props.board,
              className: "h-8 min-w-[220px]",
              "aria-label": "Switch kanban board",
              title: "Boards are independent work streams. Each board has its own tasks, tenants, and assignees.",
            }, selectChangeHandler(function (v) { if (v) props.onSwitch(v); })),
              list.map(function (b) {
                const label = b.total > 0
                  ? `${b.name || b.slug} · ${b.total}`
                  : (b.name || b.slug);
                return h(SelectOption, { key: b.slug, value: b.slug }, label);
              }),
            ),
            h("span", { className: "text-xs text-muted-foreground" },
              `${currentTotal || 0} task${currentTotal === 1 ? "" : "s"}`),
          ),
        ),
        h("div", { className: "flex-1" }),
        h(DocsLink, null),
        h(Button, {
          onClick: props.onNewClick,
          size: "sm",
          className: "h-8",
          title: "Create a new board. Useful when you want an unrelated work stream (different project, different team, isolated scratch area).",
        }, tx(t, "newBoard", "+ New board")),
        props.board !== "default"
          ? h(Button, {
            onClick: function () {
              const msg = tx(t, "archiveBoardConfirm",
                "Archive board '{name}'? It will be moved to boards/_archived/ so you can recover it later. Tasks on this board will no longer appear anywhere in the UI.",
                { name: currentName });
              if (window.confirm(msg)) props.onDeleteBoard(props.board);
            },
            size: "sm",
            className: "h-8",
            title: tx(t, "archiveBoardTitle", "Archive this board"),
          }, tx(t, "archive", "Archive"))
          : null,
      ),
    );
  }

  function NewBoardDialog(props) {
    const { t } = useI18n();
    const [slug, setSlug] = useState("");
    const [name, setName] = useState("");
    const [description, setDescription] = useState("");
    const [icon, setIcon] = useState("");
    const [switchTo, setSwitchTo] = useState(true);
    const [submitting, setSubmitting] = useState(false);
    const [err, setErr] = useState(null);

    // Auto-derive a name from the slug if the user hasn't typed one.
    const autoName = useMemo(function () {
      if (!slug) return "";
      return slug.replace(/[-_]+/g, " ")
        .split(" ")
        .filter(Boolean)
        .map(function (w) { return w[0].toUpperCase() + w.slice(1); })
        .join(" ");
    }, [slug]);

    function onSubmit(ev) {
      if (ev) ev.preventDefault();
      if (!slug.trim()) { setErr("slug is required"); return; }
      setSubmitting(true);
      setErr(null);
      props.onCreate({
        slug: slug.trim(),
        name: name.trim() || autoName || undefined,
        description: description.trim() || undefined,
        icon: icon.trim() || undefined,
        switch: switchTo,
      }).catch(function (e) {
        setErr(String(e && e.message ? e.message : e));
        setSubmitting(false);
      });
    }

    return h("div", {
      className: "hermes-kanban-dialog-backdrop",
      onClick: function (e) { if (e.target === e.currentTarget) props.onCancel(); },
    },
      h("form", {
        className: "hermes-kanban-dialog",
        onSubmit: onSubmit,
      },
        h("div", { className: "hermes-kanban-dialog-title" },
          tx(t, "newBoardTitle", "New board")),
        h("div", { className: "text-xs text-muted-foreground mb-2" },
          tx(t, "newBoardDescription",
            "Boards let you separate unrelated streams of work — one per project, repo, or domain. Workers on one board never see another board's tasks.")),
        h("div", { className: "flex flex-col gap-3" },
          h("div", { className: "flex flex-col gap-1" },
            h(Label, { className: "text-xs" }, tx(t, "slug", "Slug"), " ",
              h("span", { className: "text-muted-foreground" },
                tx(t, "slugHint", "— lowercase, hyphens, e.g. atm10-server"))),
            h(Input, {
              value: slug,
              onChange: function (e) { setSlug(e.target.value.toLowerCase().replace(/[^a-z0-9\-_]/g, "-")); },
              placeholder: "atm10-server",
              autoFocus: true,
              className: "h-8",
            }),
          ),
          h("div", { className: "flex flex-col gap-1" },
            h(Label, { className: "text-xs" }, tx(t, "displayName", "Display name"), " ",
              h("span", { className: "text-muted-foreground" },
                tx(t, "displayNameHint", "(optional)"))),
            h(Input, {
              value: name,
              onChange: function (e) { setName(e.target.value); },
              placeholder: autoName || tx(t, "displayName", "Display name"),
              className: "h-8",
            }),
          ),
          h("div", { className: "flex flex-col gap-1" },
            h(Label, { className: "text-xs" }, tx(t, "description", "Description"), " ",
              h("span", { className: "text-muted-foreground" },
                tx(t, "descriptionHint", "(optional)"))),
            h(Input, {
              value: description,
              onChange: function (e) { setDescription(e.target.value); },
              placeholder: "What goes on this board?",
              className: "h-8",
            }),
          ),
          h("div", { className: "flex flex-col gap-1" },
            h(Label, { className: "text-xs" }, tx(t, "icon", "Icon"), " ",
              h("span", { className: "text-muted-foreground" },
                tx(t, "iconHint", "(single character or emoji)"))),
            h(Input, {
              value: icon,
              onChange: function (e) { setIcon(e.target.value.slice(0, 4)); },
              placeholder: "📦",
              className: "h-8 w-24",
            }),
          ),
          h("label", { className: "flex items-center gap-2 text-xs" },
            h("input", {
              type: "checkbox",
              checked: switchTo,
              onChange: function (e) { setSwitchTo(e.target.checked); },
            }),
            tx(t, "switchAfterCreate", "Switch to this board after creating it"),
          ),
        ),
        err ? h("div", { className: "text-xs text-destructive mt-2" }, err) : null,
        h("div", { className: "hermes-kanban-dialog-actions" },
          h(Button, {
            type: "button",
            onClick: props.onCancel,
            size: "sm",
            disabled: submitting,
          }, tx(t, "cancel", "Cancel")),
          h(Button, {
            type: "submit",
            size: "sm",
            disabled: submitting || !slug.trim(),
          }, submitting ? tx(t, "creating", "Creating…") : tx(t, "createBoard", "Create board")),
        ),
      ),
    );
  }

  // -------------------------------------------------------------------------
  // Toolbar
  // -------------------------------------------------------------------------

  function BoardToolbar(props) {
    const { t } = useI18n();
    const tenants = (props.board && props.board.tenants) || [];
    const assignees = (props.board && props.board.assignees) || [];
    return h("div", { className: "flex flex-wrap items-end gap-3" },
      h("div", { className: "flex flex-col gap-1",
                 title: "Fuzzy-match tasks by id, title, or description. Matches across all columns." },
        h(Label, { className: "text-xs text-muted-foreground" }, tx(t, "search", "Search")),
        h(Input, {
          placeholder: tx(t, "filterCards", "Filter cards…"),
          value: props.search,
          onChange: function (e) { props.setSearch(e.target.value); },
          className: "w-56 h-8",
        }),
      ),
      h("div", { className: "flex flex-col gap-1",
                 title: "Tenants are free-form tags on a task (e.g. customer, project, team). Set them via the task drawer or kanban_create." },
        h(Label, { className: "text-xs text-muted-foreground" }, tx(t, "tenant", "Tenant")),
        h(Select, Object.assign({
          value: props.tenantFilter,
          className: "h-8",
        }, selectChangeHandler(props.setTenantFilter)),
          h(SelectOption, { value: "" }, tx(t, "allTenants", "All tenants")),
          tenants.map(function (tn) {
            return h(SelectOption, { key: tn, value: tn }, tn);
          }),
        ),
      ),
      h("div", { className: "flex flex-col gap-1",
                 title: "Filter by assigned Hermes profile. Profiles are the named agent identities that claim and work on tasks." },
        h(Label, { className: "text-xs text-muted-foreground" }, tx(t, "assignee", "Assignee")),
        h(Select, Object.assign({
          value: props.assigneeFilter,
          className: "h-8",
        }, selectChangeHandler(props.setAssigneeFilter)),
          h(SelectOption, { value: "" }, tx(t, "allProfiles", "All profiles")),
          assignees.map(function (a) {
            return h(SelectOption, { key: a, value: a }, a);
          }),
        ),
      ),
      h("label", { className: "flex items-center gap-2 text-xs",
                   title: "Include archived tasks in the board view. Archived tasks are hidden by default." },
        h("input", {
          type: "checkbox",
          checked: props.includeArchived,
          onChange: function (e) { props.setIncludeArchived(e.target.checked); },
        }),
        tx(t, "showArchived", "Show archived"),
      ),
      h("label", { className: "flex items-center gap-2 text-xs",
                   title: "Group the Running column by assigned profile" },
        h("input", {
          type: "checkbox",
          checked: props.laneByProfile,
          onChange: function (e) { props.setLaneByProfile(e.target.checked); },
        }),
        tx(t, "lanesByProfile", "Lanes by profile"),
      ),
      h("div", { className: "flex-1" }),
      h(Button, {
        onClick: props.onNudgeDispatch,
        size: "sm",
        title: "Wake the dispatcher to claim ready tasks now instead of waiting for the next tick. Use this after adding tasks if you want them picked up immediately.",
      }, tx(t, "nudgeDispatcher", "Nudge dispatcher")),
      h(Button, {
        onClick: props.onRefresh,
        size: "sm",
        title: "Reload the board from the database. The board auto-refreshes on task events; this is for forcing a re-read.",
      }, tx(t, "refresh", "Refresh")),
      h(Button, {
        onClick: function () {
          props.setSearch("");
          props.setTenantFilter("");
          props.setAssigneeFilter("");
          props.setIncludeArchived(false);
        },
        size: "sm",
        title: "Clear all active filters (search, tenant, assignee, archived).",
      }, tx(t, "clearFilters", "Clear filters")),
    );
  }

  // -------------------------------------------------------------------------
  // Bulk action bar (appears when >= 1 card is selected)
  // -------------------------------------------------------------------------

  function BulkActionBar(props) {
    const { t } = useI18n();
    const [assignee, setAssignee] = useState("");
    const [reclaimFirst, setReclaimFirst] = useState(false);
    const [priority, setPriority] = useState("");
    return h("div", { className: "hermes-kanban-bulk" },
      h("span", { className: "hermes-kanban-bulk-count" },
        `${props.count} ${tx(t, "selected", "selected")}`),
      h(Button, {
        onClick: function () { props.onApply({ status: "todo" }); },
        size: "sm",
        title: "Move selected tasks to Todo.",
      }, "→ todo"),
      h(Button, {
        onClick: function () { props.onApply({ status: "ready" }); },
        size: "sm",
        title: "Move selected tasks to Ready. Ready tasks are picked up by the dispatcher on the next tick.",
      }, "→ ready"),
      h(Button, {
        onClick: function () { props.onApply({ status: "blocked" },
          `Block ${props.count} task(s)?`); },
        size: "sm",
        title: "Block selected tasks. Releases any active claims.",
      }, "Block"),
      h(Button, {
        onClick: function () { props.onApply({ status: "ready" },
          `Unblock ${props.count} task(s)?`); },
        size: "sm",
        title: "Unblock selected tasks (promote to Ready).",
      }, "Unblock"),
      h(Button, {
        onClick: function () {
          props.onApply({ status: "done" },
            tx(t, "markDone", "Mark {n} task(s) as done?", { n: props.count }));
        },
        size: "sm",
        title: "Mark selected tasks as done. Releases any claims and unblocks dependent children. You'll be asked for a completion summary.",
      }, tx(t, "complete", "Complete")),
      h(Button, {
        onClick: function () {
          props.onApply({ archive: true },
            tx(t, "markArchived", "Archive {n} task(s)?", { n: props.count }));
        },
        size: "sm",
        title: "Archive selected tasks. They disappear from the default board view but remain in the database.",
      }, tx(t, "archive", "Archive")),
      h("div", { className: "hermes-kanban-bulk-priority",
                 title: "Set priority on selected tasks. Higher = claimed first." },
        h(Input, {
          type: "number",
          value: priority,
          onChange: function (e) { setPriority(e.target.value); },
          placeholder: tx(t, "priority", "pri"),
          className: "h-7 text-xs w-16",
        }),
        h(Button, {
          onClick: function () {
            if (priority === "") return;
            props.onApply({ priority: Number(priority) });
            setPriority("");
          },
          disabled: priority === "",
          size: "sm",
        }, tx(t, "setPriority", "Set priority")),
      ),
      h("div", { className: "hermes-kanban-bulk-reassign",
                 title: "Reassign selected tasks to a different Hermes profile. Pick a profile (or unassign) and click Apply." },
        h(Select, {
          value: assignee,
          onChange: function (e) { setAssignee(e.target.value); },
          className: "h-7 text-xs",
        },
          h(SelectOption, { value: "" }, "— reassign —"),
          h(SelectOption, { value: "__none__" }, "(unassign)"),
          props.assignees.map(function (a) {
            return h(SelectOption, { key: a, value: a }, a);
          }),
        ),
        h(Button, {
          onClick: function () {
            if (!assignee) return;
            props.onApply({ assignee: assignee === "__none__" ? "" : assignee, reclaim_first: reclaimFirst });
            setAssignee("");
          },
          disabled: !assignee,
          size: "sm",
          title: "Apply the selected assignee to all selected tasks.",
        }, tx(t, "apply", "Apply")),
      ),
      h("label", { className: "hermes-kanban-bulk-reclaim-first", title: "Reclaim any active claims before reassigning" },
        h("input", {
          type: "checkbox",
          checked: reclaimFirst,
          onChange: function (e) { setReclaimFirst(e.target.checked); },
        }),
        "Reclaim first",
      ),
      h("div", { className: "flex-1" }),
      h(Button, {
        onClick: props.onSelectAllVisible,
        size: "sm",
        title: "Select all visible cards across columns.",
      }, "Select all visible"),
      h(Button, {
        onClick: props.onClear,
        size: "sm",
        title: "Deselect all tasks and hide this bar.",
      }, tx(t, "clear", "Clear")),
    );
  }

  // -------------------------------------------------------------------------
  // Columns
  // -------------------------------------------------------------------------

  function BoardColumns(props) {
    const handleDragStart = useCallback(function (e) {
      const card = e.target.closest && e.target.closest(".hermes-kanban-card");
      if (!card) return;
      const taskId = card.getAttribute("data-task-id");
      if (taskId && props.onDragStart) props.onDragStart(taskId);
    }, [props.onDragStart]);
    const handleDragEnd = useCallback(function () {
      if (props.onDragEnd) props.onDragEnd();
    }, [props.onDragEnd]);
    return h("div", { className: "hermes-kanban-columns", onDragStart: handleDragStart, onDragEnd: handleDragEnd },
      props.board.columns.map(function (col) {
        return h(Column, {
          key: col.name,
          column: col,
          laneByProfile: props.laneByProfile,
          selectedIds: props.selectedIds,
          failedIds: props.failedIds,
          draggingTaskId: props.draggingTaskId,
          toggleSelected: props.toggleSelected,
          toggleRange: props.toggleRange,
          selectAllInColumn: props.selectAllInColumn,
          onMove: props.onMove,
          onMoveSelected: props.onMoveSelected,
          onOpen: props.onOpen,
          onCreate: props.onCreate,
          allTasks: props.allTasks,
        });
      }),
    );
  }

  function Column(props) {
    const { t } = useI18n();
    const [dragOver, setDragOver] = useState(false);
    const [showCreate, setShowCreate] = useState(false);
    const colRef = useRef(null);

    // Listen for our synthetic touch-drop events from attachTouchDrag().
    useEffect(function () {
      if (!colRef.current) return undefined;
      const el = colRef.current;
      function onTouchDrop(e) {
        if (e.detail && e.detail.status === props.column.name) {
          const taskId = e.detail.taskId;
          if (props.selectedIds && props.selectedIds.has(taskId) && props.selectedIds.size > 1 && props.onMoveSelected) {
            props.onMoveSelected(props.column.name);
          } else {
            props.onMove(taskId, props.column.name);
          }
        }
      }
      el.addEventListener("hermes-kanban:drop", onTouchDrop);
      return function () { el.removeEventListener("hermes-kanban:drop", onTouchDrop); };
    }, [props.column.name, props.onMove, props.selectedIds, props.onMoveSelected]);

    const handleDragOver = function (e) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      if (!dragOver) setDragOver(true);
    };
    const handleDragLeave = function () { setDragOver(false); };
    const handleDrop = function (e) {
      e.preventDefault();
      setDragOver(false);
      const taskId = e.dataTransfer.getData(MIME_TASK);
      if (!taskId) return;
      if (props.selectedIds && props.selectedIds.has(taskId) && props.selectedIds.size > 1) {
        if (props.onMoveSelected) props.onMoveSelected(props.column.name);
      } else {
        props.onMove(taskId, props.column.name);
      }
    };

    const lanes = useMemo(function () {
      if (!props.laneByProfile || props.column.name !== "running") return null;
      const byProfile = {};
      for (const tk of props.column.tasks) {
        const key = tk.assignee || "(unassigned)";
        (byProfile[key] = byProfile[key] || []).push(tk);
      }
      return Object.keys(byProfile).sort().map(function (k) {
        return { assignee: k, tasks: byProfile[k] };
      });
    }, [props.column, props.laneByProfile]);

    const colHelp = getColumnHelp(t, props.column.name);
    const colLabel = getColumnLabel(t, props.column.name);

    return h("div", {
      ref: colRef,
      "data-kanban-column": props.column.name,
      className: cn(
        "hermes-kanban-column",
        dragOver ? "hermes-kanban-column--drop" : "",
      ),
      onDragOver: handleDragOver,
      onDragLeave: handleDragLeave,
      onDrop: handleDrop,
    },
      h("div", { className: "hermes-kanban-column-header",
                 title: colHelp || "" },
        h("input", {
          type: "checkbox",
          className: "hermes-kanban-col-check",
          title: "Select all tasks in this column",
          "aria-label": `Select all tasks in ${COLUMN_LABEL[props.column.name] || props.column.name}`,
          checked: props.column.tasks.length > 0 && props.column.tasks.every(function (t) { return props.selectedIds.has(t.id); }),
          onChange: function (e) {
            e.stopPropagation();
            if (props.selectAllInColumn) props.selectAllInColumn(props.column.name);
          },
          onClick: function (e) { e.stopPropagation(); },
        }),
        h("span", { className: cn("hermes-kanban-dot", COLUMN_DOT[props.column.name]) }),
        h("span", { className: "hermes-kanban-column-label" },
          colLabel || props.column.name),
        h("span", { className: "hermes-kanban-column-count",
                    title: `${props.column.tasks.length} task${props.column.tasks.length === 1 ? "" : "s"} in this column` },
          props.column.tasks.length),
        h("button", {
          type: "button",
          className: "hermes-kanban-column-add",
          title: tx(t, "createTask", "Create task in this column"),
          onClick: function () { setShowCreate(function (v) { return !v; }); },
        }, showCreate ? "×" : "+"),
      ),
      h("div", { className: "hermes-kanban-column-sub" },
        colHelp || ""),
      showCreate ? h(InlineCreate, {
        columnName: props.column.name,
        allTasks: props.allTasks,
        onSubmit: function (body) {
          props.onCreate(body).then(function () { setShowCreate(false); });
        },
        onCancel: function () { setShowCreate(false); },
      }) : null,
      h("div", { className: "hermes-kanban-column-body" },
        props.column.tasks.length === 0
          ? h("div", { className: "hermes-kanban-empty" }, tx(t, "noTasks", "— no tasks —"))
          : lanes
            ? lanes.map(function (lane) {
                return h("div", { key: lane.assignee, className: "hermes-kanban-lane" },
                  h("div", { className: "hermes-kanban-lane-head" },
                    h("span", { className: "hermes-kanban-lane-name" }, lane.assignee),
                    h("span", { className: "hermes-kanban-lane-count" }, lane.tasks.length),
                  ),
                  lane.tasks.map(function (tk) {
                    return h(TaskCard, {
                      key: tk.id, task: tk,
                      selected: props.selectedIds.has(tk.id),
                      failed: props.failedIds && props.failedIds.has(tk.id),
                      draggingTaskId: props.draggingTaskId,
                      draggingSource: props.draggingTaskId && props.selectedIds.has(props.draggingTaskId) && props.selectedIds.size > 1 && props.selectedIds.has(tk.id),
                      toggleSelected: props.toggleSelected,
                      toggleRange: props.toggleRange,
                      onOpen: props.onOpen,
                    });
                  }),
                );
              })
            : props.column.tasks.map(function (tk) {
                return h(TaskCard, {
                  key: tk.id, task: tk,
                  selected: props.selectedIds.has(tk.id),
                  failed: props.failedIds && props.failedIds.has(tk.id),
                  draggingTaskId: props.draggingTaskId,
                  draggingSource: props.draggingTaskId && props.selectedIds.has(props.draggingTaskId) && props.selectedIds.size > 1 && props.selectedIds.has(tk.id),
                  toggleSelected: props.toggleSelected,
                  toggleRange: props.toggleRange,
                  onOpen: props.onOpen,
                });
              }),
      ),
    );
  }

  // -------------------------------------------------------------------------
  // Card
  // -------------------------------------------------------------------------

  // Staleness tiers — amber after a grace window, red when clearly stuck.
  // Values below are seconds.
  const STALENESS = {
    ready:   { amber: 1 * 60 * 60,   red: 24 * 60 * 60 },
    running: { amber: 10 * 60,       red: 60 * 60 },
    blocked: { amber: 1 * 60 * 60,   red: 24 * 60 * 60 },
    todo:    { amber: 7 * 24 * 60 * 60, red: 30 * 24 * 60 * 60 },
  };

  function stalenessClass(task) {
    if (!task || !task.age) return "";
    const age = task.status === "running"
      ? task.age.started_age_seconds
      : task.age.created_age_seconds;
    const tier = STALENESS[task.status];
    if (!tier || age == null) return "";
    if (age >= tier.red)   return "hermes-kanban-card--stale-red";
    if (age >= tier.amber) return "hermes-kanban-card--stale-amber";
    return "";
  }

  function TaskCard(props) {
    const { t: i18n } = useI18n();
    const t = props.task;
    const cardRef = useRef(null);

    useEffect(function () {
      return attachTouchDrag(cardRef.current, t.id);
    }, [t.id]);

    const handleDragStart = function (e) {
      e.dataTransfer.setData(MIME_TASK, t.id);
      e.dataTransfer.effectAllowed = "move";
      const selectedCards = document.querySelectorAll(".hermes-kanban-card--selected");
      if (selectedCards.length > 1 && props.selected) {
        const ghost = document.createElement("div");
        ghost.className = "hermes-kanban-drag-ghost";
        ghost.textContent = selectedCards.length + " cards";
        document.body.appendChild(ghost);
        e.dataTransfer.setDragImage(ghost, 0, 0);
        requestAnimationFrame(function () {
          if (ghost.parentNode) document.body.removeChild(ghost);
        });
      }
    };
    const handleClick = function (e) {
      if (e.shiftKey) {
        e.preventDefault();
        e.stopPropagation();
        if (props.toggleRange) props.toggleRange(t.id);
        return;
      }
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        e.stopPropagation();
        props.toggleSelected(t.id, true);
        return;
      }
      props.onOpen(t.id);
    };
    const handleKeyDown = function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        props.onOpen(t.id);
      }
      if (e.key === "Escape") {
        if (props.toggleSelected) props.toggleSelected(t.id, false);
      }
    };
    const handleCheckbox = function (e) {
      e.stopPropagation();
      props.toggleSelected(t.id, true);
    };

    const progress = t.progress;

    return h("div", {
      ref: cardRef,
      "data-task-id": t.id,
      className: cn(
        "hermes-kanban-card",
        props.selected ? "hermes-kanban-card--selected" : "",
        props.failed ? "hermes-kanban-card--failed" : "",
        props.draggingSource ? "hermes-kanban-card--dragging-source" : "",
        stalenessClass(t),
      ),
      draggable: true,
      tabIndex: 0,
      role: "button",
      "aria-label": `${t.title || "untitled"} — ${t.id} — ${t.status}`,
      onDragStart: handleDragStart,
      onClick: handleClick,
      onKeyDown: handleKeyDown,
    },
      h(Card, null,
        h(CardContent, { className: "hermes-kanban-card-content" },
          h("div", { className: "hermes-kanban-card-row" },
            h("label", {
              className: "hermes-kanban-card-check-wrap",
              title: tx(i18n, "selectForBulk", "Select for bulk actions"),
              onClick: function (e) { e.stopPropagation(); },
            },
              h("input", {
                type: "checkbox",
                className: "hermes-kanban-card-check",
                checked: props.selected,
                onChange: handleCheckbox,
                onClick: function (e) { e.stopPropagation(); },
                "aria-label": `Select task ${t.id}`,
              }),
            ),
            h("span", { className: "hermes-kanban-card-id",
                        title: `Task id: ${t.id}. Use this id with kanban_show, /kanban show, or hermes kanban show.` }, t.id),
            t.warnings && t.warnings.count > 0
              ? h("span", {
                  className: cn(
                    "hermes-kanban-warning-badge",
                    "hermes-kanban-warning-badge--" + (t.warnings.highest_severity || "warning"),
                  ),
                  title: (
                    `${t.warnings.count} active diagnostic` +
                    (t.warnings.count === 1 ? "" : "s") +
                    ` (severity: ${t.warnings.highest_severity || "warning"}). ` +
                    `Click to open for details.`
                  ),
                }, t.warnings.highest_severity === "critical" ? "!!!" :
                   t.warnings.highest_severity === "error" ? "!!" : "⚠")
              : null,
            t.priority > 0
              ? h(Badge, { className: "hermes-kanban-priority",
                           title: `Priority ${t.priority}. Higher-priority tasks are claimed first by the dispatcher.` }, `P${t.priority}`)
              : null,
            t.tenant
              ? h(Badge, { variant: "outline", className: "hermes-kanban-tag",
                           title: `Tenant: ${t.tenant}. Free-form tag for grouping tasks (customer, project, team).` }, t.tenant)
              : null,
            progress
              ? h("span", {
                  className: cn(
                    "hermes-kanban-progress",
                    progress.done === progress.total ? "hermes-kanban-progress--full" : "",
                  ),
                  title: `${progress.done} of ${progress.total} child tasks done`,
                }, `${progress.done}/${progress.total}`)
              : null,
          ),
          h("div", { className: "hermes-kanban-card-title" },
            t.title || tx(i18n, "untitled", "(untitled)")),
          h("div", { className: "hermes-kanban-card-row hermes-kanban-card-meta" },
            t.assignee
              ? h("span", { className: "hermes-kanban-assignee",
                            title: `Assigned to Hermes profile @${t.assignee}` }, "@", t.assignee)
              : h("span", { className: "hermes-kanban-unassigned",
                            title: "No profile assigned. The dispatcher will pick one from available profiles when the task is Ready." },
                  tx(i18n, "unassigned", "unassigned")),
            t.comment_count > 0
              ? h("span", { className: "hermes-kanban-count",
                            title: `${t.comment_count} comment${t.comment_count === 1 ? "" : "s"} on this task` }, "💬 ", t.comment_count)
              : null,
            t.link_counts && (t.link_counts.parents + t.link_counts.children) > 0
              ? h("span", { className: "hermes-kanban-count",
                            title: `${t.link_counts.parents} parent${t.link_counts.parents === 1 ? "" : "s"}, ${t.link_counts.children} child${t.link_counts.children === 1 ? "" : "ren"}. Children stay blocked until their parent is done.` },
                  "↔ ", t.link_counts.parents + t.link_counts.children)
              : null,
            h("span", { className: "hermes-kanban-ago",
                        title: t.created_at ? `Created ${t.created_at}` : "" },
              timeAgo ? timeAgo(t.created_at) : ""),
          ),
        ),
      ),
    );
  }

  // -------------------------------------------------------------------------
  // Inline create (with parent selector)
  // -------------------------------------------------------------------------

  function InlineCreate(props) {
    const { t } = useI18n();
    const [title, setTitle] = useState("");
    const [assignee, setAssignee] = useState("");
    const [priority, setPriority] = useState(0);
    const [parent, setParent] = useState("");
    const [skills, setSkills] = useState("");
    // Workspace controls. `scratch` (default) ignores path; `worktree` optionally
    // takes a path (dispatcher derives one from the assignee profile otherwise);
    // `dir` requires a path. Backend enforces the rule — we only hide/show the
    // input here to save vertical space in the common `scratch` case.
    const [workspaceKind, setWorkspaceKind] = useState("scratch");
    const [workspacePath, setWorkspacePath] = useState("");

    const submit = function () {
      const trimmed = title.trim();
      if (!trimmed) return;
      const body = {
        title: trimmed,
        assignee: assignee.trim() || null,
        priority: Number(priority) || 0,
        triage: props.columnName === "triage",
      };
      if (parent) body.parents = [parent];
      // Parse comma-separated skills into a clean list. Blank = no
      // extras (omit key so backend leaves it null). The dispatcher
      // always auto-loads kanban-worker; these are extras on top.
      const skillList = skills
        .split(",")
        .map(function (s) { return s.trim(); })
        .filter(function (s) { return s.length > 0; });
      if (skillList.length > 0) body.skills = skillList;
      // Only send workspace_kind when it's non-default. Keeps the request
      // shape small and interoperable with older dispatcher versions.
      if (workspaceKind && workspaceKind !== "scratch") {
        body.workspace_kind = workspaceKind;
      }
      const wpTrim = workspacePath.trim();
      if (wpTrim) body.workspace_path = wpTrim;
      props.onSubmit(body);
      setTitle(""); setAssignee(""); setPriority(0); setParent(""); setSkills("");
      setWorkspaceKind("scratch"); setWorkspacePath("");
    };

    const showPathInput = workspaceKind !== "scratch";
    const pathPlaceholder = workspaceKind === "dir"
      ? tx(t, "workspacePathDir", "workspace path (required, e.g. ~/projects/my-app)")
      : tx(t, "workspacePathOptional",
          "workspace path (optional, derived from assignee if blank)");

    return h("div", { className: "hermes-kanban-inline-create" },
      h("textarea", {
        value: title,
        onChange: function (e) { setTitle(e.target.value); },
        onKeyDown: function (e) {
          if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
          if (e.key === "Escape") props.onCancel();
        },
        placeholder: props.columnName === "triage"
          ? tx(t, "triagePlaceholder", "Rough idea — AI will spec it…")
          : tx(t, "taskTitlePlaceholder", "New task title…"),
        autoFocus: true,
        className: "text-sm min-h-[2rem] max-h-32 resize-y w-full border border-input bg-transparent px-2 py-1 rounded-md focus:outline-none focus:ring-2 focus:ring-ring",
        rows: 2,
      }),
      h("div", { className: "flex gap-2" },
        h(Input, {
          value: assignee,
          onChange: function (e) { setAssignee(e.target.value); },
          placeholder: props.columnName === "triage"
            ? tx(t, "specifier", "specifier")
            : tx(t, "assigneePlaceholder", "assignee"),
          className: "h-7 text-xs flex-1",
          title: props.columnName === "triage"
            ? "Hermes profile that will spec this task (default: the dispatcher's configured specifier). Leave blank to let the dispatcher pick."
            : "Hermes profile to assign. Leave blank and the dispatcher will pick from available profiles when the task is Ready.",
          style: { textTransform: "none" },
          autoCapitalize: "none",
          autoCorrect: "off",
          spellCheck: false,
        }),
        h(Input, {
          type: "number",
          value: priority,
          onChange: function (e) { setPriority(e.target.value); },
          placeholder: "pri",
          className: "h-7 text-xs w-16",
          title: "Priority. Higher-priority tasks are claimed first by the dispatcher. 0 = default.",
        }),
      ),
      h(Input, {
        value: skills,
        onChange: function (e) { setSkills(e.target.value); },
        placeholder: tx(t, "skillsPlaceholder",
          "skills (optional, comma-separated): translation, github-code-review"),
        title: "Force-load these skills into the worker (in addition to the built-in kanban-worker).",
        className: "h-7 text-xs",
      }),
      h("div", { className: "flex gap-2" },
        h(Select, {
          value: workspaceKind,
          onChange: function (e) { setWorkspaceKind(e.target.value); },
          title: "scratch: isolated temp dir (default). worktree: git worktree on the assignee profile. dir: exact path (required below).",
          className: "h-7 text-xs w-28",
        },
          h(SelectOption, { value: "scratch" }, "scratch"),
          h(SelectOption, { value: "worktree" }, "worktree"),
          h(SelectOption, { value: "dir" }, "dir"),
        ),
        showPathInput ? h(Input, {
          value: workspacePath,
          onChange: function (e) { setWorkspacePath(e.target.value); },
          placeholder: pathPlaceholder,
          className: "h-7 text-xs flex-1",
        }) : null,
      ),
      h(Select, {
        value: parent,
        onChange: function (e) { setParent(e.target.value); },
        className: "h-7 text-xs",
        title: "Optional parent task. A child stays blocked in its current column until the parent is marked done.",
      },
        h(SelectOption, { value: "" }, tx(t, "noParent", "— no parent —")),
        (props.allTasks || []).map(function (task) {
          return h(SelectOption, { key: task.id, value: task.id },
            `${task.id} — ${(task.title || "").slice(0, 50)}`);
        }),
      ),
      h("div", { className: "flex gap-2" },
        h(Button, {
          onClick: submit,
          size: "sm",
        }, "Create"),
        h(Button, {
          onClick: props.onCancel,
          size: "sm",
        }, tx(t, "cancel", "Cancel")),
      ),
    );
  }

  // -------------------------------------------------------------------------
  // Task drawer
  // -------------------------------------------------------------------------

  function TaskDrawer(props) {
    const { t } = useI18n();
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [err, setErr] = useState(null);
    const [newComment, setNewComment] = useState("");
    const [editing, setEditing] = useState(false);
    // Home-channel notification toggles. homeChannels is the list of platforms
    // the user has a /sethome on; each entry has a `subscribed` bool telling
    // us whether this task is currently subscribed via that platform's home.
    const [homeChannels, setHomeChannels] = useState([]);
    const [homeBusy, setHomeBusy] = useState({});
    const boardSlug = props.boardSlug;

    const load = useCallback(function () {
      return SDK.fetchJSON(withBoard(`${API}/tasks/${encodeURIComponent(props.taskId)}`, boardSlug))
        .then(function (d) { setData(d); setErr(null); })
        .catch(function (e) { setErr(String(e.message || e)); })
        .finally(function () { setLoading(false); });
    }, [props.taskId, boardSlug]);

    const loadHomeChannels = useCallback(function () {
      const qs = new URLSearchParams({ task_id: props.taskId });
      const url = withBoard(`${API}/home-channels?${qs}`, boardSlug);
      return SDK.fetchJSON(url)
        .then(function (d) { setHomeChannels(d.home_channels || []); })
        .catch(function () { /* silent — endpoint optional on older gateways */ });
    }, [props.taskId, boardSlug]);

    // Reload when the WS stream reports new events for this task id
    // (completion, block, crash, etc. — anything that'd make the drawer
    // show stale data if we only loaded on mount).
    useEffect(function () { load(); }, [load, props.eventTick]);
    useEffect(function () { loadHomeChannels(); }, [loadHomeChannels]);
    useEffect(function () {
      function onKey(e) { if (e.key === "Escape" && !editing) props.onClose(); }
      window.addEventListener("keydown", onKey);
      return function () { window.removeEventListener("keydown", onKey); };
    }, [props.onClose, editing]);

    const handleComment = function () {
      const body = newComment.trim();
      if (!body) return;
      SDK.fetchJSON(withBoard(`${API}/tasks/${encodeURIComponent(props.taskId)}/comments`, boardSlug), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ body }),
      }).then(function () {
        setNewComment("");
        load();
        props.onRefresh();
      }).catch(function (e) { setErr(String(e.message || e)); });
    };

    const doPatch = function (patch, opts) {
      if (opts && opts.confirm && !window.confirm(opts.confirm)) {
        return Promise.resolve();
      }
      const finalPatch = withCompletionSummary(patch, 1);
      if (!finalPatch) return Promise.resolve();
      return SDK.fetchJSON(withBoard(`${API}/tasks/${encodeURIComponent(props.taskId)}`, boardSlug), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(finalPatch),
      }).then(function () { load(); props.onRefresh(); });
    };

    // Triage specifier — calls the auxiliary LLM to flesh out a rough
    // idea in the Triage column into a concrete spec (title + body with
    // goal, approach, acceptance criteria) and promotes it to todo.
    // Not a PATCH: runs through a dedicated POST endpoint because the
    // LLM call can take tens of seconds, and its outcome is richer than
    // a status flip (may update title AND body AND emit an audit
    // comment — or fail with a human-readable reason that the UI
    // surfaces inline without treating it as an HTTP error).
    const doSpecify = function () {
      return SDK.fetchJSON(
        withBoard(`${API}/tasks/${encodeURIComponent(props.taskId)}/specify`, boardSlug),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        }
      ).then(function (res) {
        load();
        props.onRefresh();
        return res;
      });
    };

    const addLink = function (parentId) {
      return SDK.fetchJSON(withBoard(`${API}/links`, boardSlug), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ parent_id: parentId, child_id: props.taskId }),
      }).then(function () { load(); props.onRefresh(); })
        .catch(function (e) { setErr(String(e.message || e)); });
    };
    const removeLink = function (parentId) {
      const qs = new URLSearchParams({ parent_id: parentId, child_id: props.taskId });
      return SDK.fetchJSON(withBoard(`${API}/links?${qs}`, boardSlug), { method: "DELETE" })
        .then(function () { load(); props.onRefresh(); })
        .catch(function (e) { setErr(String(e.message || e)); });
    };
    const addChild = function (childId) {
      return SDK.fetchJSON(withBoard(`${API}/links`, boardSlug), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ parent_id: props.taskId, child_id: childId }),
      }).then(function () { load(); props.onRefresh(); })
        .catch(function (e) { setErr(String(e.message || e)); });
    };
    const removeChild = function (childId) {
      const qs = new URLSearchParams({ parent_id: props.taskId, child_id: childId });
      return SDK.fetchJSON(withBoard(`${API}/links?${qs}`, boardSlug), { method: "DELETE" })
        .then(function () { load(); props.onRefresh(); })
        .catch(function (e) { setErr(String(e.message || e)); });
    };

    const toggleHomeSubscription = function (platform, currentlySubscribed) {
      // Optimistic flip + busy flag to keep double-clicks idempotent.
      setHomeBusy(function (b) { return Object.assign({}, b, { [platform]: true }); });
      setHomeChannels(function (list) {
        return list.map(function (h) {
          return h.platform === platform
            ? Object.assign({}, h, { subscribed: !currentlySubscribed })
            : h;
        });
      });
      const method = currentlySubscribed ? "DELETE" : "POST";
      const url = withBoard(
        `${API}/tasks/${encodeURIComponent(props.taskId)}/home-subscribe/${encodeURIComponent(platform)}`,
        boardSlug,
      );
      return SDK.fetchJSON(url, { method: method })
        .then(function () { return loadHomeChannels(); })
        .catch(function (e) {
          // Revert optimistic flip on failure.
          setHomeChannels(function (list) {
            return list.map(function (h) {
              return h.platform === platform
                ? Object.assign({}, h, { subscribed: currentlySubscribed })
                : h;
            });
          });
          setErr(String(e.message || e));
        })
        .finally(function () {
          setHomeBusy(function (b) {
            const next = Object.assign({}, b);
            delete next[platform];
            return next;
          });
        });
    };

    return h("div", { className: "hermes-kanban-drawer-shade", onClick: props.onClose },
      h("div", {
        className: "hermes-kanban-drawer",
        onClick: function (e) { e.stopPropagation(); },
      },
        h("div", { className: "hermes-kanban-drawer-head" },
          h("span", { className: "text-xs text-muted-foreground" }, props.taskId),
          h("button", {
            type: "button",
            onClick: props.onClose,
            className: "hermes-kanban-drawer-close",
            title: tx(t, "close", "Close (Esc)"),
          }, "×"),
        ),
        loading ? h("div", { className: "p-4 text-sm text-muted-foreground" },
          tx(t, "loadingDetail", "Loading…")) :
        err ? h("div", { className: "p-4 text-sm text-destructive" }, err) :
        data ? h(TaskDetail, {
          data, editing, setEditing,
          renderMarkdown: props.renderMarkdown,
          allTasks: props.allTasks,
          assignees: props.assignees || [],
          boardSlug: boardSlug,
          onPatch: doPatch,
          onSpecify: doSpecify,
          onAddParent: addLink,
          onRemoveParent: removeLink,
          onAddChild: addChild,
          onRemoveChild: removeChild,
          homeChannels: homeChannels,
          homeBusy: homeBusy,
          onToggleHomeSub: toggleHomeSubscription,
          onRefresh: props.onRefresh,
        }) : null,
        data ? h("div", { className: "hermes-kanban-drawer-comment-row" },
          h(Input, {
            value: newComment,
            onChange: function (e) { setNewComment(e.target.value); },
            onKeyDown: function (e) {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault(); handleComment();
              }
            },
            placeholder: tx(t, "addComment", "Add a comment… (Enter to submit)"),
            className: "h-8 text-sm flex-1",
          }),
          h(Button, {
            onClick: handleComment,
            size: "sm",
          }, tx(t, "comment", "Comment")),
        ) : null,
      ),
    );
  }

  function TaskDetail(props) {
    const { t: i18n } = useI18n();
    const t = props.data.task;
    const comments = props.data.comments || [];
    const events = props.data.events || [];
    const links = props.data.links || { parents: [], children: [] };

    return h("div", { className: "hermes-kanban-drawer-body" },
      h("div", { className: "hermes-kanban-drawer-title" },
        h("span", { className: cn("hermes-kanban-dot", COLUMN_DOT[t.status]) }),
        props.editing
          ? h(TitleEditor, {
              initial: t.title || "",
              onSave: function (newTitle) {
                return props.onPatch({ title: newTitle }).then(function () { props.setEditing(false); });
              },
              onCancel: function () { props.setEditing(false); },
            })
          : h("span", {
              className: "hermes-kanban-drawer-title-text",
              title: tx(i18n, "clickToEdit", "Click to edit"),
              onClick: function () { props.setEditing(true); },
            }, t.title || tx(i18n, "untitled", "(untitled)")),
      ),
      h("div", { className: "hermes-kanban-drawer-meta" },
        h(MetaRow, { label: tx(i18n, "status", "Status"), value: t.status }),
        h(AssigneeEditor, { task: t, onPatch: props.onPatch }),
        h(PriorityEditor, { task: t, onPatch: props.onPatch }),
        t.tenant ? h(MetaRow, { label: tx(i18n, "tenant", "Tenant"), value: t.tenant }) : null,
        h(MetaRow, {
          label: tx(i18n, "workspace", "Workspace"),
          value: `${t.workspace_kind}${t.workspace_path ? ": " + t.workspace_path : ""}`,
        }),
        (t.skills && t.skills.length > 0) ? h(MetaRow, {
          label: tx(i18n, "skills", "Skills"),
          value: t.skills.join(", "),
        }) : null,
        t.created_by ? h(MetaRow, { label: tx(i18n, "createdBy", "Created by"), value: t.created_by }) : null,
      ),
      h(StatusActions, {
        task: t,
        onPatch: props.onPatch,
        onSpecify: props.onSpecify,
      }),
      h(DiagnosticsSection, {
        task: t,
        boardSlug: props.boardSlug,
        assignees: props.assignees,
        diagnostics: t.diagnostics || [],
        onRefresh: props.onRefresh,
      }),
      h(HomeSubsSection, {
        homeChannels: props.homeChannels || [],
        homeBusy: props.homeBusy || {},
        onToggle: props.onToggleHomeSub,
      }),
      h(BodyEditor, {
        task: t,
        renderMarkdown: props.renderMarkdown,
        onPatch: props.onPatch,
      }),
      h(DependencyEditor, {
        task: t,
        links, allTasks: props.allTasks,
        onAddParent: props.onAddParent,
        onRemoveParent: props.onRemoveParent,
        onAddChild: props.onAddChild,
        onRemoveChild: props.onRemoveChild,
      }),
      t.result ? h("div", { className: "hermes-kanban-section" },
        h("div", { className: "hermes-kanban-section-head" }, tx(i18n, "result", "Result")),
        h(MarkdownBlock, { source: t.result, enabled: props.renderMarkdown }),
      ) : null,
      h("div", { className: "hermes-kanban-section" },
        h("div", { className: "hermes-kanban-section-head" },
          `${tx(i18n, "comments", "Comments")} (${comments.length})`),
        comments.length === 0
          ? h("div", { className: "text-xs text-muted-foreground" },
              tx(i18n, "noComments", "— no comments —"))
          : comments.map(function (c) {
              return h("div", { key: c.id, className: "hermes-kanban-comment" },
                h("div", { className: "hermes-kanban-comment-head" },
                  h("span", { className: "hermes-kanban-comment-author" }, c.author || "anon"),
                  h("span", { className: "hermes-kanban-comment-ago" },
                    timeAgo ? timeAgo(c.created_at) : ""),
                ),
                h(MarkdownBlock, { source: c.body, enabled: props.renderMarkdown }),
              );
            }),
      ),
      h("div", { className: "hermes-kanban-section" },
        h("div", { className: "hermes-kanban-section-head" },
          `${tx(i18n, "events", "Events")} (${events.length})`),
        events.slice().reverse().slice(0, 20).map(function (e) {
          const isDiag = isDiagnosticEvent(e.kind);
          const phantoms = isDiag ? phantomIdsFromEvent(e) : [];
          return h("div", {
            key: e.id,
            className: cn(
              "hermes-kanban-event",
              isDiag ? "hermes-kanban-event--hallucination" : "",
            ),
          },
            isDiag
              ? h("div", { className: "hermes-kanban-event-header" },
                  h("span", { className: "hermes-kanban-event-warning-icon" }, "⚠"),
                  h("span", { className: "hermes-kanban-event-warning-label" },
                    getDiagnosticEventLabel(i18n, e.kind) || e.kind),
                  h("span", { className: "hermes-kanban-event-ago" },
                    timeAgo ? timeAgo(e.created_at) : ""),
                )
              : h("div", { className: "hermes-kanban-event-header-plain" },
                  h("span", { className: "hermes-kanban-event-kind" }, e.kind),
                  h("span", { className: "hermes-kanban-event-ago" },
                    timeAgo ? timeAgo(e.created_at) : ""),
                ),
            isDiag && phantoms.length > 0
              ? h("div", { className: "hermes-kanban-event-phantom-row" },
                  h("span", { className: "hermes-kanban-event-phantom-label" },
                    tx(i18n, "phantomIds", "Phantom ids:")),
                  phantoms.map(function (pid) {
                    return h("code", {
                      key: pid,
                      className: "hermes-kanban-event-phantom-chip",
                    }, pid);
                  }),
                )
              : null,
            e.payload && !isDiag
              ? h("code", { className: "hermes-kanban-event-payload" },
                  JSON.stringify(e.payload))
              : null,
          );
        }),
      ),
      h(WorkerLogSection, { taskId: t.id, boardSlug: props.boardSlug }),
      h(RunHistorySection, { runs: props.data.runs || [] }),
    );
  }

  // Per-attempt history. Closed runs first (most recent last), then the
  // active run if any. Each row shows profile / outcome / elapsed /
  // summary. Collapsed by default when there are more than three runs.
  function RunHistorySection(props) {
    const { t } = useI18n();
    const runs = props.runs || [];
    const [expanded, setExpanded] = useState(false);
    if (runs.length === 0) return null;
    const showAll = expanded || runs.length <= 3;
    const visible = showAll ? runs : runs.slice(-3);

    const fmtElapsed = function (run) {
      if (!run || !run.started_at) return "";
      const end = run.ended_at || Math.floor(Date.now() / 1000);
      const secs = Math.max(0, end - run.started_at);
      if (secs < 60) return `${secs}s`;
      if (secs < 3600) return `${Math.round(secs / 60)}m`;
      return `${(secs / 3600).toFixed(1)}h`;
    };

    return h("div", { className: "hermes-kanban-section" },
      h("div", { className: "hermes-kanban-section-head-row" },
        h("span", { className: "hermes-kanban-section-head" },
          `${tx(t, "runHistory", "Run history")} (${runs.length})`),
        !showAll
          ? h("button", {
              type: "button",
              onClick: function () { setExpanded(true); },
              className: "hermes-kanban-edit-link",
              title: tx(t, "showAllAttempts", "Show all attempts"),
            }, `+${runs.length - 3} earlier`)
          : null,
      ),
      visible.map(function (r) {
        const outcomeClass = r.ended_at
          ? `hermes-kanban-run--${r.outcome || r.status || "ended"}`
          : "hermes-kanban-run--active";
        return h("div", { key: r.id, className: cn("hermes-kanban-run", outcomeClass) },
          h("div", { className: "hermes-kanban-run-head" },
            h("span", { className: "hermes-kanban-run-outcome" },
              r.ended_at ? (r.outcome || r.status || tx(t, "ended", "ended")) : tx(t, "active", "active")),
            h("span", { className: "hermes-kanban-run-profile" },
              r.profile ? `@${r.profile}` : tx(t, "noProfile", "(no profile)")),
            h("span", { className: "hermes-kanban-run-elapsed" }, fmtElapsed(r)),
            h("span", { className: "hermes-kanban-run-ago" },
              timeAgo ? timeAgo(r.started_at) : ""),
          ),
          r.summary
            ? h("div", { className: "hermes-kanban-run-summary" }, r.summary)
            : null,
          r.error
            ? h("div", { className: "hermes-kanban-run-error" }, r.error)
            : null,
          (r.metadata && Object.keys(r.metadata).length > 0)
            ? (function () {
                var json = JSON.stringify(r.metadata, null, 2);
                var collapsed = json.length > 300;
                return h("details", {
                    className: "hermes-kanban-run-meta-block",
                    open: !collapsed,
                  },
                  h("summary", { className: "hermes-kanban-run-meta-label" }, "Metadata"),
                  h("code", { className: "hermes-kanban-run-meta" }, json),
                );
              })()
            : null,
        );
      }),
    );
  }

  // Worker log: loads lazily (one GET on mount), refresh button, tail cap.
  function WorkerLogSection(props) {
    const { t } = useI18n();
    const [state, setState] = useState({ loading: false, data: null, err: null });
    const load = useCallback(function () {
      setState({ loading: true, data: null, err: null });
      SDK.fetchJSON(withBoard(`${API}/tasks/${encodeURIComponent(props.taskId)}/log?tail=100000`, props.boardSlug))
        .then(function (d) { setState({ loading: false, data: d, err: null }); })
        .catch(function (e) { setState({ loading: false, data: null, err: String(e.message || e) }); });
    }, [props.taskId, props.boardSlug]);

    // Auto-load when the section mounts; the user opened the drawer so the
    // cost is one small HTTP round-trip.
    useEffect(function () { load(); }, [load]);

    const data = state.data;
    let body;
    if (state.loading) {
      body = h("div", { className: "text-xs text-muted-foreground" },
        tx(t, "loadingLog", "Loading log…"));
    } else if (state.err) {
      body = h("div", { className: "text-xs text-destructive" }, state.err);
    } else if (!data || !data.exists) {
      body = h("div", { className: "text-xs text-muted-foreground italic" },
        tx(t, "noWorkerLog",
          "— no worker log yet (task hasn't spawned or log was rotated away) —"));
    } else {
      body = h("pre", { className: "hermes-kanban-pre hermes-kanban-log" },
        data.content || "(empty)");
    }

    return h("div", { className: "hermes-kanban-section" },
      h("div", { className: "hermes-kanban-section-head-row" },
        h("span", { className: "hermes-kanban-section-head" },
          tx(t, "workerLog", "Worker log") + (data && data.size_bytes ? ` (${data.size_bytes} B)` : "")),
        h("button", {
          type: "button",
          onClick: load,
          className: "hermes-kanban-edit-link",
          title: "Refresh log",
        }, "refresh"),
      ),
      body,
      data && data.truncated
        ? h("div", { className: "text-xs text-muted-foreground" },
            tx(t, "logTruncated", "(showing last 100 KB — full log at "),
            data.path,
            tx(t, "logAt", ")"))
        : null,
    );
  }

  function MetaRow(props) {
    return h("div", { className: "hermes-kanban-meta-row" },
      h("span", { className: "hermes-kanban-meta-label" }, props.label),
      h("span", { className: "hermes-kanban-meta-value" }, props.value),
    );
  }

  function TitleEditor(props) {
    const { t } = useI18n();
    const [v, setV] = useState(props.initial);
    const save = function () {
      const trimmed = v.trim();
      if (!trimmed) return;
      props.onSave(trimmed);
    };
    return h("div", { className: "hermes-kanban-edit-row" },
      h(Input, {
        value: v, autoFocus: true,
        onChange: function (e) { setV(e.target.value); },
        onKeyDown: function (e) {
          if (e.key === "Enter") { e.preventDefault(); save(); }
          if (e.key === "Escape") props.onCancel();
        },
        className: "h-8 text-sm flex-1",
      }),
      h(Button, { onClick: save,
        size: "sm",
      }, tx(t, "save", "Save")),
      h(Button, { onClick: props.onCancel,
        size: "sm",
      }, tx(t, "cancel", "Cancel")),
    );
  }

  function AssigneeEditor(props) {
    const { t } = useI18n();
    const [editing, setEditing] = useState(false);
    const [v, setV] = useState(props.task.assignee || "");
    useEffect(function () { setV(props.task.assignee || ""); }, [props.task.assignee]);
    if (!editing) {
      return h("div", { className: "hermes-kanban-meta-row" },
        h("span", { className: "hermes-kanban-meta-label" }, tx(t, "assignee", "Assignee")),
        h("span", {
          className: "hermes-kanban-meta-value hermes-kanban-editable",
          onClick: function () { setEditing(true); },
          title: tx(t, "clickToEditAssignee", "Click to edit assignee"),
        }, props.task.assignee || tx(t, "unassigned", "unassigned")),
      );
    }
    const save = function () {
      props.onPatch({ assignee: v.trim() || "" }).then(function () { setEditing(false); });
    };
    return h("div", { className: "hermes-kanban-meta-row" },
      h("span", { className: "hermes-kanban-meta-label" }, tx(t, "assignee", "Assignee")),
      h(Input, {
        value: v, autoFocus: true,
        onChange: function (e) { setV(e.target.value); },
        onKeyDown: function (e) {
          if (e.key === "Enter") { e.preventDefault(); save(); }
          if (e.key === "Escape") setEditing(false);
        },
        placeholder: tx(t, "emptyAssignee", "(empty = unassign)"),
        className: "h-7 text-xs flex-1",
        style: { textTransform: "none" },
        autoCapitalize: "none",
        autoCorrect: "off",
        spellCheck: false,
      }),
    );
  }

  function PriorityEditor(props) {
    const { t } = useI18n();
    const [editing, setEditing] = useState(false);
    const [v, setV] = useState(String(props.task.priority || 0));
    useEffect(function () { setV(String(props.task.priority || 0)); }, [props.task.priority]);
    if (!editing) {
      return h("div", { className: "hermes-kanban-meta-row" },
        h("span", { className: "hermes-kanban-meta-label" }, tx(t, "priority", "Priority")),
        h("span", {
          className: "hermes-kanban-meta-value hermes-kanban-editable",
          onClick: function () { setEditing(true); },
          title: tx(t, "clickToEdit", "Click to edit"),
        }, String(props.task.priority)),
      );
    }
    const save = function () {
      props.onPatch({ priority: Number(v) || 0 }).then(function () { setEditing(false); });
    };
    return h("div", { className: "hermes-kanban-meta-row" },
      h("span", { className: "hermes-kanban-meta-label" }, tx(t, "priority", "Priority")),
      h(Input, {
        type: "number", value: v, autoFocus: true,
        onChange: function (e) { setV(e.target.value); },
        onKeyDown: function (e) {
          if (e.key === "Enter") { e.preventDefault(); save(); }
          if (e.key === "Escape") setEditing(false);
        },
        className: "h-7 text-xs w-20",
      }),
    );
  }

  function BodyEditor(props) {
    const { t } = useI18n();
    const [editing, setEditing] = useState(false);
    const [v, setV] = useState(props.task.body || "");
    useEffect(function () { setV(props.task.body || ""); }, [props.task.body]);
    const save = function () {
      props.onPatch({ body: v }).then(function () { setEditing(false); });
    };
    return h("div", { className: "hermes-kanban-section" },
      h("div", { className: "hermes-kanban-section-head-row" },
        h("span", { className: "hermes-kanban-section-head" }, tx(t, "description", "Description")),
        editing
          ? h("div", { className: "flex gap-1" },
              h(Button, { onClick: save,
                size: "sm",
              }, tx(t, "save", "Save")),
              h(Button, { onClick: function () { setEditing(false); setV(props.task.body || ""); },
                size: "sm",
              }, tx(t, "cancel", "Cancel")),
            )
          : h("button", {
              type: "button",
              onClick: function () { setEditing(true); },
              className: "hermes-kanban-edit-link",
              title: "Edit description",
            }, tx(t, "edit", "edit")),
      ),
      editing
        ? h("textarea", {
            className: "hermes-kanban-textarea",
            value: v,
            rows: 8,
            onChange: function (e) { setV(e.target.value); },
          })
        : props.task.body
          ? h(MarkdownBlock, { source: props.task.body, enabled: props.renderMarkdown })
          : h("div", { className: "text-xs text-muted-foreground italic" },
              tx(t, "noDescription", "— no description —")),
    );
  }

  function DependencyEditor(props) {
    const { t } = useI18n();
    const { task, links, allTasks } = props;
    const [newParent, setNewParent] = useState("");
    const [newChild, setNewChild] = useState("");
    // Filter out self + existing links when offering the "add" dropdown.
    const candidatesFor = function (excludeSet) {
      return (allTasks || []).filter(function (tk) {
        return tk.id !== task.id && !excludeSet.has(tk.id);
      });
    };
    const parentExclude = new Set([task.id, ...(links.parents || [])]);
    const childExclude  = new Set([task.id, ...(links.children || [])]);

    return h("div", { className: "hermes-kanban-section" },
      h("div", { className: "hermes-kanban-section-head" }, tx(t, "dependencies", "Dependencies")),
      h("div", { className: "hermes-kanban-deps-row" },
        h("span", { className: "hermes-kanban-deps-label" }, tx(t, "parents", "Parents:")),
        h("div", { className: "hermes-kanban-deps-chips" },
          (links.parents || []).length === 0
            ? h("span", { className: "hermes-kanban-deps-empty" }, tx(t, "none", "none"))
            : (links.parents || []).map(function (id) {
                return h("span", { key: id, className: "hermes-kanban-dep-chip" },
                  id,
                  h("button", {
                    type: "button",
                    className: "hermes-kanban-dep-chip-x",
                    onClick: function () { props.onRemoveParent(id); },
                    title: tx(t, "removeDependency", "Remove dependency"),
                  }, "×"),
                );
              }),
        ),
      ),
      h("div", { className: "hermes-kanban-deps-row" },
        h(Select, Object.assign({
          value: newParent,
          className: "h-7 text-xs flex-1",
        }, selectChangeHandler(setNewParent)),
          h(SelectOption, { value: "" }, tx(t, "addParent", "— add parent —")),
          candidatesFor(parentExclude).map(function (tk) {
            return h(SelectOption, { key: tk.id, value: tk.id },
              `${tk.id} — ${(tk.title || "").slice(0, 50)}`);
          }),
        ),
        h(Button, {
          onClick: function () {
            if (!newParent) return;
            props.onAddParent(newParent).then(function () { setNewParent(""); });
          },
          disabled: !newParent,
          size: "sm",
        }, "+ parent"),
      ),
      h("div", { className: "hermes-kanban-deps-row" },
        h("span", { className: "hermes-kanban-deps-label" }, tx(t, "children", "Children:")),
        h("div", { className: "hermes-kanban-deps-chips" },
          (links.children || []).length === 0
            ? h("span", { className: "hermes-kanban-deps-empty" }, tx(t, "none", "none"))
            : (links.children || []).map(function (id) {
                return h("span", { key: id, className: "hermes-kanban-dep-chip" },
                  id,
                  h("button", {
                    type: "button",
                    className: "hermes-kanban-dep-chip-x",
                    onClick: function () { props.onRemoveChild(id); },
                    title: tx(t, "removeDependency", "Remove dependency"),
                  }, "×"),
                );
              }),
        ),
      ),
      h("div", { className: "hermes-kanban-deps-row" },
        h(Select, Object.assign({
          value: newChild,
          className: "h-7 text-xs flex-1",
        }, selectChangeHandler(setNewChild)),
          h(SelectOption, { value: "" }, tx(t, "addChild", "— add child —")),
          candidatesFor(childExclude).map(function (tk) {
            return h(SelectOption, { key: tk.id, value: tk.id },
              `${tk.id} — ${(tk.title || "").slice(0, 50)}`);
          }),
        ),
        h(Button, {
          onClick: function () {
            if (!newChild) return;
            props.onAddChild(newChild).then(function () { setNewChild(""); });
          },
          disabled: !newChild,
          size: "sm",
        }, "+ child"),
      ),
    );
  }

  function StatusActions(props) {
    const { t } = useI18n();
    const task = props.task;
    const [specifyBusy, setSpecifyBusy] = useState(false);
    const [specifyMsg, setSpecifyMsg] = useState(null);
    const b = function (label, patch, enabled, confirmMsg) {
      return h(Button, {
        onClick: function () { if (enabled !== false) props.onPatch(patch, { confirm: confirmMsg }); },
        disabled: enabled === false,
        size: "sm",
      }, label);
    };

    // "Specify" appears only when the task is in the Triage column — the
    // one column where an auxiliary LLM pass is meaningful. Elsewhere
    // the backend would return ok:false with "not in triage" anyway,
    // so hiding the button keeps the action row uncluttered.
    const specifyButton = (task.status === "triage" && props.onSpecify)
      ? h(Button, {
          onClick: function () {
            if (specifyBusy) return;
            setSpecifyBusy(true);
            setSpecifyMsg(null);
            props.onSpecify().then(function (res) {
              if (res && res.ok) {
                const suffix = res.new_title
                  ? ` — retitled: ${res.new_title}`
                  : "";
                setSpecifyMsg({ ok: true, text: `Specified${suffix}` });
              } else {
                setSpecifyMsg({
                  ok: false,
                  text: "Specify failed: " + ((res && res.reason) || "unknown error"),
                });
              }
            }).catch(function (err) {
              setSpecifyMsg({
                ok: false,
                text: "Specify failed: " + (err.message || String(err)),
              });
            }).then(function () {
              setSpecifyBusy(false);
            });
          },
          disabled: specifyBusy,
          size: "sm",
        }, specifyBusy ? "Specifying…" : "✨ Specify")
      : null;

    return h("div", null,
      h("div", { className: "hermes-kanban-actions" },
        specifyButton,
        b("→ triage",  { status: "triage" },   task.status !== "triage"),
        b("→ ready",   { status: "ready" },    task.status !== "ready"),
        // No direct → running button: /tasks/:id PATCH rejects status=running
        // with 400 (issue #19535). Tasks enter running only through the
        // dispatcher's claim_task path, which atomically creates the run row,
        // claim lock, and worker process metadata.
        b(tx(t, "block", "Block"),     { status: "blocked" },
          task.status === "running" || task.status === "ready",
          getDestructiveConfirm(t, "blocked")),
        b(tx(t, "unblock", "Unblock"),   { status: "ready" },    task.status === "blocked"),
        b(tx(t, "complete", "Complete"),  { status: "done" },
          task.status === "running" || task.status === "ready" || task.status === "blocked",
          getDestructiveConfirm(t, "done")),
        b(tx(t, "archive", "Archive"),   { status: "archived" }, task.status !== "archived",
          getDestructiveConfirm(t, "archived")),
      ),
      specifyMsg ? h("div", {
        className: specifyMsg.ok
          ? "hermes-kanban-msg-ok"
          : "hermes-kanban-msg-err",
      }, specifyMsg.text) : null,
    );
  }


  // One toggle per gateway platform the user has a home channel set on
  // (telegram, discord, slack, etc.). Toggling on creates a kanban_notify_subs
  // row routed to that platform's home; toggling off removes it. Nothing
  // renders when no platforms have a home configured — this section stays
  // invisible for users who haven't set one up.
  function HomeSubsSection(props) {
    const { t } = useI18n();
    const channels = props.homeChannels || [];
    if (channels.length === 0) return null;
    const busy = props.homeBusy || {};
    return h("div", { className: "hermes-kanban-section" },
      h("div", { className: "hermes-kanban-section-head" },
        tx(t, "notifyHomeChannels", "Notify home channels")),
      h("div", { className: "hermes-kanban-home-subs" },
        channels.map(function (hc) {
          const isBusy = !!busy[hc.platform];
          const label = hc.subscribed ? "✓ " + hc.platform : hc.platform;
          const target = `${hc.name} (${hc.chat_id}${hc.thread_id ? " / " + hc.thread_id : ""})`;
          const title = hc.subscribed
            ? `${tx(t, "sendingUpdates", "Sending updates to")} ${target}. Click to stop.`
            : `${tx(t, "sendNotifications", "Send completed / blocked / gave_up notifications to")} ${target}.`;
          return h(Button, {
            key: hc.platform,
            size: "sm",
            title: title,
            disabled: isBusy || !props.onToggle,
            onClick: function () {
              if (props.onToggle) props.onToggle(hc.platform, hc.subscribed);
            },
            className: hc.subscribed
              ? "hermes-kanban-home-sub hermes-kanban-home-sub--on"
              : "hermes-kanban-home-sub",
          }, label);
        })
      )
    );
  }

  // -------------------------------------------------------------------------
  // Register
  // -------------------------------------------------------------------------

  if (window.__HERMES_PLUGINS__ && typeof window.__HERMES_PLUGINS__.register === "function") {
    window.__HERMES_PLUGINS__.register("kanban", KanbanPage);
  }
})();

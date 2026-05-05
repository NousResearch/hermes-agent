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

  // Order matches BOARD_COLUMNS in plugin_api.py.
  const COLUMN_ORDER = ["triage", "todo", "ready", "running", "blocked", "done"];
  const COLUMN_LABEL = {
    triage: "Triage",
    todo: "Todo",
    ready: "Ready",
    running: "In Progress",
    blocked: "Blocked",
    done: "Done",
    archived: "Archived",
  };
  const COLUMN_HELP = {
    triage: "Raw ideas — a specifier will flesh out the spec",
    todo: "Waiting on dependencies or unassigned",
    ready: "Assigned and waiting for a dispatcher tick",
    running: "Claimed by a worker — in-flight",
    blocked: "Worker asked for human input",
    done: "Completed",
    archived: "Archived",
  };
  const COLUMN_DOT = {
    triage: "hermes-kanban-dot-triage",
    todo: "hermes-kanban-dot-todo",
    ready: "hermes-kanban-dot-ready",
    running: "hermes-kanban-dot-running",
    blocked: "hermes-kanban-dot-blocked",
    done: "hermes-kanban-dot-done",
    archived: "hermes-kanban-dot-archived",
  };

  const DESTRUCTIVE_TRANSITIONS = {
    done: "Mark this task as done? The worker's claim is released and dependent children become ready.",
    archived: "Archive this task? It disappears from the default board view.",
    blocked: "Mark this task as blocked? The worker's claim is released.",
  };

  function withCompletionSummary(patch, count) {
    if (!patch || patch.status !== "done") return patch;
    const label = count && count > 1 ? `${count} selected task(s)` : "this task";
    const value = window.prompt(
      `Completion summary for ${label}. This is stored as the task result.`,
      "",
    );
    if (value === null) return null;
    const summary = value.trim();
    if (!summary) {
      window.alert("Completion summary is required before marking a task done.");
      return null;
    }
    return Object.assign({}, patch, { result: summary, summary });
  }

  const API = "/api/plugins/kanban";
  const MIME_TASK = "text/x-hermes-task";

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
      if (slug && slug !== "default") window.localStorage.setItem(LS_BOARD_KEY, slug);
      else window.localStorage.removeItem(LS_BOARD_KEY);
    } catch (_e) { /* ignore quota / private mode */ }
  }

  function withBoard(url, board) {
    // Append ?board=<slug> when a non-default board is active. Omitted
    // for default so the URL stays clean and the backend falls through
    // to its own resolution chain (env var → ``current`` file →
    // default) which is already correct.
    if (!board || board === "default") return url;
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

  class ErrorBoundary extends React.Component {
    constructor(props) { super(props); this.state = { error: null }; }
    static getDerivedStateFromError(error) { return { error }; }
    componentDidCatch(error, info) {
      // eslint-disable-next-line no-console
      console.error("Kanban plugin crashed:", error, info);
    }
    render() {
      if (this.state.error) {
        return h(Card, null,
          h(CardContent, { className: "p-6 text-sm" },
            h("div", { className: "text-destructive font-semibold mb-1" },
              "Kanban tab hit a rendering error"),
            h("div", { className: "text-muted-foreground text-xs mb-3" },
              String(this.state.error && this.state.error.message || this.state.error)),
            h(Button, {
              onClick: () => this.setState({ error: null }),
              size: "sm",
            }, "Reload view"),
          ),
        );
      }
      return this.props.children;
    }
  }

  // -------------------------------------------------------------------------
  // Root page
  // -------------------------------------------------------------------------

  function KanbanPage() {
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
      SDK.fetchJSON(`${API}/config`)
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
      return SDK.fetchJSON(`${API}/boards`)
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
        // from other boards don't bleed in. Only set for non-default so
        // single-board installs keep the cleaner URL.
        if (board && board !== "default") qsParams.board = board;
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
            setError("WebSocket auth failed — reload the page to refresh the session token.");
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
        if (assigneeFilter && t.assignee !== assigneeFilter) return false;
        if (q) {
          const hay = `${t.id} ${t.title || ""} ${t.assignee || ""} ${t.tenant || ""}`.toLowerCase();
          if (hay.indexOf(q) === -1) return false;
        }
        return true;
      };
      return Object.assign({}, boardData, {
        columns: boardData.columns.map(function (col) {
          return Object.assign({}, col, { tasks: col.tasks.filter(filterTask) });
        }),
      });
    }, [boardData, assigneeFilter, search]);

    // --- actions ------------------------------------------------------------
    const moveTask = useCallback(function (taskId, newStatus) {
      const confirmMsg = DESTRUCTIVE_TRANSITIONS[newStatus];
      if (confirmMsg && !window.confirm(confirmMsg)) return;
      const patch = withCompletionSummary({ status: newStatus }, 1);
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
        setError(`Move failed: ${err.message || err}`);
        loadBoard();
      });
    }, [loadBoard, board]);

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
          setError("Task created, but: " + res.warning);
        }
        loadBoard();
        loadBoardList();  // refresh counts in the switcher
        return res;
      });
    }, [loadBoard, loadBoardList, board]);

    const toggleSelected = useCallback(function (id, additive) {
      setSelectedIds(function (prev) {
        const next = new Set(additive ? prev : []);
        if (prev.has(id)) next.delete(id);
        else next.add(id);
        return next;
      });
    }, []);
    const clearSelected = useCallback(function () { setSelectedIds(new Set()); }, []);

    const applyBulk = useCallback(function (patch, confirmMsg) {
      if (selectedIds.size === 0) return;
      if (confirmMsg && !window.confirm(confirmMsg)) return;
      const finalPatch = withCompletionSummary(patch, selectedIds.size);
      if (!finalPatch) return;
      const body = Object.assign({ ids: Array.from(selectedIds) }, finalPatch);
      SDK.fetchJSON(withBoard(`${API}/tasks/bulk`, board), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
        .then(function (res) {
          const failed = (res.results || []).filter(function (r) { return !r.ok; });
          if (failed.length > 0) {
            setError(`Bulk: ${failed.length} of ${res.results.length} failed: ` +
              failed.slice(0, 3).map(function (f) { return `${f.id} (${f.error})`; }).join("; "));
          }
          clearSelected();
          loadBoard();
        })
        .catch(function (e) { setError(String(e.message || e)); });
    }, [selectedIds, loadBoard, clearSelected, board]);

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
    }, [board]);

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
    }, [loadBoardList, switchBoard]);

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
        "Loading Kanban board…");
    }
    if (error && !boardData) {
      return h(Card, null,
        h(CardContent, { className: "p-6" },
          h("div", { className: "text-sm text-destructive" },
            "Failed to load Kanban board: ", error),
          h("div", { className: "text-xs text-muted-foreground mt-2" },
            "The backend auto-creates kanban.db on first read. If this persists, check the dashboard logs."),
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
        }) : null,
        error ? h("div", { className: "text-xs text-destructive px-2" }, error) : null,
        h(BoardColumns, {
          board: filteredBoard,
          laneByProfile,
          selectedIds,
          toggleSelected,
          onMove: moveTask,
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
          eventTick: taskEventTick[selectedTaskId] || 0,
        }) : null,
      ),
    );
  }

  // -------------------------------------------------------------------------
  // Board switcher (multi-project)
  // -------------------------------------------------------------------------

  function BoardSwitcher(props) {
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
        title: "Boards let you separate unrelated streams of work",
      },
        h(Button, {
          onClick: props.onNewClick,
          size: "sm",
          className: "h-7 text-xs",
        }, "+ New board"),
      );
    }

    return h("div", { className: "hermes-kanban-boardswitcher" },
      h("div", { className: "hermes-kanban-boardswitcher-inner" },
        h("div", { className: "flex flex-col gap-0.5" },
          h("div", { className: "text-[11px] uppercase tracking-wider text-muted-foreground" },
            "Board"),
          h("div", { className: "flex items-center gap-2" },
            h(Select, Object.assign({
              value: props.board,
              className: "h-8 min-w-[220px]",
              "aria-label": "Switch kanban board",
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
        h(Button, {
          onClick: props.onNewClick,
          size: "sm",
          className: "h-8",
        }, "+ New board"),
        props.board !== "default"
          ? h(Button, {
            onClick: function () {
              const msg =
                `Archive board '${currentName}'? ` +
                `It will be moved to boards/_archived/ so you can recover it later. ` +
                `Tasks on this board will no longer appear anywhere in the UI.`;
              if (window.confirm(msg)) props.onDeleteBoard(props.board);
            },
            size: "sm",
            className: "h-8",
            title: "Archive this board",
          }, "Archive")
          : null,
      ),
    );
  }

  function NewBoardDialog(props) {
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
        h("div", { className: "hermes-kanban-dialog-title" }, "New board"),
        h("div", { className: "text-xs text-muted-foreground mb-2" },
          "Boards let you separate unrelated streams of work — one per project, repo, or domain. Workers on one board never see another board's tasks."),
        h("div", { className: "flex flex-col gap-3" },
          h("div", { className: "flex flex-col gap-1" },
            h(Label, { className: "text-xs" }, "Slug ",
              h("span", { className: "text-muted-foreground" },
                "— lowercase, hyphens, e.g. atm10-server")),
            h(Input, {
              value: slug,
              onChange: function (e) { setSlug(e.target.value.toLowerCase().replace(/[^a-z0-9\-_]/g, "-")); },
              placeholder: "atm10-server",
              autoFocus: true,
              className: "h-8",
            }),
          ),
          h("div", { className: "flex flex-col gap-1" },
            h(Label, { className: "text-xs" }, "Display name ",
              h("span", { className: "text-muted-foreground" }, "(optional)")),
            h(Input, {
              value: name,
              onChange: function (e) { setName(e.target.value); },
              placeholder: autoName || "Display name",
              className: "h-8",
            }),
          ),
          h("div", { className: "flex flex-col gap-1" },
            h(Label, { className: "text-xs" }, "Description ",
              h("span", { className: "text-muted-foreground" }, "(optional)")),
            h(Input, {
              value: description,
              onChange: function (e) { setDescription(e.target.value); },
              placeholder: "What goes on this board?",
              className: "h-8",
            }),
          ),
          h("div", { className: "flex flex-col gap-1" },
            h(Label, { className: "text-xs" }, "Icon ",
              h("span", { className: "text-muted-foreground" }, "(single character or emoji)")),
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
            "Switch to this board after creating it",
          ),
        ),
        err ? h("div", { className: "text-xs text-destructive mt-2" }, err) : null,
        h("div", { className: "hermes-kanban-dialog-actions" },
          h(Button, {
            type: "button",
            onClick: props.onCancel,
            size: "sm",
            disabled: submitting,
          }, "Cancel"),
          h(Button, {
            type: "submit",
            size: "sm",
            disabled: submitting || !slug.trim(),
          }, submitting ? "Creating…" : "Create board"),
        ),
      ),
    );
  }

  // -------------------------------------------------------------------------
  // Toolbar
  // -------------------------------------------------------------------------

  function BoardToolbar(props) {
    const tenants = (props.board && props.board.tenants) || [];
    const assignees = (props.board && props.board.assignees) || [];
    return h("div", { className: "flex flex-wrap items-end gap-3" },
      h("div", { className: "flex flex-col gap-1" },
        h(Label, { className: "text-xs text-muted-foreground" }, "Search"),
        h(Input, {
          placeholder: "Filter cards…",
          value: props.search,
          onChange: function (e) { props.setSearch(e.target.value); },
          className: "w-56 h-8",
        }),
      ),
      h("div", { className: "flex flex-col gap-1" },
        h(Label, { className: "text-xs text-muted-foreground" }, "Tenant"),
        h(Select, Object.assign({
          value: props.tenantFilter,
          className: "h-8",
        }, selectChangeHandler(props.setTenantFilter)),
          h(SelectOption, { value: "" }, "All tenants"),
          tenants.map(function (t) {
            return h(SelectOption, { key: t, value: t }, t);
          }),
        ),
      ),
      h("div", { className: "flex flex-col gap-1" },
        h(Label, { className: "text-xs text-muted-foreground" }, "Assignee"),
        h(Select, Object.assign({
          value: props.assigneeFilter,
          className: "h-8",
        }, selectChangeHandler(props.setAssigneeFilter)),
          h(SelectOption, { value: "" }, "All profiles"),
          assignees.map(function (a) {
            return h(SelectOption, { key: a, value: a }, a);
          }),
        ),
      ),
      h("label", { className: "flex items-center gap-2 text-xs" },
        h("input", {
          type: "checkbox",
          checked: props.includeArchived,
          onChange: function (e) { props.setIncludeArchived(e.target.checked); },
        }),
        "Show archived",
      ),
      h("label", { className: "flex items-center gap-2 text-xs",
                   title: "Group the Running column by assigned profile" },
        h("input", {
          type: "checkbox",
          checked: props.laneByProfile,
          onChange: function (e) { props.setLaneByProfile(e.target.checked); },
        }),
        "Lanes by profile",
      ),
      h("div", { className: "flex-1" }),
      h(Button, {
        onClick: props.onNudgeDispatch,
        size: "sm",
      }, "Nudge dispatcher"),
      h(Button, {
        onClick: props.onRefresh,
        size: "sm",
      }, "Refresh"),
    );
  }

  // -------------------------------------------------------------------------
  // Bulk action bar (appears when >= 1 card is selected)
  // -------------------------------------------------------------------------

  function BulkActionBar(props) {
    const [assignee, setAssignee] = useState("");
    return h("div", { className: "hermes-kanban-bulk" },
      h("span", { className: "hermes-kanban-bulk-count" },
        `${props.count} selected`),
      h(Button, {
        onClick: function () { props.onApply({ status: "ready" }); },
        size: "sm",
      }, "→ ready"),
      h(Button, {
        onClick: function () {
          props.onApply({ status: "done" },
            `Mark ${props.count} task(s) as done?`);
        },
        size: "sm",
      }, "Complete"),
      h(Button, {
        onClick: function () {
          props.onApply({ archive: true },
            `Archive ${props.count} task(s)?`);
        },
        size: "sm",
      }, "Archive"),
      h("div", { className: "hermes-kanban-bulk-reassign" },
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
            props.onApply({ assignee: assignee === "__none__" ? "" : assignee });
            setAssignee("");
          },
          disabled: !assignee,
          size: "sm",
        }, "Apply"),
      ),
      h("div", { className: "flex-1" }),
      h(Button, {
        onClick: props.onClear,
        size: "sm",
      }, "Clear"),
    );
  }

  // -------------------------------------------------------------------------
  // Columns
  // -------------------------------------------------------------------------

  function BoardColumns(props) {
    return h("div", { className: "hermes-kanban-columns" },
      props.board.columns.map(function (col) {
        return h(Column, {
          key: col.name,
          column: col,
          laneByProfile: props.laneByProfile,
          selectedIds: props.selectedIds,
          toggleSelected: props.toggleSelected,
          onMove: props.onMove,
          onOpen: props.onOpen,
          onCreate: props.onCreate,
          allTasks: props.allTasks,
        });
      }),
    );
  }

  function Column(props) {
    const [dragOver, setDragOver] = useState(false);
    const [showCreate, setShowCreate] = useState(false);
    const colRef = useRef(null);

    // Listen for our synthetic touch-drop events from attachTouchDrag().
    useEffect(function () {
      if (!colRef.current) return undefined;
      const el = colRef.current;
      function onTouchDrop(e) {
        if (e.detail && e.detail.status === props.column.name) {
          props.onMove(e.detail.taskId, props.column.name);
        }
      }
      el.addEventListener("hermes-kanban:drop", onTouchDrop);
      return function () { el.removeEventListener("hermes-kanban:drop", onTouchDrop); };
    }, [props.column.name, props.onMove]);

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
      if (taskId) props.onMove(taskId, props.column.name);
    };

    const lanes = useMemo(function () {
      if (!props.laneByProfile || props.column.name !== "running") return null;
      const byProfile = {};
      for (const t of props.column.tasks) {
        const key = t.assignee || "(unassigned)";
        (byProfile[key] = byProfile[key] || []).push(t);
      }
      return Object.keys(byProfile).sort().map(function (k) {
        return { assignee: k, tasks: byProfile[k] };
      });
    }, [props.column, props.laneByProfile]);

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
      h("div", { className: "hermes-kanban-column-header" },
        h("span", { className: cn("hermes-kanban-dot", COLUMN_DOT[props.column.name]) }),
        h("span", { className: "hermes-kanban-column-label" },
          COLUMN_LABEL[props.column.name] || props.column.name),
        h("span", { className: "hermes-kanban-column-count" },
          props.column.tasks.length),
        h("button", {
          type: "button",
          className: "hermes-kanban-column-add",
          title: "Create task in this column",
          onClick: function () { setShowCreate(function (v) { return !v; }); },
        }, showCreate ? "×" : "+"),
      ),
      h("div", { className: "hermes-kanban-column-sub" },
        COLUMN_HELP[props.column.name] || ""),
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
          ? h("div", { className: "hermes-kanban-empty" }, "— no tasks —")
          : lanes
            ? lanes.map(function (lane) {
                return h("div", { key: lane.assignee, className: "hermes-kanban-lane" },
                  h("div", { className: "hermes-kanban-lane-head" },
                    h("span", { className: "hermes-kanban-lane-name" }, lane.assignee),
                    h("span", { className: "hermes-kanban-lane-count" }, lane.tasks.length),
                  ),
                  lane.tasks.map(function (t) {
                    return h(TaskCard, {
                      key: t.id, task: t,
                      selected: props.selectedIds.has(t.id),
                      toggleSelected: props.toggleSelected,
                      onOpen: props.onOpen,
                    });
                  }),
                );
              })
            : props.column.tasks.map(function (t) {
                return h(TaskCard, {
                  key: t.id, task: t,
                  selected: props.selectedIds.has(t.id),
                  toggleSelected: props.toggleSelected,
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
    const t = props.task;
    const cardRef = useRef(null);

    useEffect(function () {
      return attachTouchDrag(cardRef.current, t.id);
    }, [t.id]);

    const handleDragStart = function (e) {
      e.dataTransfer.setData(MIME_TASK, t.id);
      e.dataTransfer.effectAllowed = "move";
    };
    const handleClick = function (e) {
      // Shift-click or ctrl/cmd-click toggles selection instead of opening.
      if (e.shiftKey || e.ctrlKey || e.metaKey) {
        e.preventDefault();
        e.stopPropagation();
        props.toggleSelected(t.id, e.ctrlKey || e.metaKey);
        return;
      }
      props.onOpen(t.id);
    };
    const handleCheckbox = function (e) {
      e.stopPropagation();
      props.toggleSelected(t.id, true);
    };

    const progress = t.progress;

    return h("div", {
      ref: cardRef,
      className: cn(
        "hermes-kanban-card",
        props.selected ? "hermes-kanban-card--selected" : "",
        stalenessClass(t),
      ),
      draggable: true,
      onDragStart: handleDragStart,
      onClick: handleClick,
    },
      h(Card, null,
        h(CardContent, { className: "hermes-kanban-card-content" },
          h("div", { className: "hermes-kanban-card-row" },
            h("input", {
              type: "checkbox",
              className: "hermes-kanban-card-check",
              checked: props.selected,
              onChange: handleCheckbox,
              onClick: function (e) { e.stopPropagation(); },
              title: "Select for bulk actions",
            }),
            h("span", { className: "hermes-kanban-card-id" }, t.id),
            t.priority > 0
              ? h(Badge, { className: "hermes-kanban-priority" }, `P${t.priority}`)
              : null,
            t.tenant
              ? h(Badge, { variant: "outline", className: "hermes-kanban-tag" }, t.tenant)
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
          h("div", { className: "hermes-kanban-card-title" }, t.title || "(untitled)"),
          h("div", { className: "hermes-kanban-card-row hermes-kanban-card-meta" },
            t.assignee
              ? h("span", { className: "hermes-kanban-assignee" }, "@", t.assignee)
              : h("span", { className: "hermes-kanban-unassigned" }, "unassigned"),
            t.comment_count > 0
              ? h("span", { className: "hermes-kanban-count" }, "💬 ", t.comment_count)
              : null,
            t.link_counts && (t.link_counts.parents + t.link_counts.children) > 0
              ? h("span", { className: "hermes-kanban-count" },
                  "↔ ", t.link_counts.parents + t.link_counts.children)
              : null,
            h("span", { className: "hermes-kanban-ago" },
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
      ? "workspace path (required, e.g. ~/projects/my-app)"
      : "workspace path (optional, derived from assignee if blank)";

    return h("div", { className: "hermes-kanban-inline-create" },
      h(Input, {
        value: title,
        onChange: function (e) { setTitle(e.target.value); },
        onKeyDown: function (e) {
          if (e.key === "Enter") { e.preventDefault(); submit(); }
          if (e.key === "Escape") props.onCancel();
        },
        placeholder: props.columnName === "triage"
          ? "Rough idea — AI will spec it…"
          : "New task title…",
        autoFocus: true,
        className: "h-8 text-sm",
      }),
      h("div", { className: "flex gap-2" },
        h(Input, {
          value: assignee,
          onChange: function (e) { setAssignee(e.target.value); },
          placeholder: props.columnName === "triage" ? "specifier" : "assignee",
          className: "h-7 text-xs flex-1",
        }),
        h(Input, {
          type: "number",
          value: priority,
          onChange: function (e) { setPriority(e.target.value); },
          placeholder: "pri",
          className: "h-7 text-xs w-16",
        }),
      ),
      h(Input, {
        value: skills,
        onChange: function (e) { setSkills(e.target.value); },
        placeholder: "skills (optional, comma-separated): translation, github-code-review",
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
      },
        h(SelectOption, { value: "" }, "— no parent —"),
        (props.allTasks || []).map(function (t) {
          return h(SelectOption, { key: t.id, value: t.id },
            `${t.id} — ${(t.title || "").slice(0, 50)}`);
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
        }, "Cancel"),
      ),
    );
  }

  // -------------------------------------------------------------------------
  // Task drawer
  // -------------------------------------------------------------------------

  function TaskDrawer(props) {
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
            title: "Close (Esc)",
          }, "×"),
        ),
        loading ? h("div", { className: "p-4 text-sm text-muted-foreground" }, "Loading…") :
        err ? h("div", { className: "p-4 text-sm text-destructive" }, err) :
        data ? h(TaskDetail, {
          data, editing, setEditing,
          renderMarkdown: props.renderMarkdown,
          allTasks: props.allTasks,
          boardSlug: boardSlug,
          onPatch: doPatch,
          onAddParent: addLink,
          onRemoveParent: removeLink,
          onAddChild: addChild,
          onRemoveChild: removeChild,
          homeChannels: homeChannels,
          homeBusy: homeBusy,
          onToggleHomeSub: toggleHomeSubscription,
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
            placeholder: "Add a comment… (Enter to submit)",
            className: "h-8 text-sm flex-1",
          }),
          h(Button, {
            onClick: handleComment,
            size: "sm",
          }, "Comment"),
        ) : null,
      ),
    );
  }

  function TaskDetail(props) {
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
              title: "Click to edit",
              onClick: function () { props.setEditing(true); },
            }, t.title || "(untitled)"),
      ),
      h("div", { className: "hermes-kanban-drawer-meta" },
        h(MetaRow, { label: "Status", value: t.status }),
        h(AssigneeEditor, { task: t, onPatch: props.onPatch }),
        h(PriorityEditor, { task: t, onPatch: props.onPatch }),
        t.tenant ? h(MetaRow, { label: "Tenant", value: t.tenant }) : null,
        h(MetaRow, {
          label: "Workspace",
          value: `${t.workspace_kind}${t.workspace_path ? ": " + t.workspace_path : ""}`,
        }),
        (t.skills && t.skills.length > 0) ? h(MetaRow, {
          label: "Skills",
          value: t.skills.join(", "),
        }) : null,
        t.created_by ? h(MetaRow, { label: "Created by", value: t.created_by }) : null,
      ),
      h(StatusActions, { task: t, onPatch: props.onPatch }),
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
        h("div", { className: "hermes-kanban-section-head" }, "Result"),
        h(MarkdownBlock, { source: t.result, enabled: props.renderMarkdown }),
      ) : null,
      h("div", { className: "hermes-kanban-section" },
        h("div", { className: "hermes-kanban-section-head" }, `Comments (${comments.length})`),
        comments.length === 0
          ? h("div", { className: "text-xs text-muted-foreground" }, "— no comments —")
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
        h("div", { className: "hermes-kanban-section-head" }, `Events (${events.length})`),
        events.slice().reverse().slice(0, 20).map(function (e) {
          return h("div", { key: e.id, className: "hermes-kanban-event" },
            h("span", { className: "hermes-kanban-event-kind" }, e.kind),
            h("span", { className: "hermes-kanban-event-ago" },
              timeAgo ? timeAgo(e.created_at) : ""),
            e.payload
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
          `Run history (${runs.length})`),
        !showAll
          ? h("button", {
              type: "button",
              onClick: function () { setExpanded(true); },
              className: "hermes-kanban-edit-link",
              title: "Show all attempts",
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
              r.ended_at ? (r.outcome || r.status || "ended") : "active"),
            h("span", { className: "hermes-kanban-run-profile" },
              r.profile ? `@${r.profile}` : "(no profile)"),
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
          r.metadata
            ? h("code", { className: "hermes-kanban-run-meta" },
                JSON.stringify(r.metadata))
            : null,
        );
      }),
    );
  }

  // Worker log: loads lazily (one GET on mount), refresh button, tail cap.
  function WorkerLogSection(props) {
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
      body = h("div", { className: "text-xs text-muted-foreground" }, "Loading log…");
    } else if (state.err) {
      body = h("div", { className: "text-xs text-destructive" }, state.err);
    } else if (!data || !data.exists) {
      body = h("div", { className: "text-xs text-muted-foreground italic" },
        "— no worker log yet (task hasn't spawned or log was rotated away) —");
    } else {
      body = h("pre", { className: "hermes-kanban-pre hermes-kanban-log" },
        data.content || "(empty)");
    }

    return h("div", { className: "hermes-kanban-section" },
      h("div", { className: "hermes-kanban-section-head-row" },
        h("span", { className: "hermes-kanban-section-head" },
          "Worker log" + (data && data.size_bytes ? ` (${data.size_bytes} B)` : "")),
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
            "(showing last 100 KB — full log at ", data.path, ")")
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
    const [v, setV] = useState(props.initial);
    const save = function () {
      const t = v.trim();
      if (!t) return;
      props.onSave(t);
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
      }, "Save"),
      h(Button, { onClick: props.onCancel,
        size: "sm",
      }, "Cancel"),
    );
  }

  function AssigneeEditor(props) {
    const [editing, setEditing] = useState(false);
    const [v, setV] = useState(props.task.assignee || "");
    useEffect(function () { setV(props.task.assignee || ""); }, [props.task.assignee]);
    if (!editing) {
      return h("div", { className: "hermes-kanban-meta-row" },
        h("span", { className: "hermes-kanban-meta-label" }, "Assignee"),
        h("span", {
          className: "hermes-kanban-meta-value hermes-kanban-editable",
          onClick: function () { setEditing(true); },
          title: "Click to edit",
        }, props.task.assignee || "unassigned"),
      );
    }
    const save = function () {
      props.onPatch({ assignee: v.trim() || "" }).then(function () { setEditing(false); });
    };
    return h("div", { className: "hermes-kanban-meta-row" },
      h("span", { className: "hermes-kanban-meta-label" }, "Assignee"),
      h(Input, {
        value: v, autoFocus: true,
        onChange: function (e) { setV(e.target.value); },
        onKeyDown: function (e) {
          if (e.key === "Enter") { e.preventDefault(); save(); }
          if (e.key === "Escape") setEditing(false);
        },
        placeholder: "(empty = unassign)",
        className: "h-7 text-xs flex-1",
      }),
    );
  }

  function PriorityEditor(props) {
    const [editing, setEditing] = useState(false);
    const [v, setV] = useState(String(props.task.priority || 0));
    useEffect(function () { setV(String(props.task.priority || 0)); }, [props.task.priority]);
    if (!editing) {
      return h("div", { className: "hermes-kanban-meta-row" },
        h("span", { className: "hermes-kanban-meta-label" }, "Priority"),
        h("span", {
          className: "hermes-kanban-meta-value hermes-kanban-editable",
          onClick: function () { setEditing(true); },
          title: "Click to edit",
        }, String(props.task.priority)),
      );
    }
    const save = function () {
      props.onPatch({ priority: Number(v) || 0 }).then(function () { setEditing(false); });
    };
    return h("div", { className: "hermes-kanban-meta-row" },
      h("span", { className: "hermes-kanban-meta-label" }, "Priority"),
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
    const [editing, setEditing] = useState(false);
    const [v, setV] = useState(props.task.body || "");
    useEffect(function () { setV(props.task.body || ""); }, [props.task.body]);
    const save = function () {
      props.onPatch({ body: v }).then(function () { setEditing(false); });
    };
    return h("div", { className: "hermes-kanban-section" },
      h("div", { className: "hermes-kanban-section-head-row" },
        h("span", { className: "hermes-kanban-section-head" }, "Description"),
        editing
          ? h("div", { className: "flex gap-1" },
              h(Button, { onClick: save,
                size: "sm",
              }, "Save"),
              h(Button, { onClick: function () { setEditing(false); setV(props.task.body || ""); },
                size: "sm",
              }, "Cancel"),
            )
          : h("button", {
              type: "button",
              onClick: function () { setEditing(true); },
              className: "hermes-kanban-edit-link",
              title: "Edit description",
            }, "edit"),
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
          : h("div", { className: "text-xs text-muted-foreground italic" }, "— no description —"),
    );
  }

  function DependencyEditor(props) {
    const { task, links, allTasks } = props;
    const [newParent, setNewParent] = useState("");
    const [newChild, setNewChild] = useState("");
    // Filter out self + existing links when offering the "add" dropdown.
    const candidatesFor = function (excludeSet) {
      return (allTasks || []).filter(function (t) {
        return t.id !== task.id && !excludeSet.has(t.id);
      });
    };
    const parentExclude = new Set([task.id, ...(links.parents || [])]);
    const childExclude  = new Set([task.id, ...(links.children || [])]);

    return h("div", { className: "hermes-kanban-section" },
      h("div", { className: "hermes-kanban-section-head" }, "Dependencies"),
      h("div", { className: "hermes-kanban-deps-row" },
        h("span", { className: "hermes-kanban-deps-label" }, "Parents:"),
        h("div", { className: "hermes-kanban-deps-chips" },
          (links.parents || []).length === 0
            ? h("span", { className: "hermes-kanban-deps-empty" }, "none")
            : (links.parents || []).map(function (id) {
                return h("span", { key: id, className: "hermes-kanban-dep-chip" },
                  id,
                  h("button", {
                    type: "button",
                    className: "hermes-kanban-dep-chip-x",
                    onClick: function () { props.onRemoveParent(id); },
                    title: "Remove dependency",
                  }, "×"),
                );
              }),
        ),
      ),
      h("div", { className: "hermes-kanban-deps-row" },
        h(Select, {
          value: newParent,
          onChange: function (e) { setNewParent(e.target.value); },
          className: "h-7 text-xs flex-1",
        },
          h(SelectOption, { value: "" }, "— add parent —"),
          candidatesFor(parentExclude).map(function (t) {
            return h(SelectOption, { key: t.id, value: t.id },
              `${t.id} — ${(t.title || "").slice(0, 50)}`);
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
        h("span", { className: "hermes-kanban-deps-label" }, "Children:"),
        h("div", { className: "hermes-kanban-deps-chips" },
          (links.children || []).length === 0
            ? h("span", { className: "hermes-kanban-deps-empty" }, "none")
            : (links.children || []).map(function (id) {
                return h("span", { key: id, className: "hermes-kanban-dep-chip" },
                  id,
                  h("button", {
                    type: "button",
                    className: "hermes-kanban-dep-chip-x",
                    onClick: function () { props.onRemoveChild(id); },
                    title: "Remove dependency",
                  }, "×"),
                );
              }),
        ),
      ),
      h("div", { className: "hermes-kanban-deps-row" },
        h(Select, {
          value: newChild,
          onChange: function (e) { setNewChild(e.target.value); },
          className: "h-7 text-xs flex-1",
        },
          h(SelectOption, { value: "" }, "— add child —"),
          candidatesFor(childExclude).map(function (t) {
            return h(SelectOption, { key: t.id, value: t.id },
              `${t.id} — ${(t.title || "").slice(0, 50)}`);
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
    const t = props.task;
    const b = function (label, patch, enabled, confirmMsg) {
      return h(Button, {
        onClick: function () { if (enabled !== false) props.onPatch(patch, { confirm: confirmMsg }); },
        disabled: enabled === false,
        size: "sm",
      }, label);
    };
    return h("div", { className: "hermes-kanban-actions" },
      b("→ triage",  { status: "triage" },   t.status !== "triage"),
      b("→ ready",   { status: "ready" },    t.status !== "ready"),
      // No direct → running button: /tasks/:id PATCH rejects status=running
      // with 400 (issue #19535). Tasks enter running only through the
      // dispatcher's claim_task path, which atomically creates the run row,
      // claim lock, and worker process metadata.
      b("Block",     { status: "blocked" },
        t.status === "running" || t.status === "ready",
        DESTRUCTIVE_TRANSITIONS.blocked),
      b("Unblock",   { status: "ready" },    t.status === "blocked"),
      b("Complete",  { status: "done" },
        t.status === "running" || t.status === "ready" || t.status === "blocked",
        DESTRUCTIVE_TRANSITIONS.done),
      b("Archive",   { status: "archived" }, t.status !== "archived",
        DESTRUCTIVE_TRANSITIONS.archived),
    );
  }


  // One toggle per gateway platform the user has a home channel set on
  // (telegram, discord, slack, etc.). Toggling on creates a kanban_notify_subs
  // row routed to that platform's home; toggling off removes it. Nothing
  // renders when no platforms have a home configured — this section stays
  // invisible for users who haven't set one up.
  function HomeSubsSection(props) {
    const channels = props.homeChannels || [];
    if (channels.length === 0) return null;
    const busy = props.homeBusy || {};
    return h("div", { className: "hermes-kanban-section" },
      h("div", { className: "hermes-kanban-section-head" },
        "Notify home channels"),
      h("div", { className: "hermes-kanban-home-subs" },
        channels.map(function (hc) {
          const isBusy = !!busy[hc.platform];
          const label = hc.subscribed ? "✓ " + hc.platform : hc.platform;
          const title = hc.subscribed
            ? `Sending updates to ${hc.name} (${hc.chat_id}${hc.thread_id ? " / " + hc.thread_id : ""}). Click to stop.`
            : `Send completed / blocked / gave_up notifications to ${hc.name} (${hc.chat_id}${hc.thread_id ? " / " + hc.thread_id : ""}).`;
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

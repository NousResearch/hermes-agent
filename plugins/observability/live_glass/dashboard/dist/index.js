/**
 * Hermes Live Glass — Dashboard Plugin
 *
 * Live view into computer_use activity. Connects to the live-glass
 * WebSocket stream at /api/plugins/live-glass/events and renders
 * three stacked regions: screenshot viewport, scrolling event log,
 * and approval controls.
 *
 * Plain IIFE, no build step. Uses window.__HERMES_PLUGIN_SDK__ for
 * React + shadcn primitives; WebSocket for live updates.
 */
(function () {
  "use strict";

  var SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK) return;

  var React = SDK.React;
  var h = React.createElement;
  var useState = SDK.hooks.useState;
  var useEffect = SDK.hooks.useEffect;
  var useRef = SDK.hooks.useRef;
  var useCallback = SDK.hooks.useCallback;
  var Card = SDK.components.Card;
  var CardContent = SDK.components.CardContent;
  var Button = SDK.components.Button;
  var Badge = SDK.components.Badge;
  var cn = SDK.utils.cn;

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  /** Format a unix timestamp (seconds) to HH:MM:SS. */
  function formatTime(ts) {
    var d = new Date(ts * 1000);
    var hh = String(d.getHours()).padStart(2, "0");
    var mm = String(d.getMinutes()).padStart(2, "0");
    var ss = String(d.getSeconds()).padStart(2, "0");
    return hh + ":" + mm + ":" + ss;
  }

  /** Status icon for log entries. */
  function statusIcon(status) {
    if (status === "ok" || status === "success") return "\u2713"; // ✓
    if (status === "error" || status === "timeout") return "\u2717"; // ✗
    return "\u2022"; // •
  }

  /** Status class for coloring. */
  function statusClass(status) {
    if (status === "ok" || status === "success") return "hermes-lg-log-ok";
    if (status === "error" || status === "timeout") return "hermes-lg-log-error";
    return "hermes-lg-log-neutral";
  }

  /** Build a WebSocket URL for the live-glass events endpoint. */
  function buildWsUrl() {
    if (SDK.buildWsUrl) {
      return SDK.buildWsUrl("/api/plugins/live-glass/events");
    }
    var proto = window.location.protocol === "https:" ? "wss" : "ws";
    return proto + "://" + window.location.host + "/api/plugins/live-glass/events";
  }

  // ---------------------------------------------------------------------------
  // Approval Request Controls
  // ---------------------------------------------------------------------------

  function ApprovalControls(props) {
    var request = props.request;
    var onDismiss = props.onDismiss;

    if (!request) return null;

    var payload = request.payload || {};
    var command = payload.command || "";
    var description = payload.description || "";
    var surface = payload.surface || "";
    var timestamp = formatTime(request.timestamp);

    return h("div", { className: "hermes-lg-approval" },
      h("div", { className: "hermes-lg-approval-header" },
        h(Badge, { className: "hermes-lg-approval-badge" }, "Approval Required"),
        h("span", { className: "hermes-lg-approval-time" }, timestamp)
      ),
      h("div", { className: "hermes-lg-approval-body" },
        h("div", { className: "hermes-lg-approval-command" },
          h("code", null, command)
        ),
        description
          ? h("div", { className: "hermes-lg-approval-desc" }, description)
          : null,
        surface
          ? h("div", { className: "hermes-lg-approval-surface" },
              "Surface: ", h("strong", null, surface))
          : null
      ),
      h("div", { className: "hermes-lg-approval-actions" },
        h(Button, {
          size: "sm",
          onClick: function () {
            if (onDismiss) onDismiss("approve");
          },
          className: "hermes-lg-btn-approve",
        }, "Approve"),
        h(Button, {
          size: "sm",
          variant: "outline",
          onClick: function () {
            if (onDismiss) onDismiss("deny");
          },
          className: "hermes-lg-btn-deny",
        }, "Deny")
      )
    );
  }

  // ---------------------------------------------------------------------------
  // Event Log
  // ---------------------------------------------------------------------------

  function renderLogEntry(event) {
    var type = event.type;
    var ts = formatTime(event.timestamp);
    var payload = event.payload || {};
    var toolName = payload.tool_name || "";
    var status = payload.status || "";

    // Skip heartbeats in the log display
    if (type === "heartbeat") return null;

    var icon;
    var label;
    var cls;

    if (type === "frame") {
      icon = "\uD83D\uDCF7"; // camera emoji
      label = "Screenshot";
      if (payload.summary) {
        label += " — " + payload.summary;
      }
      cls = "hermes-lg-log-frame";
    } else if (type === "log") {
      icon = statusIcon(status);
      label = toolName || "tool";
      cls = statusClass(status);
    } else if (type === "approval_request") {
      icon = "\u26A0"; // warning
      label = "Approval: " + (payload.command || "unknown").slice(0, 60);
      cls = "hermes-lg-log-approval";
    } else {
      icon = "\u2022";
      label = type;
      cls = "hermes-lg-log-neutral";
    }

    return h("div", {
      key: event.id || event.sequence,
      className: "hermes-lg-log-entry " + cls,
    },
      h("span", { className: "hermes-lg-log-time" }, ts),
      h("span", { className: "hermes-lg-log-icon" }, icon),
      h("span", { className: "hermes-lg-log-label" }, label)
    );
  }

  function EventLog(props) {
    var events = props.events || [];
    var logRef = useRef(null);

    // Auto-scroll to bottom when new events arrive
    useEffect(function () {
      var el = logRef.current;
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    }, [events.length]);

    return h("div", { className: "hermes-lg-log" },
      h("div", { className: "hermes-lg-log-header" },
        h("span", { className: "hermes-lg-log-title" }, "Event Log"),
        h(Badge, { variant: "secondary", className: "hermes-lg-log-count" },
          events.length)
      ),
      h("div", {
        ref: logRef,
        className: "hermes-lg-log-list",
      },
        events.length === 0
          ? h("div", { className: "hermes-lg-log-empty" }, "No events yet")
          : events.map(function (ev) { return renderLogEntry(ev); })
      )
    );
  }

  // ---------------------------------------------------------------------------
  // Viewport — latest frame screenshot
  // ---------------------------------------------------------------------------

  function Viewport(props) {
    var frame = props.frame;
    var connected = props.connected;

    return h("div", { className: "hermes-lg-viewport" },
      // Connection status indicator
      h("div", { className: "hermes-lg-viewport-status" },
        h("span", {
          className: "hermes-lg-status-dot" + (connected ? " hermes-lg-status-dot--live" : ""),
        }),
        h("span", { className: "hermes-lg-status-label" },
          connected ? "Live" : "Connecting..."
        )
      ),

      frame && frame.payload && frame.payload.image_url
        ? h("div", { className: "hermes-lg-viewport-frame" },
            h("img", {
              src: frame.payload.image_url,
              alt: frame.payload.summary || "Screenshot",
              className: "hermes-lg-viewport-img",
            }),
            h("div", { className: "hermes-lg-viewport-meta" },
              frame.payload.summary
                ? h("span", { className: "hermes-lg-viewport-summary" }, frame.payload.summary)
                : null,
              h("span", { className: "hermes-lg-viewport-dims" },
                (frame.payload.width && frame.payload.height)
                  ? frame.payload.width + "\u00D7" + frame.payload.height
                  : ""
              ),
              frame.payload.mode
                ? h(Badge, { variant: "secondary", className: "hermes-lg-viewport-mode" }, frame.payload.mode)
                : null
            )
          )
        : h("div", { className: "hermes-lg-viewport-placeholder" },
            h("div", { className: "hermes-lg-viewport-placeholder-icon" }, "\uD83D\uDD2C"),
            h("div", { className: "hermes-lg-viewport-placeholder-text" },
              "Waiting for computer_use activity..."),
            h("div", { className: "hermes-lg-viewport-placeholder-hint" },
              "Screenshots will appear here when a computer_use tool runs.")
          )
    );
  }

  // ---------------------------------------------------------------------------
  // Root page
  // ---------------------------------------------------------------------------

  function LiveGlassPage() {
    var _useState = useState(null);
    var currentFrame = _useState[0];
    var setCurrentFrame = _useState[1];

    var _useState2 = useState([]);
    var events = _useState2[0];
    var setEvents = _useState2[1];

    var _useState3 = useState(null);
    var approvalRequest = _useState3[0];
    var setApprovalRequest = _useState3[1];

    var _useState4 = useState(false);
    var connected = _useState4[0];
    var setConnected = _useState4[1];

    var wsRef = useRef(null);
    var reconnectTimerRef = useRef(null);

    var connect = useCallback(function () {
      // Clean up any existing connection
      if (wsRef.current) {
        try { wsRef.current.close(); } catch (_e) { /* ignore */ }
      }

      var url = buildWsUrl();
      var ws;

      try {
        ws = new WebSocket(url);
      } catch (err) {
        // Retry after delay
        reconnectTimerRef.current = setTimeout(function () {
          connect();
        }, 3000);
        return;
      }

      wsRef.current = ws;

      ws.onopen = function () {
        setConnected(true);
      };

      ws.onclose = function () {
        setConnected(false);
        // Auto-reconnect after 3 seconds
        reconnectTimerRef.current = setTimeout(function () {
          connect();
        }, 3000);
      };

      ws.onerror = function () {
        // onclose will fire after this and handle reconnect
      };

      ws.onmessage = function (e) {
        try {
          var event = JSON.parse(e.data);

          // Handle frame events — update current frame
          if (event.type === "frame") {
            setCurrentFrame(event);
          }

          // Handle approval_request events — show controls
          if (event.type === "approval_request") {
            setApprovalRequest(event);
          }

          // Add all events (except heartbeats for display count) to log
          if (event.type !== "heartbeat") {
            setEvents(function (prev) {
              var next = prev.concat([event]);
              // Keep only last 50
              if (next.length > 50) {
                next = next.slice(next.length - 50);
              }
              return next;
            });
          }
        } catch (_err) {
          // Ignore malformed messages
        }
      };
    }, []);

    // Connect on mount, cleanup on unmount
    useEffect(function () {
      connect();
      return function () {
        if (wsRef.current) {
          try { wsRef.current.close(); } catch (_e) { /* ignore */ }
        }
        if (reconnectTimerRef.current) {
          clearTimeout(reconnectTimerRef.current);
        }
      };
    }, [connect]);

    var handleApprovalDismiss = useCallback(function (_choice) {
      // Dismiss the approval request from the UI.
      // The live-glass plugin observes but does not make approval
      // decisions — this is a display-only control for visibility.
      // In future, this could POST to a Hermes approval endpoint.
      setApprovalRequest(null);
    }, []);

    return h("div", { className: "hermes-lg-root" },
      // 1. Viewport
      h(Viewport, { frame: currentFrame, connected: connected }),

      // 2. Approval controls
      h(ApprovalControls, {
        request: approvalRequest,
        onDismiss: handleApprovalDismiss,
      }),

      // 3. Event log
      h(EventLog, { events: events })
    );
  }

  // ---------------------------------------------------------------------------
  // Styles
  // ---------------------------------------------------------------------------

  function injectStyles() {
    var id = "hermes-live-glass-styles";
    if (document.getElementById(id)) return;

    var css = [
      /* Root container */
      ".hermes-lg-root {",
      "  display: flex;",
      "  flex-direction: column;",
      "  gap: 12px;",
      "  height: 100%;",
      "  overflow: hidden;",
      "}",

      /* Viewport */
      ".hermes-lg-viewport {",
      "  flex-shrink: 0;",
      "  border: 1px solid var(--border, hsl(var(--border)));",
      "  border-radius: var(--radius, 0.5rem);",
      "  background: var(--muted, hsl(var(--muted)));",
      "  overflow: hidden;",
      "  position: relative;",
      "  min-height: 200px;",
      "  max-height: 50vh;",
      "}",
      ".hermes-lg-viewport-status {",
      "  position: absolute;",
      "  top: 8px;",
      "  right: 8px;",
      "  z-index: 2;",
      "  display: flex;",
      "  align-items: center;",
      "  gap: 6px;",
      "  background: var(--background, hsl(var(--background)));",
      "  border: 1px solid var(--border, hsl(var(--border)));",
      "  border-radius: 9999px;",
      "  padding: 2px 10px 2px 6px;",
      "  font-size: 12px;",
      "  color: var(--muted-foreground, hsl(var(--muted-foreground)));",
      "}",
      ".hermes-lg-status-dot {",
      "  width: 8px;",
      "  height: 8px;",
      "  border-radius: 50%;",
      "  background: var(--muted-foreground, hsl(var(--muted-foreground)));",
      "}",
      ".hermes-lg-status-dot--live {",
      "  background: #22c55e;",
      "  box-shadow: 0 0 4px #22c55e;",
      "}",
      ".hermes-lg-status-label {",
      "  font-weight: 500;",
      "}",
      ".hermes-lg-viewport-frame {",
      "  position: relative;",
      "}",
      ".hermes-lg-viewport-img {",
      "  display: block;",
      "  width: 100%;",
      "  max-height: 50vh;",
      "  object-fit: contain;",
      "  background: #000;",
      "}",
      ".hermes-lg-viewport-meta {",
      "  position: absolute;",
      "  bottom: 0;",
      "  left: 0;",
      "  right: 0;",
      "  display: flex;",
      "  align-items: center;",
      "  gap: 8px;",
      "  padding: 6px 12px;",
      "  background: linear-gradient(transparent, rgba(0,0,0,0.7));",
      "  font-size: 12px;",
      "  color: #fff;",
      "}",
      ".hermes-lg-viewport-summary {",
      "  flex: 1;",
      "  overflow: hidden;",
      "  text-overflow: ellipsis;",
      "  white-space: nowrap;",
      "}",
      ".hermes-lg-viewport-dims {",
      "  opacity: 0.8;",
      "}",
      ".hermes-lg-viewport-mode {",
      "  font-size: 10px;",
      "}",
      ".hermes-lg-viewport-placeholder {",
      "  display: flex;",
      "  flex-direction: column;",
      "  align-items: center;",
      "  justify-content: center;",
      "  height: 200px;",
      "  color: var(--muted-foreground, hsl(var(--muted-foreground)));",
      "  gap: 8px;",
      "}",
      ".hermes-lg-viewport-placeholder-icon {",
      "  font-size: 32px;",
      "  opacity: 0.4;",
      "}",
      ".hermes-lg-viewport-placeholder-text {",
      "  font-size: 14px;",
      "  font-weight: 500;",
      "}",
      ".hermes-lg-viewport-placeholder-hint {",
      "  font-size: 12px;",
      "  opacity: 0.6;",
      "}",

      /* Approval controls */
      ".hermes-lg-approval {",
      "  flex-shrink: 0;",
      "  border: 1px solid var(--warning, #f59e0b);",
      "  border-radius: var(--radius, 0.5rem);",
      "  background: var(--warning-foreground, hsl(48 96% 89%));",
      "  padding: 12px;",
      "}",
      ".hermes-lg-approval-header {",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: space-between;",
      "  margin-bottom: 8px;",
      "}",
      ".hermes-lg-approval-badge {",
      "  background: var(--warning, #f59e0b);",
      "  color: var(--warning-foreground, #000);",
      "}",
      ".hermes-lg-approval-time {",
      "  font-size: 12px;",
      "  color: var(--muted-foreground, hsl(var(--muted-foreground)));",
      "}",
      ".hermes-lg-approval-body {",
      "  margin-bottom: 12px;",
      "}",
      ".hermes-lg-approval-command {",
      "  font-size: 13px;",
      "  margin-bottom: 4px;",
      "}",
      ".hermes-lg-approval-command code {",
      "  background: var(--muted, hsl(var(--muted)));",
      "  padding: 2px 6px;",
      "  border-radius: 3px;",
      "  font-size: 12px;",
      "  word-break: break-all;",
      "}",
      ".hermes-lg-approval-desc {",
      "  font-size: 13px;",
      "  color: var(--muted-foreground, hsl(var(--muted-foreground)));",
      "  margin-bottom: 4px;",
      "}",
      ".hermes-lg-approval-surface {",
      "  font-size: 12px;",
      "  color: var(--muted-foreground, hsl(var(--muted-foreground)));",
      "}",
      ".hermes-lg-approval-actions {",
      "  display: flex;",
      "  gap: 8px;",
      "}",
      ".hermes-lg-btn-approve {",
      "  background: #22c55e !important;",
      "  color: #fff !important;",
      "}",
      ".hermes-lg-btn-approve:hover {",
      "  background: #16a34a !important;",
      "}",
      ".hermes-lg-btn-deny {",
      "  border-color: #ef4444 !important;",
      "  color: #ef4444 !important;",
      "}",

      /* Event log */
      ".hermes-lg-log {",
      "  flex: 1;",
      "  min-height: 0;",
      "  display: flex;",
      "  flex-direction: column;",
      "  border: 1px solid var(--border, hsl(var(--border)));",
      "  border-radius: var(--radius, 0.5rem);",
      "  background: var(--background, hsl(var(--background)));",
      "  overflow: hidden;",
      "}",
      ".hermes-lg-log-header {",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: space-between;",
      "  padding: 8px 12px;",
      "  border-bottom: 1px solid var(--border, hsl(var(--border)));",
      "  flex-shrink: 0;",
      "}",
      ".hermes-lg-log-title {",
      "  font-size: 13px;",
      "  font-weight: 600;",
      "}",
      ".hermes-lg-log-count {",
      "  font-size: 11px;",
      "}",
      ".hermes-lg-log-list {",
      "  flex: 1;",
      "  overflow-y: auto;",
      "  padding: 4px 0;",
      "  font-family: 'SF Mono', 'Fira Code', 'Fira Mono', Menlo, Consolas, monospace;",
      "  font-size: 12px;",
      "}",
      ".hermes-lg-log-entry {",
      "  display: flex;",
      "  align-items: baseline;",
      "  gap: 8px;",
      "  padding: 3px 12px;",
      "  white-space: nowrap;",
      "}",
      ".hermes-lg-log-entry:hover {",
      "  background: var(--accent, hsl(var(--accent)) / 0.08);",
      "}",
      ".hermes-lg-log-time {",
      "  color: var(--muted-foreground, hsl(var(--muted-foreground)));",
      "  flex-shrink: 0;",
      "  width: 58px;",
      "}",
      ".hermes-lg-log-icon {",
      "  flex-shrink: 0;",
      "  width: 16px;",
      "  text-align: center;",
      "}",
      ".hermes-lg-log-label {",
      "  overflow: hidden;",
      "  text-overflow: ellipsis;",
      "}",
      ".hermes-lg-log-ok .hermes-lg-log-icon {",
      "  color: #22c55e;",
      "}",
      ".hermes-lg-log-error .hermes-lg-log-icon {",
      "  color: #ef4444;",
      "}",
      ".hermes-lg-log-frame .hermes-lg-log-icon {",
      "  color: #3b82f6;",
      "}",
      ".hermes-lg-log-approval .hermes-lg-log-icon {",
      "  color: #f59e0b;",
      "}",
      ".hermes-lg-log-empty {",
      "  padding: 24px;",
      "  text-align: center;",
      "  color: var(--muted-foreground, hsl(var(--muted-foreground)));",
      "  font-size: 13px;",
      "}",
    ].join("\n");

    var style = document.createElement("style");
    style.id = id;
    style.textContent = css;
    document.head.appendChild(style);
  }

  // ---------------------------------------------------------------------------
  // Register
  // ---------------------------------------------------------------------------

  injectStyles();

  if (window.__HERMES_PLUGINS__ && typeof window.__HERMES_PLUGINS__.register === "function") {
    window.__HERMES_PLUGINS__.register("live-glass", LiveGlassPage);
  }
})();

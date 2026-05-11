/* Hermes Docs — dashboard plugin frontend bundle
   Version: 0.1.0
   Design: Mintlify (see DESIGN.md)

   Uses window.__HERMES_PLUGIN_SDK__ exclusively — no npm imports.
   All API calls go to /api/plugins/hermes-docs/* (mounted by plugin_api.py).
*/

(function () {
  "use strict";

  var SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK) { console.error("[hermes-docs] SDK not available"); return; }

  var React      = SDK.React;
  var useState   = SDK.hooks.useState;
  var useEffect  = SDK.hooks.useEffect;
  var useCallback = SDK.hooks.useCallback;
  var useRef     = SDK.hooks.useRef;
  var useMemo    = SDK.hooks.useMemo;

  var h = React.createElement;

  // --------------------------------------------------------------------------
  // Minimal SVG icons (inline, no external deps)
  // --------------------------------------------------------------------------

  function Icon(props) {
    var size  = props.size  || 14;
    var color = props.color || "currentColor";
    var style = { display: "inline-block", verticalAlign: "middle", flexShrink: 0 };
    if (props.style) Object.assign(style, props.style);

    var paths = {
      file:       "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zm4 18H6V4h7v5h5v11z",
      folder:     "M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z",
      "folder-open": "M20 6h-8l-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zM4 18V8h16v10H4z",
      plus:       "M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z",
      minus:      "M19 13H5v-2h14v2z",
      close:      "M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z",
      chevron:    "M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z",
      "chevron-down": "M16.59 8.59L12 13.17 7.41 8.59 6 10l6 6 6-6z",
      pin:        "M16 12V4h1V2H7v2h1v8l-2 2v2h5.2v6h1.6v-6H18v-2l-2-2z",
      "pin-off":  "M3.27 5l-1.27 1.27 7.01 7.01L7 14v2h5v6h2v-6h3.73L12 10.27V4h1V2H9.27L3.27 5z",
      eye:        "M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z",
      edit:       "M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04a1 1 0 0 0 0-1.41l-2.34-2.34a1 1 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z",
      send:       "M2.01 21L23 12 2.01 3 2 10l15 2-15 2z",
      docs:       "M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z",
      settings:   "M12 2A10 10 0 1 0 12 22A10 10 0 0 0 12 2ZM13.7 19.1A8.1 8.1 0 0 1 10.3 19.1L10.6 17.6C10.9 17.7 11.4 17.7 12 17.7C12.6 17.7 13.1 17.7 13.4 17.6L13.7 19.1ZM10.4 4.9C10.9 4.8 11.4 4.8 12 4.8C12.6 4.8 13.1 4.8 13.6 4.9L13.3 6.4C13 6.3 12.6 6.3 12 6.3C11.4 6.3 11 6.3 10.7 6.4L10.4 4.9ZM6.3 13.7L4.9 13.4C4.8 12.9 4.8 12.4 4.8 12C4.8 11.4 4.8 11 4.9 10.6L6.4 10.7C6.3 11 6.3 11.4 6.3 12C6.3 12.6 6.3 13 6.4 13.4L6.3 13.7ZM17.7 13.4C17.8 13 17.8 12.6 17.8 12C17.8 11.4 17.7 11 17.6 10.7L19.1 10.4C19.2 10.9 19.2 11.4 19.2 12C19.2 12.6 19.1 13.1 19 13.7L17.7 13.4Z",
      check:      "M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z",
      warn:       "M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z",
      home:       "M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z",
      "arrow-back": "M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z",
    };

    return h("svg", {
      xmlns: "http://www.w3.org/2000/svg",
      width: size,
      height: size,
      viewBox: "0 0 24 24",
      fill: color,
      style: style,
    }, h("path", { d: paths[props.name] || "" }));
  }

  // --------------------------------------------------------------------------
  // Markdown → HTML renderer (client-side, no deps)
  // --------------------------------------------------------------------------

  function renderMarkdown(raw) {
    if (!raw) return "<p><em>Empty file.</em></p>";
    var lines = raw.split("\n");
    var out   = [];
    var inCodeBlock = false;
    var codeLang    = "";
    var codeLines   = [];

    function escHtml(s) {
      return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    }

    function inlineFormat(s) {
      s = escHtml(s);
      // Bold + italic
      s = s.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
      s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
      s = s.replace(/\*(.+?)\*/g, "<em>$1</em>");
      // Inline code
      s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
      // Links
      s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
      return s;
    }

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];

      if (inCodeBlock) {
        if (line.startsWith("```")) {
          out.push('<pre><code class="language-' + codeLang + '">'
            + escHtml(codeLines.join("\n")) + "</code></pre>");
          inCodeBlock = false; codeLines = []; codeLang = "";
        } else {
          codeLines.push(line);
        }
        continue;
      }

      if (line.startsWith("```")) {
        inCodeBlock = true; codeLang = line.slice(3).trim(); continue;
      }
      if (/^#{4} /.test(line)) { out.push("<h4>" + inlineFormat(line.slice(5)) + "</h4>"); continue; }
      if (/^### /.test(line))  { out.push("<h3>" + inlineFormat(line.slice(4)) + "</h3>"); continue; }
      if (/^## /.test(line))   { out.push("<h2>" + inlineFormat(line.slice(3)) + "</h2>"); continue; }
      if (/^# /.test(line))    { out.push("<h1>" + inlineFormat(line.slice(2)) + "</h1>"); continue; }
      if (/^> /.test(line))    { out.push("<blockquote>" + inlineFormat(line.slice(2)) + "</blockquote>"); continue; }
      if (/^---+$/.test(line)) { out.push("<hr>"); continue; }
      if (/^[-*] /.test(line)) { out.push("<li>" + inlineFormat(line.slice(2)) + "</li>"); continue; }
      if (/^\d+\. /.test(line)){ out.push("<li>" + inlineFormat(line.replace(/^\d+\. /, "")) + "</li>"); continue; }
      if (line.trim() === "")  { out.push("<br>"); continue; }
      out.push("<p>" + inlineFormat(line) + "</p>");
    }

    // Wrap consecutive <li> elements in <ul>
    var html = out.join("\n");
    html = html.replace(/(<li>[\s\S]*?<\/li>\n*)+/g, function(m) {
      return "<ul>" + m + "</ul>";
    });
    return html;
  }

  // --------------------------------------------------------------------------
  // API helpers
  // --------------------------------------------------------------------------

  var BASE = "/api/plugins/hermes-docs";

  function api(method, path, body) {
    var opts = {
      method: method,
      headers: { "Content-Type": "application/json" },
    };
    if (body !== undefined) opts.body = JSON.stringify(body);
    return SDK.fetchJSON(BASE + path, opts);
  }

  var apiGet    = function(path)       { return api("GET",    path); };
  var apiPost   = function(path, body) { return api("POST",   path, body); };
  var apiPut    = function(path, body) { return api("PUT",    path, body); };
  var apiPatch  = function(path, body) { return api("PATCH",  path, body); };
  var apiDelete = function(path)       { return api("DELETE", path); };

  // --------------------------------------------------------------------------
  // WorkspaceLauncher — initial screen: list workspaces, add new
  // --------------------------------------------------------------------------

  function WorkspaceLauncher(props) {
    var workspaces  = props.workspaces;
    var onOpen      = props.onOpen;
    var onAdd       = props.onAdd;
    var onRemove    = props.onRemove;
    var loading     = props.loading;
    var error       = props.error;

    var nameRef     = useRef(null);
    var pathRef     = useRef(null);
    var adding      = useState(false);
    var setAdding   = adding[1]; adding = adding[0];
    var addError    = useState(""); var setAddError = addError[1]; addError = addError[0];
    var submitting  = useState(false); var setSubmitting = submitting[1]; submitting = submitting[0];

    function handleAdd(e) {
      e.preventDefault();
      var name = (nameRef.current ? nameRef.current.value : "").trim();
      var path = (pathRef.current ? pathRef.current.value : "").trim();
      if (!path) { setAddError("Folder path is required."); return; }
      setSubmitting(true); setAddError("");
      onAdd(name, path)
        .then(function() {
          setAdding(false); setSubmitting(false);
          if (nameRef.current) nameRef.current.value = "";
          if (pathRef.current) pathRef.current.value = "";
        })
        .catch(function(err) { setAddError(err.message || "Failed to add workspace."); setSubmitting(false); });
    }

    return h("div", { className: "hd-launcher" },
      h("div", { className: "hd-launcher__header" },
        h("h2", { className: "hd-launcher__title" }, "Hermes Docs"),
        h("p", { className: "hd-launcher__subtitle" },
          "Select a workspace or register a new local folder."
        )
      ),
      error && h("div", { className: "hd-banner hd-banner--error", style: { marginBottom: 16 } },
        h(Icon, { name: "warn", size: 14 }), " ", error
      ),
      h("div", { className: "hd-launcher__body" },
        loading
          ? h("div", { style: { color: "var(--hd-muted)", fontSize: 13, padding: "24px 0" } }, "Loading workspaces…")
          : workspaces.length === 0 && !adding
            ? h("div", { style: { color: "var(--hd-steel)", fontSize: 13, padding: "16px 0" } },
                "No workspaces yet. Register a local folder to get started."
              )
            : workspaces.map(function(ws) {
                return h("div", { key: ws.id, className: "hd-workspace-card", onClick: function() { onOpen(ws); } },
                  h("div", { className: "hd-workspace-card__icon" }, "📁"),
                  h("div", { className: "hd-workspace-card__info" },
                    h("div", { className: "hd-workspace-card__name" }, ws.name),
                    h("div", { className: "hd-workspace-card__path" }, ws.path)
                  ),
                  !ws.folder_exists && h("span", { className: "hd-workspace-card__badge hd-workspace-card__badge--missing" }, "Missing"),
                  h("button", {
                    className: "hd-workspace-card__remove",
                    title: "Remove workspace",
                    onClick: function(e) { e.stopPropagation(); onRemove(ws.id); },
                  }, h(Icon, { name: "close", size: 14 }))
                );
              }),

        adding
          ? h("form", { className: "hd-add-workspace", onSubmit: handleAdd },
              h("div", { className: "hd-add-workspace__title" }, "Register workspace"),
              h("div", { className: "hd-add-workspace__fields" },
                h("input", {
                  ref: nameRef,
                  className: "hd-input",
                  type: "text",
                  placeholder: "Workspace name (optional — defaults to folder name)",
                }),
                h("div", { className: "hd-add-workspace__row" },
                  h("input", {
                    ref: pathRef,
                    className: "hd-input",
                    type: "text",
                    placeholder: "/absolute/path/to/folder",
                    required: true,
                    autoFocus: true,
                  }),
                  h("button", { type: "submit", className: "hd-btn hd-btn--primary", disabled: submitting },
                    submitting ? "Adding…" : "Add"
                  ),
                  h("button", { type: "button", className: "hd-btn hd-btn--ghost",
                    onClick: function() { setAdding(false); setAddError(""); } }, "Cancel")
                ),
                addError && h("div", { className: "hd-banner hd-banner--error" },
                  h(Icon, { name: "warn", size: 13 }), " ", addError
                )
              )
            )
          : h("button", {
              className: "hd-btn hd-btn--ghost",
              style: { alignSelf: "flex-start", marginTop: 8 },
              onClick: function() { setAdding(true); },
            }, h(Icon, { name: "plus", size: 14 }), " Register folder")
      )
    );
  }

  // --------------------------------------------------------------------------
  // FileTree — recursive file listing
  // --------------------------------------------------------------------------

  function FileTreeNode(props) {
    var entry      = props.entry;
    var workspaceId = props.workspaceId;
    var activeFile = props.activeFile;
    var onSelect   = props.onSelect;
    var depth      = props.depth || 0;

    var expanded = useState(depth < 2);
    var setExpanded = expanded[1]; expanded = expanded[0];
    var children    = useState(null); var setChildren = children[1]; children = children[0];

    function loadChildren() {
      if (!entry.is_dir) return;
      apiGet("/workspaces/" + workspaceId + "/files?rel=" + encodeURIComponent(entry.rel))
        .then(setChildren).catch(function() { setChildren([]); });
    }

    useEffect(function() {
      if (entry.is_dir && expanded && children === null) loadChildren();
    }, [expanded]); // eslint-disable-line

    var isActive = !entry.is_dir && activeFile === entry.rel;
    var itemClass = "hd-filetree__item" + (isActive ? " hd-active" : "");
    var indent = depth * 12;

    return h("li", null,
      h("div", {
        className: itemClass,
        style: { paddingLeft: (16 + indent) + "px" },
        onClick: function() {
          if (entry.is_dir) setExpanded(function(v) { return !v; });
          else onSelect(entry.rel);
        },
      },
        entry.is_dir
          ? h(Icon, { name: expanded ? "folder-open" : "folder", size: 13, color: "var(--hd-brand-green-deep)" })
          : h(Icon, { name: "file", size: 13, color: "var(--hd-muted)" }),
        h("span", { className: "hd-filetree__name" }, entry.name)
      ),
      entry.is_dir && expanded && children && h("ul", { className: "hd-filetree hd-filetree__subtree", style: { paddingLeft: 0 } },
        children.map(function(child) {
          return h(FileTreeNode, {
            key: child.rel,
            entry: child,
            workspaceId: workspaceId,
            activeFile: activeFile,
            onSelect: onSelect,
            depth: depth + 1,
          });
        })
      )
    );
  }

  function FileTree(props) {
    var files       = props.files;
    var workspaceId = props.workspaceId;
    var activeFile  = props.activeFile;
    var onSelect    = props.onSelect;

    if (!files || files.length === 0) {
      return h("div", { style: { padding: "16px", fontSize: 13, color: "var(--hd-muted)" } },
        "No files found."
      );
    }
    return h("ul", { className: "hd-filetree" },
      files.map(function(entry) {
        return h(FileTreeNode, {
          key: entry.rel,
          entry: entry,
          workspaceId: workspaceId,
          activeFile: activeFile,
          onSelect: onSelect,
          depth: 0,
        });
      })
    );
  }

  // --------------------------------------------------------------------------
  // WorkspaceDrawer
  // --------------------------------------------------------------------------

  function WorkspaceDrawer(props) {
    var workspace   = props.workspace;
    var files       = props.files;
    var activeFile  = props.activeFile;
    var pinned      = props.pinned;
    var open        = props.open;
    var onSelect    = props.onSelect;
    var onPin       = props.onPin;
    var onBack      = props.onBack;

    var modeClass = pinned ? "hd-drawer--docked" : "hd-drawer--overlay" + (open ? " hd-drawer--open" : "");

    return h("div", { className: "hd-drawer " + modeClass },
      h("div", { className: "hd-drawer__header" },
        h("span", { className: "hd-drawer__title", title: workspace.path },
          workspace.name
        ),
        h("div", { className: "hd-drawer__actions" },
          h("button", {
            className: "hd-btn hd-btn--icon" + (pinned ? " hd-active" : ""),
            title: pinned ? "Unpin drawer" : "Pin drawer open",
            onClick: onPin,
          }, h(Icon, { name: pinned ? "pin" : "pin-off", size: 14 })),
          h("button", {
            className: "hd-btn hd-btn--icon",
            title: "All workspaces",
            onClick: onBack,
          }, h(Icon, { name: "arrow-back", size: 14 }))
        )
      ),
      h("div", { className: "hd-drawer__body" },
        h(FileTree, {
          files: files,
          workspaceId: workspace.id,
          activeFile: activeFile,
          onSelect: onSelect,
        })
      )
    );
  }

  // --------------------------------------------------------------------------
  // EditorSurface
  // --------------------------------------------------------------------------

  function EditorSurface(props) {
    var content    = props.content;
    var mode       = props.mode;
    var onChange   = props.onChange;
    var activeFile = props.activeFile;

    if (!activeFile) {
      return h("div", { className: "hd-editor-empty" },
        h("div", { className: "hd-editor-empty__icon" }, "📄"),
        h("div", { className: "hd-editor-empty__text" },
          "Select a file from the workspace drawer to start editing."
        )
      );
    }

    if (mode === "preview") {
      return h("div", {
        className: "hd-editor-preview",
        dangerouslySetInnerHTML: { __html: renderMarkdown(content) },
      });
    }

    return h("textarea", {
      className: "hd-editor-textarea",
      value: content,
      onChange: function(e) { onChange(e.target.value); },
      spellCheck: mode !== "source",
      placeholder: "Start writing Markdown…",
    });
  }

  // --------------------------------------------------------------------------
  // SideChat
  // --------------------------------------------------------------------------

  function SideChat(props) {
    var messages    = props.messages;
    var onSend      = props.onSend;
    var visible     = props.visible;
    var onToggle    = props.onToggle;
    var sending     = props.sending;

    var inputRef    = useRef(null);
    var bottomRef   = useRef(null);

    useEffect(function() {
      if (bottomRef.current) bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    function handleKeyDown(e) {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        var val = inputRef.current ? inputRef.current.value.trim() : "";
        if (val) { onSend(val); if (inputRef.current) inputRef.current.value = ""; }
      }
    }

    function handleSend() {
      var val = inputRef.current ? inputRef.current.value.trim() : "";
      if (val) { onSend(val); if (inputRef.current) inputRef.current.value = ""; }
    }

    return h("div", { className: "hd-sidechat" + (visible ? "" : " hd-sidechat--hidden") },
      h("div", { className: "hd-sidechat__header" },
        h("span", { className: "hd-sidechat__title" }, "Side Chat"),
        h("button", { className: "hd-btn hd-btn--icon", title: "Close side chat", onClick: onToggle },
          h(Icon, { name: "close", size: 14 })
        )
      ),
      h("div", { className: "hd-sidechat__messages" },
        messages.length === 0
          ? h("div", { className: "hd-sidechat__empty" },
              "Ask questions, brainstorm, or request outline suggestions."
            )
          : messages.map(function(msg, i) {
              return h("div", { key: i, className: "hd-chat-msg hd-chat-msg--" + msg.role },
                h("div", { className: "hd-chat-msg__role" }, msg.role === "user" ? "You" : "Docs Agent"),
                h("div", { className: "hd-chat-msg__body" }, msg.content)
              );
            }),
        h("div", { ref: bottomRef })
      ),
      h("div", { className: "hd-sidechat__input-area" },
        h("textarea", {
          ref: inputRef,
          className: "hd-sidechat__textarea",
          placeholder: "Ask the Docs Agent…",
          onKeyDown: handleKeyDown,
          disabled: sending,
          rows: 3,
        }),
        h("div", { className: "hd-sidechat__actions" },
          h("span", { className: "hd-sidechat__hint" }, "⌘↵ to send"),
          h("button", {
            className: "hd-btn hd-btn--accent",
            onClick: handleSend,
            disabled: sending,
            style: { height: 28, padding: "0 12px", fontSize: 12 },
          }, sending ? "…" : h(Icon, { name: "send", size: 13 }), " Send")
        )
      )
    );
  }

  // --------------------------------------------------------------------------
  // CommandBar
  // --------------------------------------------------------------------------

  var COMMANDS = [
    { cmd: ":preview",  desc: "Toggle preview mode" },
    { cmd: ":save",     desc: "Save file" },
    { cmd: ":chat",     desc: "Open side chat" },
    { cmd: ":settings", desc: "Open settings" },
    { cmd: ":back",     desc: "Return to workspace list" },
  ];

  function CommandBar(props) {
    var onCommand   = props.onCommand;
    var activeFile  = props.activeFile;
    var dirty       = props.dirty;

    var input       = useState(""); var setInput = input[1]; input = input[0];
    var suggestions = useState([]); var setSuggestions = suggestions[1]; suggestions = suggestions[0];

    function handleChange(e) {
      var val = e.target.value;
      setInput(val);
      if (val.startsWith(":")) {
        setSuggestions(COMMANDS.filter(function(c) { return c.cmd.startsWith(val); }));
      } else {
        setSuggestions([]);
      }
    }

    function handleKeyDown(e) {
      if (e.key === "Enter") {
        e.preventDefault();
        onCommand(input.trim());
        setInput(""); setSuggestions([]);
      }
      if (e.key === "Escape") { setInput(""); setSuggestions([]); }
    }

    return h("div", { className: "hd-commandbar" },
      h("span", { className: "hd-commandbar__label" }, "⌘K"),
      h("div", { style: { flex: 1, position: "relative" } },
        h("input", {
          className: "hd-commandbar__input",
          type: "text",
          placeholder: activeFile
            ? "Type : for commands (e.g. :save, :preview) or describe a doc action…"
            : "Open a file to start editing",
          value: input,
          onChange: handleChange,
          onKeyDown: handleKeyDown,
        }),
        suggestions.length > 0 && h("div", {
          style: {
            position: "absolute", bottom: "100%", left: 0, right: 0,
            background: "var(--hd-canvas)", border: "1px solid var(--hd-hairline)",
            borderRadius: "var(--hd-r-md)", marginBottom: 4,
            boxShadow: "0 4px 12px rgba(0,0,0,0.08)", zIndex: 30,
          },
        },
          suggestions.map(function(s) {
            return h("div", {
              key: s.cmd,
              style: { padding: "8px 12px", cursor: "pointer", fontSize: 13 },
              onClick: function() { setInput(s.cmd); setSuggestions([]); },
            },
              h("span", { style: { fontFamily: "var(--hd-font-mono)", fontWeight: 500, marginRight: 8 } }, s.cmd),
              h("span", { style: { color: "var(--hd-steel)" } }, s.desc)
            );
          })
        )
      ),
      dirty && h("span", {
        style: { fontSize: 11, color: "var(--hd-stone)", flexShrink: 0, marginLeft: 8 },
      }, "Unsaved")
    );
  }

  // --------------------------------------------------------------------------
  // SettingsPanel (onboarding / status placeholder)
  // --------------------------------------------------------------------------

  function SettingsPanel(props) {
    var onClose = props.onClose;

    return h("div", { style: { flex: 1, overflow: "auto" } },
      h("div", { className: "hd-settings-section" },
        h("div", { style: { display: "flex", alignItems: "center", gap: 12, marginBottom: 8 } },
          h("button", { className: "hd-btn hd-btn--icon", onClick: onClose },
            h(Icon, { name: "arrow-back", size: 14 })
          ),
          h("h2", { className: "hd-settings-section__title", style: { margin: 0 } }, "Docs Settings")
        ),
        h("p", { className: "hd-settings-section__desc" },
          "Configure authentication, agent persona, and document conversion settings."
        ),

        h("div", { className: "hd-settings-row" },
          h("div", { className: "hd-settings-row__label" },
            h("div", { className: "hd-settings-row__key" }, "Codex OAuth"),
            h("div", { className: "hd-settings-row__hint" },
              "Provides the Docs Agent with Codex-backed reasoning. Brokered locally — browser never receives raw tokens."
            )
          ),
          h("span", { className: "hd-badge hd-badge--setup" }, "Set up in Hermes config")
        ),

        h("div", { className: "hd-settings-row" },
          h("div", { className: "hd-settings-row__label" },
            h("div", { className: "hd-settings-row__key" }, "Docs Agent Persona"),
            h("div", { className: "hd-settings-row__hint" },
              "A `docs` persona specialising in Markdown editing, review, and brainstorming."
            )
          ),
          h("span", { className: "hd-badge hd-badge--setup" }, "Coming in next slice")
        ),

        h("div", { className: "hd-settings-row" },
          h("div", { className: "hd-settings-row__label" },
            h("div", { className: "hd-settings-row__key" }, "Kordoc Document Conversion"),
            h("div", { className: "hd-settings-row__hint" },
              "HWP/HWPX/DOCX/XLSX/PDF → Markdown conversion via local Kordoc broker."
            )
          ),
          h("span", { className: "hd-badge hd-badge--setup" }, "Coming in next slice")
        ),

        h("div", { className: "hd-settings-row" },
          h("div", { className: "hd-settings-row__label" },
            h("div", { className: "hd-settings-row__key" }, "Storage"),
            h("div", { className: "hd-settings-row__hint" },
              "Workspace metadata stored in ~/.hermes/docs-workspaces/ — workspace source folders stay clean."
            )
          ),
          h("span", { className: "hd-badge hd-badge--ok" }, "Local")
        )
      )
    );
  }

  // --------------------------------------------------------------------------
  // DiffPreviewModal
  // --------------------------------------------------------------------------

  function DiffPreviewModal(props) {
    var diff    = props.diff;
    var onWrite = props.onWrite;
    var onClose = props.onClose;

    return h("div", { className: "hd-overlay" },
      h("div", { className: "hd-modal" },
        h("h3", { className: "hd-modal__title" }, "Preview changes"),
        h("div", { style: { fontSize: 12, color: "var(--hd-steel)", marginBottom: 4 } }, diff.rel),
        h("div", { className: "hd-modal__code" }, diff.proposed),
        h("div", { className: "hd-modal__actions" },
          h("button", { className: "hd-btn hd-btn--ghost", onClick: onClose }, "Cancel"),
          h("button", { className: "hd-btn hd-btn--accent", onClick: onWrite },
            h(Icon, { name: "check", size: 13 }), " Write file"
          )
        )
      )
    );
  }

  // --------------------------------------------------------------------------
  // Main HermesDocsApp component
  // --------------------------------------------------------------------------

  function HermesDocsApp() {
    // Global state
    var workspacesState  = useState(null); var workspaces = workspacesState[0]; var setWorkspaces = workspacesState[1];
    var loadingState     = useState(true);  var loading = loadingState[0]; var setLoading = loadingState[1];
    var errorState       = useState("");    var error = errorState[0]; var setError = errorState[1];

    // Active workspace
    var activeWsState    = useState(null); var activeWs = activeWsState[0]; var setActiveWs = activeWsState[1];
    var filesState       = useState([]);   var files = filesState[0]; var setFiles = filesState[1];

    // Drawer
    var pinnedState      = useState(false); var pinned = pinnedState[0]; var setPinned = pinnedState[1];
    var drawerOpenState  = useState(false); var drawerOpen = drawerOpenState[0]; var setDrawerOpen = drawerOpenState[1];

    // Editor
    var activeFileState  = useState(null); var activeFile = activeFileState[0]; var setActiveFile = activeFileState[1];
    var contentState     = useState(""); var content = contentState[0]; var setContent = contentState[1];
    var savedContentState = useState(""); var savedContent = savedContentState[0]; var setSavedContent = savedContentState[1];
    var editorModeState  = useState("edit"); var editorMode = editorModeState[0]; var setEditorMode = editorModeState[1];

    // Side chat
    var messagesState    = useState([]); var messages = messagesState[0]; var setMessages = messagesState[1];
    var sideChatOpenState = useState(false); var sideChatOpen = sideChatOpenState[0]; var setSideChatOpen = sideChatOpenState[1];
    var sendingState     = useState(false); var sending = sendingState[0]; var setSending = sendingState[1];

    // Settings
    var showSettingsState = useState(false); var showSettings = showSettingsState[0]; var setShowSettings = showSettingsState[1];

    // Diff preview
    var diffState        = useState(null); var diff = diffState[0]; var setDiff = diffState[1];

    var dirty = useMemo(function() { return content !== savedContent; }, [content, savedContent]);

    // ── Load workspaces on mount ──
    useEffect(function() {
      setLoading(true);
      apiGet("/workspaces")
        .then(function(data) { setWorkspaces(data || []); setLoading(false); })
        .catch(function(err) { setError(err.message || "Failed to load workspaces"); setLoading(false); });
    }, []);

    // ── Load root files when workspace changes ──
    useEffect(function() {
      if (!activeWs) return;
      apiGet("/workspaces/" + activeWs.id + "/files")
        .then(setFiles)
        .catch(function() { setFiles([]); });
    }, [activeWs && activeWs.id]); // eslint-disable-line

    // ── Load preferences when workspace changes ──
    useEffect(function() {
      if (!activeWs) return;
      apiGet("/workspaces/" + activeWs.id + "/preferences")
        .then(function(prefs) { setPinned(!!prefs.drawer_pinned); })
        .catch(function() {});
    }, [activeWs && activeWs.id]); // eslint-disable-line

    // ── Workspace actions ──
    function handleOpenWorkspace(ws) {
      apiPost("/workspaces/" + ws.id + "/open").catch(function() {});
      setActiveWs(ws);
      setActiveFile(null); setContent(""); setSavedContent("");
      setMessages([]); setShowSettings(false);
    }

    function handleAddWorkspace(name, path) {
      return apiPost("/workspaces", { name: name, path: path })
        .then(function(ws) { setWorkspaces(function(prev) { return (prev || []).concat([ws]); }); });
    }

    function handleRemoveWorkspace(id) {
      if (!window.confirm("Remove this workspace? (Hermes metadata will be detached but source files are unaffected.)")) return;
      apiDelete("/workspaces/" + id)
        .then(function() { setWorkspaces(function(prev) { return (prev || []).filter(function(w) { return w.id !== id; }); }); })
        .catch(function(err) { setError(err.message || "Failed to remove"); });
    }

    // ── File actions ──
    function handleSelectFile(rel) {
      if (dirty) {
        if (!window.confirm("You have unsaved changes. Discard and open this file?")) return;
      }
      setActiveFile(rel);
      apiGet("/workspaces/" + activeWs.id + "/file?rel=" + encodeURIComponent(rel))
        .then(function(data) { setContent(data.content || ""); setSavedContent(data.content || ""); })
        .catch(function(err) { setContent(""); setSavedContent(""); setError(err.message || "Failed to read file"); });
    }

    function handleSave() {
      if (!activeFile || !activeWs) return;
      apiPut("/workspaces/" + activeWs.id + "/file?rel=" + encodeURIComponent(activeFile),
        { content: content, preview: true })
        .then(function(d) { if (d && d.preview) setDiff(d); })
        .catch(function(err) { setError(err.message || "Preview failed"); });
    }

    function handleConfirmWrite() {
      if (!diff) return;
      apiPut("/workspaces/" + activeWs.id + "/file?rel=" + encodeURIComponent(activeFile),
        { content: content, preview: false })
        .then(function() { setSavedContent(content); setDiff(null); })
        .catch(function(err) { setError(err.message || "Write failed"); setDiff(null); });
    }

    // ── Pin drawer ──
    function handlePin() {
      var next = !pinned;
      setPinned(next);
      if (!activeWs) return;
      apiPut("/workspaces/" + activeWs.id + "/preferences", { drawer_pinned: next }).catch(function() {});
    }

    // ── Side chat ──
    function handleSend(text) {
      var msg = { role: "user", content: text };
      setMessages(function(prev) { return prev.concat([msg]); });
      setSending(true);
      apiPost("/workspaces/" + activeWs.id + "/sidechat", {
        content: text,
        document: activeFile || null,
      })
        .then(function(r) {
          setMessages(function(prev) { return prev.concat([{ role: "assistant", content: r.content }]); });
          setSending(false);
        })
        .catch(function(err) {
          setMessages(function(prev) { return prev.concat([{ role: "assistant", content: "Error: " + err.message }]); });
          setSending(false);
        });
    }

    // ── Command bar ──
    function handleCommand(cmd) {
      if (cmd === ":preview")   { setEditorMode(function(m) { return m === "preview" ? "edit" : "preview"; }); return; }
      if (cmd === ":save")      { handleSave(); return; }
      if (cmd === ":chat")      { setSideChatOpen(true); return; }
      if (cmd === ":settings")  { setShowSettings(true); return; }
      if (cmd === ":back")      { setActiveWs(null); return; }
      // Pass unrecognised commands to side chat
      if (cmd) { setSideChatOpen(true); handleSend(cmd); }
    }

    // ── Determine dark theme ──
    var themeAttr = "auto";

    // ─── Render ───
    if (!activeWs) {
      return h("div", { className: "hd-app", "data-theme": themeAttr },
        h(WorkspaceLauncher, {
          workspaces: workspaces || [],
          loading: loading,
          error: error,
          onOpen: handleOpenWorkspace,
          onAdd: handleAddWorkspace,
          onRemove: handleRemoveWorkspace,
        })
      );
    }

    return h("div", { className: "hd-app", "data-theme": themeAttr },
      h("div", { className: "hd-shell" },

        // ── Rail ──
        h("div", { className: "hd-rail" },
          h("button", {
            className: "hd-rail__btn" + (drawerOpen || pinned ? " hd-active" : ""),
            title: pinned ? "Workspace drawer (pinned)" : (drawerOpen ? "Close drawer" : "Open drawer"),
            onClick: function() { if (!pinned) setDrawerOpen(function(v) { return !v; }); },
          }, h(Icon, { name: "folder", size: 15 })),

          h("div", { className: "hd-rail__divider" }),

          h("button", {
            className: "hd-rail__btn" + (sideChatOpen ? " hd-active" : ""),
            title: "Side Chat",
            onClick: function() { setSideChatOpen(function(v) { return !v; }); },
          }, h(Icon, { name: "docs", size: 15 })),

          h("div", { className: "hd-rail__spacer" }),

          h("button", {
            className: "hd-rail__btn" + (showSettings ? " hd-active" : ""),
            title: "Settings",
            onClick: function() { setShowSettings(function(v) { return !v; }); },
          }, h(Icon, { name: "settings", size: 15 })),

          h("button", {
            className: "hd-rail__btn",
            title: "All workspaces",
            onClick: function() { setActiveWs(null); },
          }, h(Icon, { name: "home", size: 15 }))
        ),

        // ── Workspace area ──
        h("div", { className: "hd-workspace" },

          // Drawer
          h(WorkspaceDrawer, {
            workspace: activeWs,
            files: files,
            activeFile: activeFile,
            pinned: pinned,
            open: drawerOpen,
            onSelect: function(rel) { setDrawerOpen(false); handleSelectFile(rel); },
            onPin: handlePin,
            onBack: function() { setActiveWs(null); },
          }),

          // Backdrop for overlay drawer
          !pinned && drawerOpen && h("div", {
            style: {
              position: "absolute", inset: 0, zIndex: 10,
              cursor: "default",
            },
            onClick: function() { setDrawerOpen(false); },
          }),

          // Editor area
          h("div", { className: "hd-editor-area" },

            // Editor header
            showSettings ? null : h("div", { className: "hd-editor-header" },
              h("div", { className: "hd-editor-header__breadcrumb" },
                activeFile
                  ? h("span", { className: "hd-editor-header__filename" }, activeFile.split("/").pop())
                  : h("span", null, activeWs.name)
              ),
              activeFile && h("div", { className: "hd-editor-mode-tabs" },
                ["edit", "preview", "source"].map(function(m) {
                  return h("button", {
                    key: m,
                    className: "hd-editor-mode-tab" + (editorMode === m ? " hd-active" : ""),
                    onClick: function() { setEditorMode(m); },
                  }, m.charAt(0).toUpperCase() + m.slice(1));
                })
              )
            ),

            // Editor body
            showSettings
              ? h(SettingsPanel, { onClose: function() { setShowSettings(false); } })
              : h("div", { className: "hd-editor-body" },
                  h(EditorSurface, {
                    content: content,
                    mode: editorMode,
                    onChange: setContent,
                    activeFile: activeFile,
                  })
                ),

            // Command bar
            !showSettings && h(CommandBar, {
              onCommand: handleCommand,
              activeFile: activeFile,
              dirty: dirty,
            })
          ),

          // Side chat
          h(SideChat, {
            messages: messages,
            onSend: handleSend,
            visible: sideChatOpen,
            onToggle: function() { setSideChatOpen(function(v) { return !v; }); },
            sending: sending,
          })
        )
      ),

      // Diff preview modal
      diff && h(DiffPreviewModal, {
        diff: diff,
        onWrite: handleConfirmWrite,
        onClose: function() { setDiff(null); },
      }),

      // Error toast (auto-dismiss)
      error && h("div", {
        className: "hd-banner hd-banner--error",
        style: {
          position: "absolute", bottom: 48, left: "50%", transform: "translateX(-50%)",
          zIndex: 200, whiteSpace: "nowrap", boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
          cursor: "pointer",
        },
        onClick: function() { setError(""); },
      }, h(Icon, { name: "warn", size: 13 }), " ", error, " ✕")
    );
  }

  // --------------------------------------------------------------------------
  // Register with dashboard
  // --------------------------------------------------------------------------

  if (window.__HERMES_PLUGINS__ && typeof window.__HERMES_PLUGINS__.register === "function") {
    window.__HERMES_PLUGINS__.register("hermes-docs", HermesDocsApp);
  } else {
    console.error("[hermes-docs] __HERMES_PLUGINS__.register not available");
  }

}());

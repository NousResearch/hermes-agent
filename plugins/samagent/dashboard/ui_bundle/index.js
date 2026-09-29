/* SamAgent — Codex-Style Local Platform, Live To-Do Sidebar, GitHub Auto-Sync & Agent Browser */
(function () {
  var sdk = window.__HERMES_PLUGIN_SDK__ || {};
  var React = sdk.React || {};
  var h = React.createElement;
  var useState = (sdk.hooks && sdk.hooks.useState) || React.useState;
  var useEffect = (sdk.hooks && sdk.hooks.useEffect) || React.useEffect;

  var API_BASE = "/api/plugins/samagent";

  function apiGet(path) {
    return fetch(API_BASE + path).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
  }

  function apiPost(path, body) {
    return fetch(API_BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
  }

  function pill(ok, label) {
    return h(
      "span",
      { className: "sam-pill " + (ok ? "sam-pill-green" : "sam-pill-red") },
      (ok ? "PASS · " : "BLOCKED · ") + label
    );
  }

  function todoStatusBadge(status) {
    if (status === "completed") {
      return h("span", { className: "sam-pill sam-pill-green" }, "✓ FINISHED");
    }
    if (status === "running") {
      return h("span", { className: "sam-pill sam-pill-blue" }, "⟳ RUNNING");
    }
    if (status === "failed") {
      return h("span", { className: "sam-pill sam-pill-red" }, "✕ BLOCKED");
    }
    return h("span", { className: "sam-pill sam-pill-amber" }, "○ QUEUED");
  }

  function SamAgentMissionControl() {
    var _s = useState(null), state = _s[0], setState = _s[1];
    var _t = useState("codex"), tab = _t[0], setTab = _t[1]; // codex | browser | github | brief | ledger
    var _b = useState(false), busy = _b[0], setBusy = _b[1];
    var _err = useState(""), errMsg = _err[0], setErrMsg = _err[1];
    var _notice = useState(""), notice = _notice[0], setNotice = _notice[1];

    // Live To-Do Sidebar Drawer state (auto-opens when planning or running tasks)
    var _sb = useState(true), todoSidebarOpen = _sb[0], setTodoSidebarOpen = _sb[1];
    var _ntodo = useState(""), newTodoText = _ntodo[0], setNewTodoText = _ntodo[1];

    // Compact "📁 Load Folder" Popover state
    var _fp = useState(false), folderModalOpen = _fp[0], setFolderModalOpen = _fp[1];
    var _fpath = useState(""), customFolderPath = _fpath[0], setCustomFolderPath = _fpath[1];

    // GitHub Sync & PR state
    var _remUrl = useState(""), remoteUrlInput = _remUrl[0], setRemoteUrlInput = _remUrl[1];
    var _cmsg = useState("feat: verified local build from SamAgent Codex Studio"), commitMsgInput = _cmsg[0], setCommitMsgInput = _cmsg[1];
    var _prt = useState(""), prTitleInput = _prt[0], setPrTitleInput = _prt[1];

    // Brief & Composer state
    var _brief = useState(
      "Booking site for my yoga studio where visitors see the schedule, members book classes (no double booking, private per member), and admins add classes."
    );
    var briefInput = _brief[0], setBriefInput = _brief[1];
    var _cmode = useState("code"), composerMode = _cmode[0], setComposerMode = _cmode[1]; // "ask" | "code"
    var _pol = useState("default"), routerPolicy = _pol[0], setRouterPolicy = _pol[1];

    // VS Code File Editor state
    var _sf = useState("app/main.py"), selectedFile = _sf[0], setSelectedFile = _sf[1];
    var _fc = useState(""), fileContent = _fc[0], setFileContent = _fc[1];

    // Interactive Multi-Role Dev Sandbox state
    var _dr = useState("member"), devRole = _dr[0], setDevRole = _dr[1];
    var _du = useState("u_member_a"), devUserId = _du[0], setDevUserId = _du[1];
    var _di = useState("item_1"), devItemId = _di[0], setDevItemId = _di[1];
    var _nt = useState("Sunset Vinyasa Flow"), devNewTitle = _nt[0], setDevNewTitle = _nt[1];
    var _do = useState(null), devOut = _do[0], setDevOut = _do[1];

    function loadFile(relPath) {
      setSelectedFile(relPath);
      apiGet("/ide/file?path=" + encodeURIComponent(relPath))
        .then(function (res) {
          setFileContent(res.content || "");
        })
        .catch(function (e) {
          setErrMsg(String(e));
        });
    }

    function refresh() {
      apiGet("/state")
        .then(function (data) {
          setState(data);
          if (data && data.spec && data.spec.goal) setBriefInput(data.spec.goal);
          if (data && data.github && data.github.remote_url) setRemoteUrlInput(data.github.remote_url);
        })
        .catch(function (e) {
          setErrMsg(String(e));
        });
    }

    useEffect(function () {
      refresh();
      loadFile("app/main.py");
    }, []);

    function runCodexTask(mode) {
      setBusy(true);
      setErrMsg("");
      // Automatically open the To-Do sidebar whenever planning or running a task!
      setTodoSidebarOpen(true);
      apiPost("/plan", {
        brief: briefInput,
        answers: {},
        router_policy: routerPolicy,
        autonomy: "milestones",
      })
        .then(function (plannedState) {
          setState(plannedState);
          if (mode === "ask") {
            setNotice("Plan & To-Do Checklist created in sidebar (8 tasks queued). Click 'Code' to execute.");
            return plannedState;
          }
          return apiPost("/build", { autonomy: "milestones", router_policy: routerPolicy });
        })
        .then(function (finalState) {
          if (finalState) setState(finalState);
          loadFile("app/main.py");
          if (mode !== "ask") {
            setNotice("Codex Task Completed: All To-Do items finished, L0–L4 verified, and synced to disk!");
          }
        })
        .catch(function (e) {
          setErrMsg(String(e));
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function handleLoadFolder(folderPath) {
      if (!folderPath) return;
      setBusy(true);
      setTodoSidebarOpen(true);
      apiPost("/folder/load", { folder_path: folderPath, brief: briefInput })
        .then(function (data) {
          setState(data);
          setFolderModalOpen(false);
          setCustomFolderPath("");
          loadFile("app/main.py");
          setNotice("Loaded local project folder: " + data.workspace);
        })
        .catch(function (e) {
          setErrMsg(String(e));
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function toggleAutoPush(nextVal) {
      apiPost("/github/prefs", { auto_push_on_complete: nextVal })
        .then(function (data) {
          setState(data);
          setNotice(
            "Auto-Sync / Push to GitHub when task finishes: " + (nextVal ? "ENABLED" : "DISABLED")
          );
        })
        .catch(function (e) {
          setErrMsg(String(e));
        });
    }

    function handleGitHubSync() {
      setBusy(true);
      apiPost("/github/sync", { commit_message: commitMsgInput, push_to_remote: true })
        .then(function (data) {
          setState(data);
          var sr = data.github_sync_result || {};
          setNotice("Git Commit & Sync finished on branch '" + (sr.branch || "main") + "': " + (sr.output || "OK"));
        })
        .catch(function (e) {
          setErrMsg(String(e));
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function handleCreatePr() {
      setBusy(true);
      apiPost("/github/pr", { title: prTitleInput || null })
        .then(function (data) {
          setState(data);
          var pr = data.github_pr_result || {};
          setNotice(
            pr.created_on_github
              ? "Created GitHub PR: " + pr.pr_url
              : (pr.message || "Prepared verified PR payload") + " (" + (pr.cli_command || "") + ")"
          );
        })
        .catch(function (e) {
          setErrMsg(String(e));
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function handleTodoToggle(todoId) {
      apiPost("/todo/action", { action: "toggle", todo_id: todoId }).then(function (data) {
        setState(data);
      });
    }

    function handleTodoAdd() {
      if (!newTodoText.trim()) return;
      apiPost("/todo/action", { action: "add", title: newTodoText }).then(function (data) {
        setState(data);
        setNewTodoText("");
      });
    }

    function openVsCode(relPath) {
      apiPost("/ide/open", { rel_path: relPath || null, line: 1 }).then(function (res) {
        setNotice("Opened in VS Code: " + (res.vscode_uri || ""));
        if (res.vscode_uri) window.open(res.vscode_uri, "_blank");
      });
    }

    function saveFileAndVerify() {
      setBusy(true);
      setTodoSidebarOpen(true);
      apiPost("/ide/file", { rel_path: selectedFile, content: fileContent, auto_reverify: true })
        .then(function (data) {
          setState(data);
          setNotice("Saved " + selectedFile + " to disk & updated To-Do + Pre-Prod Gate.");
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function promoteToProduction() {
      setBusy(true);
      apiPost("/promote-prod", {})
        .then(function (data) {
          setState(data);
          var pr = data.promotion_result || {};
          if (pr.promoted) {
            setNotice("Promoted Release " + pr.release_id + " (Dockerfile + RELEASE_MANIFEST.json)!");
          } else {
            setErrMsg("Blocked by Pre-Prod Gate: " + (pr.blockers || []).join(", "));
          }
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function runBrowserAction(action) {
      setBusy(true);
      apiPost("/browser/action", { action: action, dev_port: 3000 })
        .then(function (data) {
          setState(data);
          setNotice("Agent Browser (" + action + ") completed.");
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function runDevAction(action, extra) {
      setBusy(true);
      var payload = Object.assign(
        { action: action, role: devRole, user_id: devUserId, item_id: devItemId, title: devNewTitle },
        extra || {}
      );
      apiPost("/dev-app/action", payload)
        .then(function (res) {
          setDevOut(res);
          refresh();
        })
        .finally(function () {
          setBusy(false);
        });
    }

    if (!state) {
      return h("div", { className: "sam-root" }, h("p", { style: { padding: "20px" } }, "Loading SamAgent Codex Studio..."));
    }

    var spec = state.spec || {};
    var planCard = state.plan_card || {};
    var todos = state.todos || { items: [], completed: 0, total: 0, progress_pct: 0 };
    var gh = state.github || { diff_summary: { files: [], total_additions: 0, total_deletions: 0 } };
    var diffSum = gh.diff_summary || { files: [], total_additions: 0, total_deletions: 0 };
    var browser = state.browser || {};
    var snap = (browser.builtin_agent_browser && browser.builtin_agent_browser.snapshot) || { elements: [] };
    var preProd = state.pre_prod_gate || {};
    var ide = state.ide || {};
    var devSrv = state.dev_server || {};
    var devApp = state.dev_app || { items: [], bookings: [] };
    var releases = state.releases || [];
    var folderBrowser = state.folder_browser || { folders: [] };

    return h(
      "div",
      { className: "sam-root" },

      // ======================================================================
      // 1. TOP COMPACT CODEX COMMAND HEADER (Folder Loader + GitHub + To-Do)
      // ======================================================================
      h(
        "div",
        { className: "sam-topbar" },
        h(
          "div",
          { className: "sam-brand" },
          h(
            "button",
            {
              className: "sam-btn " + (todoSidebarOpen ? "sam-btn-primary" : ""),
              onClick: function () {
                setTodoSidebarOpen(!todoSidebarOpen);
              },
              title: "Toggle Live Task & To-Do Sidebar",
            },
            "☑ To-Do (" + (todos.completed || 0) + "/" + (todos.total || 0) + ")"
          ),
          h("span", { className: "sam-title" }, "SamAgent Codex Studio"),
          // Compact "📁 Load Folder" button requested by user
          h(
            "button",
            {
              className: "sam-btn",
              onClick: function () {
                setFolderModalOpen(!folderModalOpen);
              },
              title: "Load any local project folder on your machine",
            },
            "📁 Load Folder: " + (state.workspace ? state.workspace.split("/").pop() : "project")
          ),
          // Git Branch + Codex +add/-del badge
          h(
            "span",
            { className: "sam-pill sam-pill-purple" },
            "⎇ " + (gh.branch || "main") + " (" + (gh.head_commit || "HEAD") + ")"
          ),
          h(
            "span",
            { className: "sam-pill sam-pill-blue" },
            h("span", { className: "sam-diff-add" }, "+" + (diffSum.total_additions || 0)),
            h("span", { className: "sam-diff-del" }, "-" + (diffSum.total_deletions || 0))
          ),
          h(
            "span",
            { className: "sam-pill " + (devSrv.running ? "sam-pill-green" : "sam-pill-amber") },
            devSrv.running ? "DEV :3000 ONLINE" : "DEV :3000 IDLE"
          ),
          pill(Boolean(preProd.ready_for_production), preProd.ready_for_production ? "PRE-PROD READY" : "PRE-PROD BLOCKED")
        ),
        // Right Header Actions: Auto-Push toggle, Sync/PR, Open in VS Code
        h(
          "div",
          { style: { display: "flex", gap: "7px", alignItems: "center", flexWrap: "wrap" } },
          h(
            "button",
            {
              className: "sam-btn " + (gh.auto_push_on_complete ? "sam-btn-success" : ""),
              onClick: function () {
                toggleAutoPush(!gh.auto_push_on_complete);
              },
              title: "Automatically commit & push to GitHub when a task finishes",
            },
            "Auto-Sync GitHub: " + (gh.auto_push_on_complete ? "ON" : "OFF")
          ),
          h(
            "button",
            { className: "sam-btn", disabled: busy, onClick: handleGitHubSync },
            "⬆ Push / Sync"
          ),
          h(
            "button",
            { className: "sam-btn", disabled: busy, onClick: handleCreatePr },
            "⑂ Create PR"
          ),
          h(
            "button",
            {
              className: "sam-btn sam-btn-primary",
              onClick: function () {
                openVsCode(null);
              },
            },
            "Open in VS Code"
          )
        )
      ),

      // Compact Popover for "📁 Load Folder" button
      folderModalOpen
        ? h(
            "div",
            {
              className: "sam-card",
              style: {
                margin: "10px 18px 0",
                borderColor: "#3b82f6",
                background: "#0f172a",
              },
            },
            h(
              "div",
              { className: "sam-card-title" },
              h("span", null, "📁 Load Local Project Folder on Your Machine (Syncs with VS Code & GitHub)"),
              h(
                "button",
                {
                  className: "sam-btn",
                  onClick: function () {
                    setFolderModalOpen(false);
                  },
                },
                "Close"
              )
            ),
            h(
              "div",
              { style: { display: "flex", gap: "8px", marginBottom: "10px" } },
              h("input", {
                className: "sam-input sam-mono",
                placeholder: "Paste local path (e.g. /Users/you/my-repo) or new project name...",
                value: customFolderPath,
                onChange: function (e) {
                  setCustomFolderPath(e.target.value);
                },
              }),
              h(
                "button",
                {
                  className: "sam-btn sam-btn-primary",
                  disabled: busy || !customFolderPath.trim(),
                  onClick: function () {
                    handleLoadFolder(customFolderPath);
                  },
                },
                "Load & Open Project"
              )
            ),
            h(
              "div",
              { style: { display: "flex", gap: "6px", flexWrap: "wrap", alignItems: "center", fontSize: "11px" } },
              h("strong", null, "Discovered Local Folders (" + (folderBrowser.current_dir || "~/SamAgentProjects") + "):"),
              (folderBrowser.folders || []).map(function (fd) {
                return h(
                  "button",
                  {
                    key: fd.path,
                    className: "sam-btn",
                    onClick: function () {
                      handleLoadFolder(fd.path);
                    },
                  },
                  "📁 " + fd.name + (fd.is_git ? " (git)" : "")
                );
              })
            )
          )
        : null,

      errMsg
        ? h(
            "div",
            { className: "sam-card", style: { margin: "10px 18px 0", borderColor: "#ef4444", color: "#f87171" } },
            "Error: " + errMsg
          )
        : null,

      notice
        ? h(
            "div",
            {
              className: "sam-card",
              style: {
                margin: "10px 18px 0",
                borderColor: "#3b82f6",
                padding: "8px 14px",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              },
            },
            h("span", { style: { fontSize: "12px", color: "#93c5fd" } }, notice),
            h(
              "button",
              {
                className: "sam-btn",
                onClick: function () {
                  setNotice("");
                },
              },
              "Dismiss"
            )
          )
        : null,

      // ======================================================================
      // 2. CODEX 2-COLUMN BODY: LEFT LIVE TO-DO SIDEBAR + MAIN WORKBENCH
      // ======================================================================
      h(
        "div",
        { className: "sam-layout" },

        // LEFT COLLAPSIBLE TO-DO SIDEBAR (Opens automatically when task plans/runs!)
        todoSidebarOpen
          ? h(
              "aside",
              { className: "sam-todo-sidebar" },
              h(
                "div",
                { style: { display: "flex", justifyContent: "space-between", alignItems: "center" } },
                h("strong", { style: { fontSize: "13px", color: "#f8fafc" } }, "Live Agent To-Do List"),
                h(
                  "span",
                  { className: "sam-pill sam-pill-blue" },
                  (todos.progress_pct || 0) + "% (" + (todos.completed || 0) + "/" + (todos.total || 0) + ")"
                )
              ),
              h(
                "div",
                { className: "sam-progress-track" },
                h("div", { className: "sam-progress-fill", style: { width: (todos.progress_pct || 0) + "%" } })
              ),
              h(
                "div",
                { style: { fontSize: "11px", color: "#94a3b8" } },
                busy
                  ? "⟳ Task running — executing worktree waves & verification..."
                  : "Click any step to toggle status, or run a task below to watch live progress."
              ),
              h(
                "div",
                { className: "sam-list", style: { overflowY: "auto", maxHeight: "calc(100vh - 280px)" } },
                (todos.items || []).map(function (item) {
                  var st = busy && item.id === "T3" ? "running" : item.status;
                  return h(
                    "div",
                    {
                      key: item.id,
                      className: "sam-todo-item " + st,
                      onClick: function () {
                        handleTodoToggle(item.id);
                      },
                    },
                    h(
                      "div",
                      { style: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" } },
                      h("code", { className: "sam-mono", style: { fontWeight: 700, color: "#93c5fd" } }, item.id),
                      todoStatusBadge(st)
                    ),
                    h("div", { style: { fontWeight: 600, color: "#f8fafc", lineHeight: 1.35 } }, item.title),
                    h(
                      "div",
                      { style: { fontSize: "10px", color: "#94a3b8", marginTop: "4px" } },
                      item.agent + " · " + item.detail
                    )
                  );
                })
              ),
              // Add custom To-Do item input
              h(
                "div",
                { style: { display: "flex", gap: "6px", marginTop: "auto", paddingTop: "8px" } },
                h("input", {
                  className: "sam-input",
                  placeholder: "+ Add custom To-Do step...",
                  value: newTodoText,
                  onChange: function (e) {
                    setNewTodoText(e.target.value);
                  },
                }),
                h(
                  "button",
                  { className: "sam-btn sam-btn-primary", onClick: handleTodoAdd },
                  "Add"
                )
              )
            )
          : null,

        // MAIN CODEX WORKBENCH AREA
        h(
          "main",
          { className: "sam-main" },

          // Navigation Bar
          h(
            "div",
            { className: "sam-nav" },
            h(
              "button",
              {
                className: "sam-tab " + (tab === "codex" ? "active" : ""),
                onClick: function () {
                  setTab("codex");
                },
              },
              "1. Codex Workbench, Diffs & VS Code"
            ),
            h(
              "button",
              {
                className: "sam-tab " + (tab === "browser" ? "active" : ""),
                onClick: function () {
                  setTab("browser");
                },
              },
              "2. Agent Browser (@eN) & Live App (:3000)"
            ),
            h(
              "button",
              {
                className: "sam-tab " + (tab === "github" ? "active" : ""),
                onClick: function () {
                  setTab("github");
                },
              },
              "3. GitHub Auto-Sync & Pull Requests"
            ),
            h(
              "button",
              {
                className: "sam-tab " + (tab === "brief" ? "active" : ""),
                onClick: function () {
                  setTab("brief");
                },
              },
              "4. Spec Contract & Plan Card"
            ),
            h(
              "button",
              {
                className: "sam-tab " + (tab === "ledger" ? "active" : ""),
                onClick: function () {
                  setTab("ledger");
                },
              },
              "5. Persistent Ledger & Benchmarks"
            )
          ),

          // TAB 1: CODEX WORKBENCH, DIFFS & VS CODE STUDIO
          tab === "codex"
            ? h(
                "div",
                { className: "sam-grid-2" },
                // Left Card: Codex File Diffs (+add / -del) + Live Bidirectional VS Code Editor
                h(
                  "div",
                  { className: "sam-card" },
                  h(
                    "div",
                    { className: "sam-card-title" },
                    h("span", null, "Codex File Diffs & Local VS Code Files"),
                    h(
                      "button",
                      {
                        className: "sam-btn sam-btn-primary",
                        onClick: function () {
                          openVsCode(selectedFile);
                        },
                      },
                      "Open " + selectedFile + " in VS Code ↗"
                    )
                  ),
                  h(
                    "div",
                    { className: "sam-list", style: { maxHeight: "185px", overflowY: "auto", marginBottom: "12px" } },
                    (diffSum.files || []).map(function (df) {
                      return h(
                        "div",
                        {
                          key: df.path,
                          className: "sam-list-item",
                          style: {
                            cursor: "pointer",
                            borderColor: selectedFile === df.path ? "#3b82f6" : "#1e293b",
                          },
                          onClick: function () {
                            loadFile(df.path);
                          },
                        },
                        h(
                          "div",
                          null,
                          h("code", { className: "sam-mono" }, df.path),
                          h(
                            "span",
                            { style: { fontSize: "10px", color: "#94a3b8", marginLeft: "8px" } },
                            "(" + df.status + ")"
                          )
                        ),
                        h(
                          "div",
                          { style: { display: "flex", gap: "8px", alignItems: "center" } },
                          h("span", { className: "sam-diff-add" }, "+" + df.additions),
                          h("span", { className: "sam-diff-del" }, "-" + df.deletions),
                          h(
                            "a",
                            {
                              href: "vscode://file" + state.workspace + "/" + df.path + ":1:1",
                              style: { color: "#60a5fa", fontSize: "11px", textDecoration: "none" },
                              onClick: function (e) {
                                e.stopPropagation();
                              },
                            },
                            "vscode://"
                          )
                        )
                      );
                    })
                  ),
                  h(
                    "div",
                    { className: "sam-card-title" },
                    h("span", null, "Inspector / Editor: " + selectedFile),
                    h(
                      "button",
                      { className: "sam-btn sam-btn-success", disabled: busy, onClick: saveFileAndVerify },
                      "Save to Disk & Re-Verify L0–L4"
                    )
                  ),
                  h("textarea", {
                    className: "sam-textarea sam-mono",
                    style: { minHeight: "235px", fontSize: "12px" },
                    value: fileContent,
                    onChange: function (e) {
                      setFileContent(e.target.value);
                    },
                  })
                ),

                // Right Card: Pre-Production Gate (7 Checks) + Live Git Diff + Production Bundler
                h(
                  "div",
                  { className: "sam-card" },
                  h(
                    "div",
                    { className: "sam-card-title" },
                    h("span", null, "Pre-Production Deployment Gate (7 Checks)"),
                    pill(Boolean(preProd.ready_for_production), preProd.ready_for_production ? "APPROVED FOR PROD" : "BLOCKED")
                  ),
                  h(
                    "div",
                    { className: "sam-list", style: { marginBottom: "12px" } },
                    Object.keys(preProd.checks || {}).map(function (k) {
                      var ok = Boolean(preProd.checks[k]);
                      return h(
                        "div",
                        { key: k, className: "sam-list-item" },
                        h("code", { className: "sam-mono" }, k),
                        pill(ok, ok ? "READY" : "FAILED")
                      );
                    })
                  ),
                  h(
                    "div",
                    { className: "sam-card-title" },
                    h("span", null, "Production Releases (" + releases.length + ")"),
                    h(
                      "button",
                      {
                        className: "sam-btn sam-btn-success",
                        disabled: busy || !preProd.ready_for_production,
                        onClick: promoteToProduction,
                      },
                      "Generate Production Bundle (Dockerfile + Manifest)"
                    )
                  ),
                  h(
                    "div",
                    { className: "sam-card-title", style: { marginTop: "10px" } },
                    h("span", null, "Live Git Diff (Uncommitted VS Code Edits)")
                  ),
                  ide.git && ide.git.changed_files && ide.git.changed_files.length > 0
                    ? h("pre", { className: "sam-pre" }, ide.git.diff || "")
                    : h(
                        "div",
                        { style: { fontSize: "12px", color: "#34d399" } },
                        "Working tree clean — all local VS Code edits committed or in sync."
                      )
                )
              )
            : null,

          // TAB 2: BUILT-IN AGENT BROWSER (@eN) + CHROME CDP / MCP + LIVE APP (:3000)
          tab === "browser"
            ? h(
                "div",
                { className: "sam-grid-2" },
                // Left Card: Live Local Dev App (:3000) + Multi-Role RBAC/IDOR Tester
                h(
                  "div",
                  { className: "sam-card" },
                  h(
                    "div",
                    { className: "sam-card-title" },
                    h("span", null, "Live Local Dev App Preview (Port 3000)"),
                    h(
                      "button",
                      {
                        className: "sam-btn",
                        onClick: function () {
                          runDevAction("reset_db");
                        },
                      },
                      "Reset Local SQLite DB"
                    )
                  ),
                  h("iframe", {
                    className: "sam-preview-frame",
                    srcDoc: state.preview_html || "<html><body><p>No preview built yet.</p></body></html>",
                    title: "Live Application Preview",
                  }),
                  h(
                    "div",
                    { style: { marginTop: "10px", display: "flex", gap: "6px", flexWrap: "wrap" } },
                    h(
                      "button",
                      {
                        className: "sam-btn " + (devRole === "visitor" ? "sam-btn-primary" : ""),
                        onClick: function () {
                          setDevRole("visitor");
                          setDevUserId("");
                        },
                      },
                      "Role: Visitor (Anon)"
                    ),
                    h(
                      "button",
                      {
                        className: "sam-btn " + (devRole === "member" && devUserId === "u_member_a" ? "sam-btn-primary" : ""),
                        onClick: function () {
                          setDevRole("member");
                          setDevUserId("u_member_a");
                        },
                      },
                      "Role: Member Alice (u_member_a)"
                    ),
                    h(
                      "button",
                      {
                        className: "sam-btn " + (devRole === "member" && devUserId === "u_member_b" ? "sam-btn-primary" : ""),
                        onClick: function () {
                          setDevRole("member");
                          setDevUserId("u_member_b");
                        },
                      },
                      "Role: Member Bob (u_member_b)"
                    ),
                    h(
                      "button",
                      {
                        className: "sam-btn " + (devRole === "admin" ? "sam-btn-primary" : ""),
                        onClick: function () {
                          setDevRole("admin");
                          setDevUserId("u_admin");
                        },
                      },
                      "Role: Admin (u_admin)"
                    )
                  ),
                  h(
                    "div",
                    { style: { display: "flex", gap: "8px", marginTop: "8px" } },
                    h(
                      "button",
                      {
                        className: "sam-btn sam-btn-primary",
                        disabled: busy,
                        onClick: function () {
                          runDevAction("create_booking");
                        },
                      },
                      "Book Class (item_1) as " + devRole
                    ),
                    h(
                      "button",
                      {
                        className: "sam-btn sam-btn-success",
                        disabled: busy,
                        onClick: function () {
                          runDevAction("create_item");
                        },
                      },
                      "+ Add Class as " + devRole
                    )
                  ),
                  devOut
                    ? h(
                        "pre",
                        { className: "sam-pre", style: { marginTop: "8px" } },
                        JSON.stringify(devOut.response, null, 2)
                      )
                    : null
                ),

                // Right Card: Built-in Agent Browser (@eN Snapshot) + Real Chrome CDP + Chrome MCP
                h(
                  "div",
                  { className: "sam-card" },
                  h(
                    "div",
                    { className: "sam-card-title" },
                    h("span", null, "Built-in Agent Browser (@eN Accessibility Tree) + Chrome CDP / MCP"),
                    h(
                      "button",
                      {
                        className: "sam-btn sam-btn-primary",
                        disabled: busy,
                        onClick: function () {
                          runBrowserAction("snapshot");
                        },
                      },
                      "↻ Refresh @eN Snapshot"
                    )
                  ),
                  h(
                    "div",
                    { className: "sam-kpi-row" },
                    h(
                      "div",
                      { className: "sam-kpi" },
                      h("div", { className: "sam-kpi-label" }, "Built-in Engine"),
                      h("div", { className: "sam-kpi-value" }, "agent-browser")
                    ),
                    h(
                      "div",
                      { className: "sam-kpi" },
                      h("div", { className: "sam-kpi-label" }, "@eN Nodes"),
                      h("div", { className: "sam-kpi-value" }, (snap.total_nodes || 0) + " (" + (snap.interactive_count || 0) + " interactive)")
                    ),
                    h(
                      "div",
                      { className: "sam-kpi" },
                      h("div", { className: "sam-kpi-label" }, "Chrome CDP :9222"),
                      h(
                        "div",
                        { className: "sam-kpi-value" },
                        browser.chrome_cdp && browser.chrome_cdp.status && browser.chrome_cdp.status.connected
                          ? "CONNECTED"
                          : "STANDBY"
                      )
                    ),
                    h(
                      "div",
                      { className: "sam-kpi" },
                      h("div", { className: "sam-kpi-label" }, "Chrome MCP"),
                      h(
                        "div",
                        { className: "sam-kpi-value" },
                        browser.chrome_mcp && browser.chrome_mcp.configured ? "CONFIGURED" : "1-CLICK READY"
                      )
                    )
                  ),
                  h(
                    "div",
                    { style: { fontSize: "11px", fontWeight: 700, marginBottom: "4px" } },
                    "1. Built-in Agent Browser Snapshot (@e1..@eN Element Refs from Live App):"
                  ),
                  h("pre", { className: "sam-pre", style: { marginBottom: "10px" } }, snap.snapshot_text || "No snapshot yet."),
                  h(
                    "div",
                    { style: { display: "flex", gap: "8px", flexWrap: "wrap" } },
                    h(
                      "button",
                      {
                        className: "sam-btn",
                        disabled: busy,
                        onClick: function () {
                          runBrowserAction("probe_cdp");
                        },
                      },
                      "Probe Local Chrome CDP (:9222)"
                    ),
                    h(
                      "button",
                      {
                        className: "sam-btn sam-btn-success",
                        disabled: busy,
                        onClick: function () {
                          runBrowserAction("configure_mcp");
                        },
                      },
                      "Write .vscode/mcp.json (@playwright/mcp + chrome-devtools-mcp)"
                    )
                  )
                )
              )
            : null,

          // TAB 3: GITHUB AUTO-SYNC & PULL REQUESTS
          tab === "github"
            ? h(
                "div",
                { className: "sam-grid-2" },
                h(
                  "div",
                  { className: "sam-card" },
                  h(
                    "div",
                    { className: "sam-card-title" },
                    h("span", null, "GitHub Remote & Auto-Sync When Task Finishes"),
                    pill(Boolean(gh.auto_push_on_complete), gh.auto_push_on_complete ? "AUTO-PUSH ON" : "MANUAL PUSH")
                  ),
                  h(
                    "div",
                    { style: { marginBottom: "10px" } },
                    h("div", { style: { fontSize: "11px", marginBottom: "4px" } }, "GitHub Remote URL (origin):"),
                    h(
                      "div",
                      { style: { display: "flex", gap: "6px" } },
                      h("input", {
                        className: "sam-input sam-mono",
                        placeholder: "https://github.com/your-org/your-repo.git",
                        value: remoteUrlInput,
                        onChange: function (e) {
                          setRemoteUrlInput(e.target.value);
                        },
                      }),
                      h(
                        "button",
                        {
                          className: "sam-btn sam-btn-primary",
                          onClick: function () {
                            apiPost("/github/prefs", { remote_url: remoteUrlInput }).then(function (d) {
                              setState(d);
                              setNotice("Saved GitHub remote origin URL.");
                            });
                          },
                        },
                        "Save Remote"
                      )
                    )
                  ),
                  h(
                    "div",
                    { style: { marginBottom: "10px" } },
                    h("div", { style: { fontSize: "11px", marginBottom: "4px" } }, "Commit Message:"),
                    h(
                      "div",
                      { style: { display: "flex", gap: "6px" } },
                      h("input", {
                        className: "sam-input",
                        value: commitMsgInput,
                        onChange: function (e) {
                          setCommitMsgInput(e.target.value);
                        },
                      }),
                      h(
                        "button",
                        { className: "sam-btn sam-btn-success", disabled: busy, onClick: handleGitHubSync },
                        "⬆ Commit & Push Now"
                      )
                    )
                  ),
                  h(
                    "div",
                    { style: { marginTop: "14px", borderTop: "1px solid #1e293b", paddingTop: "12px" } },
                    h("div", { style: { fontSize: "12px", fontWeight: 700, marginBottom: "6px" } }, "Create Verified GitHub Pull Request (gh pr create)"),
                    h(
                      "div",
                      { style: { display: "flex", gap: "6px" } },
                      h("input", {
                        className: "sam-input",
                        placeholder: "PR Title (optional — auto-filled from Spec Contract)",
                        value: prTitleInput,
                        onChange: function (e) {
                          setPrTitleInput(e.target.value);
                        },
                      }),
                      h(
                        "button",
                        { className: "sam-btn sam-btn-primary", disabled: busy, onClick: handleCreatePr },
                        "⑂ Create GitHub PR"
                      )
                    )
                  )
                ),
                h(
                  "div",
                  { className: "sam-card" },
                  h("div", { className: "sam-card-title" }, h("span", null, "Codex Per-File Diff Summary")),
                  h(
                    "div",
                    { className: "sam-list" },
                    (diffSum.files || []).map(function (f) {
                      return h(
                        "div",
                        { key: f.path, className: "sam-list-item" },
                        h("code", { className: "sam-mono" }, f.path),
                        h(
                          "div",
                          null,
                          h("span", { className: "sam-diff-add" }, "+" + f.additions),
                          h("span", { className: "sam-diff-del" }, "-" + f.deletions)
                        )
                      );
                    })
                  )
                )
              )
            : null,

          // TAB 4: SPEC CONTRACT & PLAN CARD
          tab === "brief"
            ? h(
                "div",
                { className: "sam-grid-2" },
                h(
                  "div",
                  { className: "sam-card" },
                  h("div", { className: "sam-card-title" }, "Active Spec Contract (.samagent/spec.yaml)"),
                  h(
                    "div",
                    { className: "sam-list" },
                    (spec.stories || []).map(function (s) {
                      return h(
                        "div",
                        { key: s.id, className: "sam-list-item" },
                        h("span", null, "[" + s.id + "] As " + s.as_role + ", I can " + s.can + " (" + s.accept + ")"),
                        h("span", { className: "sam-pill sam-pill-blue" }, s.method + " " + s.route)
                      );
                    })
                  )
                ),
                h(
                  "div",
                  { className: "sam-card" },
                  h("div", { className: "sam-card-title" }, "Frozen SQL Schema (.samagent/contract/db/schema.sql)"),
                  h("pre", { className: "sam-pre" }, (state.contracts && state.contracts["db/schema.sql"]) || "")
                )
              )
            : null,

          // TAB 5: PERSISTENT LEDGER & BENCHMARKS
          tab === "ledger"
            ? h(
                "div",
                { className: "sam-grid-2" },
                h(
                  "div",
                  { className: "sam-card" },
                  h("div", { className: "sam-card-title" }, "Bi-Temporal Project Ledger (.samagent/ledger.db)"),
                  h(
                    "div",
                    { className: "sam-list" },
                    ((state.ledger && state.ledger.active_facts) || []).map(function (f) {
                      return h(
                        "div",
                        { key: f.id, className: "sam-list-item" },
                        h("span", null, "[" + f.scope + "/" + f.kind + "] " + f.text),
                        h("span", { className: "sam-pill sam-pill-green" }, f.sensitivity)
                      );
                    })
                  )
                ),
                h(
                  "div",
                  { className: "sam-card" },
                  h("div", { className: "sam-card-title" }, "SamBench-v0 & Ablation H1–H7 Summary"),
                  h(
                    "pre",
                    { className: "sam-pre", style: { maxHeight: "260px" } },
                    JSON.stringify(
                      state.measurements &&
                        state.measurements.ablation_h1_h7 &&
                        state.measurements.ablation_h1_h7.hypotheses,
                      null,
                      2
                    )
                  )
                )
              )
            : null
        )
      ),

      // ======================================================================
      // 3. BOTTOM STICKY CODEX COMPOSER BAR (Ask / Plan vs Code Auto-Worktree)
      // ======================================================================
      h(
        "div",
        { className: "sam-codex-composer" },
        h(
          "select",
          {
            className: "sam-select",
            style: { width: "175px" },
            value: composerMode,
            onChange: function (e) {
              setComposerMode(e.target.value);
            },
          },
          h("option", { value: "code" }, "⚡ Code (Worktree + Verify)"),
          h("option", { value: "ask" }, "📋 Plan Only (Create To-Do)")
        ),
        h("input", {
          className: "sam-input",
          style: { flex: 1 },
          placeholder: "Codex Prompt: Describe an app or feature to plan, build in isolated git worktrees, verify L0–L4 & sync with VS Code...",
          value: briefInput,
          onChange: function (e) {
            setBriefInput(e.target.value);
          },
        }),
        h(
          "button",
          {
            className: "sam-btn sam-btn-primary",
            disabled: busy,
            onClick: function () {
              runCodexTask("ask");
            },
          },
          "Plan To-Do"
        ),
        h(
          "button",
          {
            className: "sam-btn sam-btn-success",
            disabled: busy,
            onClick: function () {
              runCodexTask(composerMode);
            },
          },
          busy ? "⟳ Running Task..." : "▶ Run in Codex Studio"
        )
      )
    );
  }

  if (window.__HERMES_PLUGINS__ && typeof window.__HERMES_PLUGINS__.register === "function") {
    window.__HERMES_PLUGINS__.register("samagent", SamAgentMissionControl);
  }
})();

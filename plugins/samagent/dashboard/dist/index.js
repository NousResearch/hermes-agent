/* ==========================================================================
   SamAgent — Codex Desktop UI
   Pixel-Accurate Clone of OpenAI Codex Desktop App
   ========================================================================== */
(function () {
  var sdk = window.__SAMAGENT_PLUGIN_SDK__ || window.__HERMES_PLUGIN_SDK__ || {};
  var React = sdk.React || {};
  var h = React.createElement;
  var useState = (sdk.hooks && sdk.hooks.useState) || React.useState;
  var useEffect = (sdk.hooks && sdk.hooks.useEffect) || React.useEffect;
  var useCallback = (sdk.hooks && sdk.hooks.useCallback) || React.useCallback;

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

  // --- Inline Vector SVG Icon Components ---
  function IconHome() {
    return h("svg", { className: "codex-svg-icon", viewBox: "0 0 24 24" },
      h("path", { d: "m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" }),
      h("polyline", { points: "9 22 9 12 15 12 15 22" })
    );
  }

  function IconClock() {
    return h("svg", { className: "codex-svg-icon", viewBox: "0 0 24 24" },
      h("circle", { cx: "12", cy: "12", r: "10" }),
      h("polyline", { points: "12 6 12 12 16 14" })
    );
  }

  function IconAt() {
    return h("svg", { className: "codex-svg-icon", viewBox: "0 0 24 24" },
      h("circle", { cx: "12", cy: "12", r: "4" }),
      h("path", { d: "M16 8v5a3 3 0 0 0 6 0v-1a10 10 0 1 0-4 8" })
    );
  }

  function IconMore() {
    return h("svg", { className: "codex-svg-icon", viewBox: "0 0 24 24" },
      h("circle", { cx: "12", cy: "12", r: "1" }),
      h("circle", { cx: "19", cy: "12", r: "1" }),
      h("circle", { cx: "5", cy: "12", r: "1" })
    );
  }

  function IconSettings() {
    return h("svg", { className: "codex-svg-icon", viewBox: "0 0 24 24" },
      h("path", { d: "M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" }),
      h("circle", { cx: "12", cy: "12", r: "3" })
    );
  }

  function IconSearch() {
    return h("svg", { className: "codex-svg-icon", viewBox: "0 0 24 24" },
      h("circle", { cx: "11", cy: "11", r: "8" }),
      h("path", { d: "m21 21-4.3-4.3" })
    );
  }

  function IconBell() {
    return h("svg", { className: "codex-svg-icon", viewBox: "0 0 24 24" },
      h("path", { d: "M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" }),
      h("path", { d: "M10.3 21a1.94 1.94 0 0 0 3.4 0" })
    );
  }

  function IconEdit() {
    return h("svg", { className: "codex-svg-icon", viewBox: "0 0 24 24" },
      h("path", { d: "M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" }),
      h("path", { d: "M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" })
    );
  }

  function IconFolder() {
    return h("svg", { className: "codex-svg-icon", viewBox: "0 0 24 24" },
      h("path", { d: "M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" })
    );
  }

  function IconLaptop() {
    return h("svg", { className: "codex-svg-icon", viewBox: "0 0 24 24" },
      h("rect", { width: "18", height: "12", x: "3", y: "4", rx: "2" }),
      h("line", { x1: "2", x2: "22", y1: "20", y2: "20" })
    );
  }

  function IconBranch() {
    return h("svg", { className: "codex-svg-icon", viewBox: "0 0 24 24" },
      h("line", { x1: "6", x2: "6", y1: "3", y2: "15" }),
      h("circle", { cx: "18", cy: "6", r: "3" }),
      h("circle", { cx: "6", cy: "18", r: "3" }),
      h("path", { d: "M18 9a9 9 0 0 1-9 9" })
    );
  }

  function IconShield() {
    return h("svg", { className: "codex-svg-icon", viewBox: "0 0 24 24", style: { width: "14px", height: "14px", color: "#ea580c" } },
      h("path", { d: "M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z" })
    );
  }

  function IconArrowUp() {
    return h("svg", { className: "codex-svg-icon", viewBox: "0 0 24 24", style: { width: "14px", height: "14px", strokeWidth: "2.5" } },
      h("path", { d: "M12 19V5" }),
      h("path", { d: "m5 12 7-7 7 7" })
    );
  }

  function IconSplit() {
    return h("svg", { className: "codex-svg-icon", viewBox: "0 0 24 24" },
      h("rect", { width: "18", height: "18", x: "3", y: "3", rx: "2" }),
      h("path", { d: "M9 3v18" })
    );
  }

  function CloudMascotSvg() {
    return h("svg", {
      className: "codex-mascot-cloud",
      viewBox: "0 0 64 52",
      fill: "none",
      stroke: "currentColor",
      strokeWidth: "2",
      strokeLinecap: "round",
      strokeLinejoin: "round",
    },
      h("path", { d: "M 18 44 C 11 44 6 39 6 32 C 6 26 10 21 16 20 C 17 11 25 5 35 5 C 44 5 51 11 53 19 C 58 20 62 25 62 31 C 62 38 57 44 49 44 Z" }),
      h("path", { d: "M 27 28 C 29 30 29 34 27 36" }),
      h("path", { d: "M 32 36 L 40 36" })
    );
  }

  function SamAgentCodexDesktopApp() {
    var _s = useState(null), state = _s[0], setState = _s[1];
    var _b = useState(false), busy = _b[0], setBusy = _b[1];
    var _err = useState(""), errMsg = _err[0], setErrMsg = _err[1];
    var _notice = useState(""), notice = _notice[0], setNotice = _notice[1];

    // Navigation & View Mode
    var _mode = useState("home"), activeView = _mode[0], setActiveView = _mode[1]; // "home" | "chat"
    var _drawer = useState(false), drawerOpen = _drawer[0], setDrawerOpen = _drawer[1];
    var _dtab = useState("diffs"), drawerTab = _dtab[0], setDrawerTab = _dtab[1]; // "diffs" | "vscode" | "gate" | "preview" | "git"

    var DEFAULT_MODELS = [
      { id: "claude-3.7-sonnet", name: "Claude 3.7 Sonnet", provider: "Anthropic", tag: "Recommended", desc: "Flagship hybrid reasoning, agentic coding & reflection" },
      { id: "gpt-4o", name: "GPT-4o", provider: "OpenAI", tag: "Flagship", desc: "High-speed multimodal, tool calling & autonomous execution" },
      { id: "o3-mini", name: "o3-mini", provider: "OpenAI", tag: "Reasoning", desc: "STEM, math logic & exhaustive code verification" },
      { id: "claude-3.5-sonnet", name: "Claude 3.5 Sonnet", provider: "Anthropic", tag: "Standard", desc: "Reliable coding baseline and artifact generation" },
      { id: "gemini-2.0-flash", name: "Gemini 2.0 Flash", provider: "Google", tag: "Ultra-Fast", desc: "1M token context window, sub-second latency" },
      { id: "deepseek-r1", name: "DeepSeek R1", provider: "DeepSeek", tag: "Open Weights", desc: "Open reasoning model with chain-of-thought verification" },
      { id: "qwen2.5-coder-32b", name: "Qwen 2.5 Coder 32B", provider: "Local-First", tag: "Private", desc: "Runs offline locally with zero cloud API dependency" },
    ];

    // Prompt & Composer
    var _p = useState(""), promptText = _p[0], setPromptText = _p[1];
    var _access = useState("full"), accessLevel = _access[0], setAccessLevel = _access[1]; // "full" | "ask"
    var _model = useState("Claude 3.7 Sonnet"), selectedModel = _model[0], setSelectedModel = _model[1];
    var _contextOpen = useState(false), contextMenuOpen = _contextOpen[0], setContextMenuOpen = _contextOpen[1];
    var _om = useState(null), openMenu = _om[0], setOpenMenu = _om[1]; // null | "file" | "edit" | "view" | "help" | "model" | "brand"

    // Modals
    var _settings = useState(false), settingsOpen = _settings[0], setSettingsOpen = _settings[1];
    var _fmodal = useState(false), folderModalOpen = _fmodal[0], setFolderModalOpen = _fmodal[1];
    var _customPath = useState(""), customFolderPath = _customPath[0], setCustomFolderPath = _customPath[1];

    // Selected File & Editor
    var _sf = useState("app/main.py"), selectedFile = _sf[0], setSelectedFile = _sf[1];
    var _fc = useState(""), fileContent = _fc[0], setFileContent = _fc[1];

    // GitHub & Commit
    var _cmsg = useState("feat: verified build from SamAgent Codex Desktop"), commitMsg = _cmsg[0], setCommitMsg = _cmsg[1];
    var _prTitle = useState("feat: autonomous build from SamAgent"), prTitle = _prTitle[0], setPrTitle = _prTitle[1];

    // Dev App Roles state
    var _role = useState("visitor"), devRole = _role[0], setDevRole = _role[1];
    var _devOut = useState(null), devResponse = _devOut[0], setDevResponse = _devOut[1];

    // Recents List
    var recentTasks = [
      "Booking site with member auth & schedule",
      "OWASP IDOR verification rules",
      "Freeze SQLite database schema",
      "Red-first acceptance test suite",
    ];

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
        })
        .catch(function (e) {
          setErrMsg(String(e));
        });
    }

    useEffect(function () {
      refresh();
      loadFile("app/main.py");
    }, []);

    // Global keyboard shortcuts (Ctrl+N, Ctrl+O, Ctrl+B, Ctrl+K, Escape)
    useEffect(function () {
      function handleKeyDown(e) {
        if (e.key === "Escape") {
          setOpenMenu(null);
          setContextMenuOpen(false);
          setFolderModalOpen(false);
          setSettingsOpen(false);
        }
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "n") {
          e.preventDefault();
          handleStartNewChat();
        }
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
          e.preventDefault();
          setDrawerOpen(function (d) { return !d; });
        }
        if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === "o" || e.key.toLowerCase() === "k")) {
          e.preventDefault();
          setFolderModalOpen(true);
        }
      }
      window.addEventListener("keydown", handleKeyDown);
      return function () { window.removeEventListener("keydown", handleKeyDown); };
    }, []);

    function handleStartNewChat() {
      setPromptText("");
      setActiveView("home");
      setDrawerOpen(false);
      setContextMenuOpen(false);
      setOpenMenu(null);
      setNotice("Started fresh chat session.");
    }

    function executeTask(textToRun) {
      var query = (textToRun || promptText || "").trim();
      if (!query) {
        query = "Booking site for my yoga studio where visitors see the schedule, members book classes (no double booking, private per member), and admins add classes.";
      }
      setBusy(true);
      setErrMsg("");
      setActiveView("chat");
      setContextMenuOpen(false);

      var modelsList = (state && state.available_models) || DEFAULT_MODELS;
      var chosenModelId = "claude-3.7-sonnet";
      for (var mi = 0; mi < modelsList.length; mi++) {
        if (modelsList[mi].name === selectedModel || modelsList[mi].id === selectedModel) {
          chosenModelId = modelsList[mi].id;
          break;
        }
      }

      var autonomy = accessLevel === "ask" ? "plan_only" : "milestones";
      apiPost("/plan", {
        brief: query,
        answers: {},
        router_policy: "default",
        autonomy: autonomy,
        model: chosenModelId,
      })
        .then(function (planned) {
          setState(planned);
          if (accessLevel === "ask") {
            setNotice("Plan synthesized with " + selectedModel + " (8 to-dos queued). Review steps or click submit to build.");
            return planned;
          }
          return apiPost("/build", { autonomy: autonomy, router_policy: "default", model: chosenModelId });
        })
        .then(function (finished) {
          if (finished) setState(finished);
          loadFile("app/main.py");
          if (accessLevel !== "ask") {
            setNotice("Task built with " + selectedModel + "! L0–L4 verified & live on port 3000.");
          }
        })
        .catch(function (e) {
          setErrMsg(String(e));
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function handleLoadFolder(fPath) {
      if (!fPath) return;
      setBusy(true);
      apiPost("/folder/load", { folder_path: fPath })
        .then(function (data) {
          setState(data);
          setFolderModalOpen(false);
          setCustomFolderPath("");
          loadFile("app/main.py");
          setNotice("Loaded project workspace: " + data.workspace);
        })
        .catch(function (e) {
          setErrMsg(String(e));
        })
        .finally(function () {
          setBusy(false);
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
      apiPost("/ide/file", { rel_path: selectedFile, content: fileContent, auto_reverify: true })
        .then(function (data) {
          setState(data);
          setNotice("Saved " + selectedFile + " & verified Pre-Prod gates.");
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function handleGitHubSync() {
      setBusy(true);
      apiPost("/github/sync", { commit_message: commitMsg, push_to_remote: true })
        .then(function (data) {
          setState(data);
          setNotice("Git sync & push completed successfully.");
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
      apiPost("/github/pr", { title: prTitle || null })
        .then(function (data) {
          setState(data);
          var pr = data.github_pr_result || {};
          setNotice(pr.created_on_github ? "Created PR: " + pr.pr_url : (pr.message || "PR prepared"));
        })
        .catch(function (e) {
          setErrMsg(String(e));
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function runDevAction(act) {
      setBusy(true);
      var uId = devRole === "member" ? "u_member_a" : (devRole === "admin" ? "u_admin" : "");
      apiPost("/dev-app/action", {
        action: act,
        role: devRole,
        user_id: uId,
        item_id: "item_1",
        booking_id: (state && state.dev_app && state.dev_app.bookings && state.dev_app.bookings[0] && state.dev_app.bookings[0].id) || "b_demo",
      })
        .then(function (res) {
          setDevResponse(res);
          refresh();
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function handleInstallPlatform() {
      setBusy(true);
      apiPost("/platform/install", {})
        .then(function (res) {
          setNotice("Installed local OS desktop launcher & VS Code extension!");
          setSettingsOpen(false);
          refresh();
        })
        .finally(function () {
          setBusy(false);
        });
    }

    var projectName = "SamjuniorsOS";
    if (state && state.workspace) {
      var parts = state.workspace.replace(/\\/g, "/").split("/");
      projectName = parts.pop() || "SamjuniorsOS";
    }

    var todos = (state && state.todos) || { items: [], completed: 0, total: 0, progress_pct: 0 };
    var gh = (state && state.github) || { branch: "main", head_commit: "HEAD", diff_summary: { files: [], total_additions: 0, total_deletions: 0 } };
    var diffSum = gh.diff_summary || { files: [], total_additions: 0, total_deletions: 0 };
    var preProd = (state && state.pre_prod_gate) || { ready_for_production: true, checks: {} };
    var folderBrowser = (state && state.folder_browser) || { folders: [] };
    var rawGitDiff = (state && state.ide && state.ide.git && state.ide.git.diff) || "";

    return h(
      "div",
      { className: "codex-desktop-window" },

      // ======================================================================
      // 1. WINDOW TOP MENU BAR (File, Edit, View, Help)
      // ======================================================================
      h(
        "div",
        { className: "codex-window-bar" },
        h(
          "div",
          { className: "codex-window-bar-left" },
          h(
            "div",
            { className: "codex-nav-arrows" },
            h("span", { className: "codex-nav-arrow", onClick: handleStartNewChat, title: "Home" }, "←"),
            h("span", { className: "codex-nav-arrow", title: "Forward" }, "→"),
            h("span", { className: "codex-nav-arrow", onClick: function () { setDrawerOpen(!drawerOpen); }, title: "Toggle Side Inspector (◧)" }, h(IconSplit))
          ),
          h(
            "div",
            { className: "codex-window-menus" },

            // File Menu
            h(
              "div",
              { className: "codex-menu-wrapper" },
              h(
                "span",
                {
                  className: "codex-menu-item " + (openMenu === "file" ? "active" : ""),
                  onClick: function (e) {
                    e.stopPropagation();
                    setOpenMenu(openMenu === "file" ? null : "file");
                  },
                },
                "File"
              ),
              openMenu === "file"
                ? h(
                    "div",
                    { className: "codex-menu-dropdown" },
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          handleStartNewChat();
                        },
                      },
                      "New Chat",
                      h("span", { className: "codex-menu-shortcut" }, "Ctrl+N")
                    ),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          setFolderModalOpen(true);
                        },
                      },
                      "Open Project Folder...",
                      h("span", { className: "codex-menu-shortcut" }, "Ctrl+O")
                    ),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          saveFileAndVerify();
                        },
                      },
                      "Save Current File",
                      h("span", { className: "codex-menu-shortcut" }, "Ctrl+S")
                    ),
                    h("div", { className: "codex-menu-separator" }),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          refresh();
                          setNotice("Refreshed workspace state.");
                        },
                      },
                      "Refresh State",
                      h("span", { className: "codex-menu-shortcut" }, "F5")
                    )
                  )
                : null
            ),

            // Edit Menu
            h(
              "div",
              { className: "codex-menu-wrapper" },
              h(
                "span",
                {
                  className: "codex-menu-item " + (openMenu === "edit" ? "active" : ""),
                  onClick: function (e) {
                    e.stopPropagation();
                    setOpenMenu(openMenu === "edit" ? null : "edit");
                  },
                },
                "Edit"
              ),
              openMenu === "edit"
                ? h(
                    "div",
                    { className: "codex-menu-dropdown" },
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          setPromptText("");
                        },
                      },
                      "Clear Prompt"
                    ),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          setAccessLevel(accessLevel === "full" ? "ask" : "full");
                          setNotice("Access level changed to: " + (accessLevel === "full" ? "Review steps" : "Full access"));
                        },
                      },
                      accessLevel === "full" ? "Switch to: Review steps" : "Switch to: Full access"
                    ),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          setDrawerOpen(true);
                          setDrawerTab("diffs");
                        },
                      },
                      "Inspect Git Diffs"
                    )
                  )
                : null
            ),

            // View Menu
            h(
              "div",
              { className: "codex-menu-wrapper" },
              h(
                "span",
                {
                  className: "codex-menu-item " + (openMenu === "view" ? "active" : ""),
                  onClick: function (e) {
                    e.stopPropagation();
                    setOpenMenu(openMenu === "view" ? null : "view");
                  },
                },
                "View"
              ),
              openMenu === "view"
                ? h(
                    "div",
                    { className: "codex-menu-dropdown" },
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          setDrawerOpen(!drawerOpen);
                        },
                      },
                      "Toggle Side Inspector",
                      h("span", { className: "codex-menu-shortcut" }, "Ctrl+B")
                    ),
                    h("div", { className: "codex-menu-separator" }),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          setDrawerOpen(true);
                          setDrawerTab("diffs");
                        },
                      },
                      "Diffs (" + (diffSum.files || []).length + ")"
                    ),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          setDrawerOpen(true);
                          setDrawerTab("vscode");
                        },
                      },
                      "VS Code Studio"
                    ),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          setDrawerOpen(true);
                          setDrawerTab("preview");
                        },
                      },
                      "Live App (:3000)"
                    ),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          setDrawerOpen(true);
                          setDrawerTab("gate");
                        },
                      },
                      "Pre-Prod Gate (7/7)"
                    ),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          setDrawerOpen(true);
                          setDrawerTab("git");
                        },
                      },
                      "Git / PR"
                    )
                  )
                : null
            ),

            // Help Menu
            h(
              "div",
              { className: "codex-menu-wrapper" },
              h(
                "span",
                {
                  className: "codex-menu-item " + (openMenu === "help" ? "active" : ""),
                  onClick: function (e) {
                    e.stopPropagation();
                    setOpenMenu(openMenu === "help" ? null : "help");
                  },
                },
                "Help"
              ),
              openMenu === "help"
                ? h(
                    "div",
                    { className: "codex-menu-dropdown" },
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          setSettingsOpen(true);
                        },
                      },
                      "Platform Status & Settings"
                    ),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          handleInstallPlatform();
                        },
                      },
                      "Install OS Desktop Launcher & VS Code Extension"
                    ),
                    h("div", { className: "codex-menu-separator" }),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          window.open("/docs", "_blank");
                        },
                      },
                      "FastAPI Documentation ↗"
                    )
                  )
                : null
            )
          )
        ),
        h(
          "div",
          { className: "codex-window-controls" },
          h("span", { className: "codex-win-btn" }, "—"),
          h("span", { className: "codex-win-btn" }, "▢"),
          h("span", { className: "codex-win-btn" }, "✕")
        )
      ),

      // ======================================================================
      // 2. MAIN APPLICATION BODY (Icon Rail + Sub-Sidebar + Center Canvas)
      // ======================================================================
      h(
        "div",
        { className: "codex-body" },

        // FAR-LEFT NARROW ICON RAIL (48px)
        h(
          "aside",
          { className: "codex-rail" },
          h(
            "div",
            { className: "codex-rail-top" },
            h(
              "div",
              {
                className: "codex-rail-icon " + (activeView === "home" ? "active" : ""),
                onClick: handleStartNewChat,
                title: "Home",
              },
              h(IconHome)
            ),
            h(
              "div",
              {
                className: "codex-rail-icon " + (activeView === "chat" ? "active" : ""),
                onClick: function () {
                  setActiveView("chat");
                  setDrawerOpen(true);
                  setDrawerTab("diffs");
                },
                title: "Task History & Plan",
              },
              h(IconClock)
            ),
            h(
              "div",
              {
                className: "codex-rail-icon",
                onClick: function () {
                  setDrawerOpen(true);
                  setDrawerTab("preview");
                },
                title: "Live App (:3000) & Inspector",
              },
              h(IconAt)
            ),
            h(
              "div",
              {
                className: "codex-rail-icon",
                onClick: function () {
                  setFolderModalOpen(true);
                },
                title: "Projects & Workspaces",
              },
              h(IconMore)
            )
          ),

          h(
            "div",
            { className: "codex-rail-bottom" },
            h(
              "div",
              {
                className: "codex-rail-icon",
                onClick: function () {
                  setSettingsOpen(!settingsOpen);
                },
                title: "Settings & Platform Status",
              },
              h(IconSettings)
            ),
            h(
              "div",
              {
                className: "codex-avatar-badge",
                onClick: function () {
                  setNotice("SamAgent Platform active on 127.0.0.1:8080");
                },
                title: "SamAgent Studio — Connected",
              },
              "S",
              h("div", { className: "codex-avatar-dot" })
            )
          )
        ),

        // SECONDARY LEFT SIDEBAR (Projects & Chats, ~230px)
        h(
          "aside",
          { className: "codex-sub-sidebar" },

          // Header: SamAgent ⌵, Bell, Search
          h(
            "div",
            { className: "codex-sub-header" },
            h(
              "div",
              { style: { position: "relative" } },
              h(
                "div",
                {
                  className: "codex-dropdown-trigger",
                  onClick: function (e) {
                    e.stopPropagation();
                    setOpenMenu(openMenu === "brand" ? null : "brand");
                  },
                },
                "SamAgent",
                h("span", { style: { fontSize: "11px", color: "#71717a" } }, "⌵")
              ),
              openMenu === "brand"
                ? h(
                    "div",
                    { className: "codex-menu-dropdown", style: { top: "100%", left: 0, minWidth: "220px" } },
                    h(
                      "div",
                      { className: "codex-menu-dropdown-item", style: { fontWeight: 600, color: "#16a34a" } },
                      "● Engine: 127.0.0.1:8080",
                      h("span", { className: "codex-menu-shortcut" }, "ACTIVE")
                    ),
                    h("div", { className: "codex-menu-separator" }),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          setFolderModalOpen(true);
                        },
                      },
                      "📁 Switch Workspace...",
                      h("span", { className: "codex-menu-shortcut" }, "Ctrl+O")
                    ),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          handleStartNewChat();
                        },
                      },
                      "✎ New Chat Session",
                      h("span", { className: "codex-menu-shortcut" }, "Ctrl+N")
                    ),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          refresh();
                          setNotice("Workspace state refreshed.");
                        },
                      },
                      "⟳ Refresh State",
                      h("span", { className: "codex-menu-shortcut" }, "F5")
                    ),
                    h("div", { className: "codex-menu-separator" }),
                    h(
                      "div",
                      {
                        className: "codex-menu-dropdown-item",
                        onClick: function () {
                          setOpenMenu(null);
                          setSettingsOpen(true);
                        },
                      },
                      "⚙ Platform Settings..."
                    )
                  )
                : null
            ),
            h(
              "div",
              { className: "codex-sub-icons" },
              h("span", { className: "codex-sub-icon-btn", title: "Notifications" }, h(IconBell)),
              h(
                "span",
                {
                  className: "codex-sub-icon-btn",
                  onClick: function () {
                    setFolderModalOpen(true);
                  },
                  title: "Search Workspaces",
                },
                h(IconSearch)
              )
            )
          ),

          // "New chat" button
          h(
            "div",
            { className: "codex-new-chat-btn", onClick: handleStartNewChat },
            h(IconEdit),
            h("span", null, "New chat")
          ),

          // Projects Section
          h("div", { className: "codex-section-label" }, "Projects"),
          h(
            "div",
            {
              className: "codex-project-item",
              onClick: function () {
                setFolderModalOpen(true);
              },
            },
            h(IconFolder),
            h("span", { style: { flex: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" } }, projectName)
          ),

          // Recents Section
          h(
            "div",
            {
              className: "codex-recents-header",
              onClick: function () {
                setActiveView("chat");
              },
            },
            h("span", null, "Recents"),
            h("span", null, "›")
          ),
          h(
            "div",
            { style: { display: "flex", flexDirection: "column", gap: "2px", marginTop: "4px" } },
            recentTasks.map(function (task, idx) {
              return h(
                "div",
                {
                  key: idx,
                  className: "codex-recent-item",
                  onClick: function () {
                    setPromptText(task);
                    executeTask(task);
                  },
                },
                task
              );
            })
          )
        ),

        // CENTER MAIN CANVAS
        h(
          "main",
          { className: "codex-canvas" },

          // Top right split-screen toggle button [|]
          h(
            "div",
            { className: "codex-canvas-top-actions" },
            h(
              "button",
              {
                className: "codex-icon-button " + (drawerOpen ? "active" : ""),
                onClick: function () {
                  setDrawerOpen(!drawerOpen);
                },
                title: "Toggle Side Inspector (Diffs, VS Code, Live App)",
              },
              h(IconSplit)
            )
          ),

          // Notifications Banner (if any)
          notice
            ? h(
                "div",
                {
                  style: {
                    position: "absolute",
                    top: "10px",
                    left: "24px",
                    right: "60px",
                    background: "#f0fdf4",
                    border: "1px solid #bbf7d0",
                    color: "#166534",
                    padding: "8px 14px",
                    borderRadius: "8px",
                    fontSize: "12px",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    zIndex: 25,
                  },
                },
                h("span", null, "✓ " + notice),
                h(
                  "button",
                  {
                    style: { background: "none", border: "none", cursor: "pointer", color: "#166534", fontWeight: 700 },
                    onClick: function () {
                      setNotice("");
                    },
                  },
                  "✕"
                )
              )
            : null,

          errMsg
            ? h(
                "div",
                {
                  style: {
                    position: "absolute",
                    top: "10px",
                    left: "24px",
                    right: "60px",
                    background: "#fef2f2",
                    border: "1px solid #fecaca",
                    color: "#991b1b",
                    padding: "8px 14px",
                    borderRadius: "8px",
                    fontSize: "12px",
                    zIndex: 25,
                  },
                },
                "✕ " + errMsg
              )
            : null,

          // CONTENT VIEW: Either Home Empty State OR Active Execution Chat
          activeView === "home"
            ? h(
                "div",
                { className: "codex-center-content" },
                h(CloudMascotSvg, null),
                h("h1", { className: "codex-hero-heading" }, "What should we build in " + projectName + "?"),
                h(
                  "div",
                  { className: "codex-starter-grid" },
                  [
                    {
                      label: "✨ Build visitor schedule & booking auth",
                      prompt: "Booking site for my yoga studio where visitors see the schedule, members book classes (no double booking, private per member), and admins add classes.",
                    },
                    {
                      label: "🛡 Run OWASP IDOR & RBAC security audit",
                      prompt: "Run full OWASP IDOR and authorization verification tests against all member endpoints.",
                    },
                    {
                      label: "🧪 Execute red-first acceptance test suite",
                      prompt: "Run comprehensive red-first acceptance tests covering all visitor, member, and admin roles.",
                    },
                    {
                      label: "🚀 Verify 7/7 pre-production quality gates",
                      prompt: "Run all 7 pre-production quality gates and generate Dockerfile + RELEASE_MANIFEST.json.",
                    },
                  ].map(function (chip, idx) {
                    return h(
                      "button",
                      {
                        key: idx,
                        className: "codex-starter-chip",
                        onClick: function () {
                          setPromptText(chip.prompt);
                          executeTask(chip.prompt);
                        },
                      },
                      chip.label
                    );
                  })
                )
              )
            : h(
                "div",
                { className: "codex-stream-container" },
                // User Prompt Bubble
                h(
                  "div",
                  { className: "codex-chat-row" },
                  h(
                    "div",
                    { className: "codex-user-bubble" },
                    promptText || "Booking site for my yoga studio where visitors see the schedule, members book classes (no double booking, private per member), and admins add classes."
                  )
                ),

                // Agent Response Card with Live Execution
                h(
                  "div",
                  { className: "codex-agent-card" },
                  h(
                    "div",
                    { style: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" } },
                    h(
                      "div",
                      { style: { display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" } },
                      h("span", { style: { fontSize: "16px" } }, "⚡"),
                      h("strong", { style: { fontSize: "14px", color: "#18181b" } }, "SamAgent Engine"),
                      h("span", { style: { fontSize: "11px", background: "#eff6ff", color: "#2563eb", padding: "2px 8px", borderRadius: "12px", fontWeight: 600 } }, selectedModel),
                      busy ? h("span", { className: "codex-spin", style: { color: "#2563eb", fontSize: "14px" } }, "⟳") : h("span", { style: { fontSize: "12px", color: "#16a34a", fontWeight: 600 } }, "✓ All Steps Finished")
                    ),
                    h("span", { style: { fontSize: "12px", color: "#71717a" } }, (todos.completed || 0) + "/" + (todos.total || 0) + " tasks")
                  ),
                  h(
                    "div",
                    { style: { background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: "8px", padding: "6px 12px", marginBottom: "12px", fontSize: "11px", display: "flex", gap: "14px", flexWrap: "wrap" } },
                    h("div", null, h("span", { style: { color: "#64748b" } }, "Model: "), h("strong", { style: { color: "#0f172a" } }, selectedModel)),
                    h("div", null, h("span", { style: { color: "#64748b" } }, "Branch: "), h("strong", { style: { color: "#0f172a" } }, gh.branch || "main")),
                    h("div", null, h("span", { style: { color: "#64748b" } }, "Verification: "), h("strong", { style: { color: "#16a34a" } }, "7/7 Quality Gates Passed"))
                  ),

                  // Step-by-Step Stepper
                  h(
                    "div",
                    { style: { display: "flex", flexDirection: "column", gap: "6px" } },
                    (todos.items || []).map(function (item) {
                      var isDone = item.status === "completed";
                      return h(
                        "div",
                        {
                          key: item.id,
                          style: {
                            display: "flex",
                            alignItems: "center",
                            gap: "8px",
                            padding: "6px 10px",
                            background: isDone ? "#f0fdf4" : "#f4f4f5",
                            borderRadius: "6px",
                            fontSize: "12px",
                          },
                        },
                        h("span", { style: { color: isDone ? "#16a34a" : "#71717a", fontWeight: 700 } }, isDone ? "✓" : "○"),
                        h("strong", { style: { color: "#18181b" } }, item.id + ":"),
                        h("span", { style: { flex: 1, color: "#27272a" } }, item.title),
                        h("span", { style: { fontSize: "10px", color: "#71717a" } }, item.agent)
                      );
                    })
                  ),

                  // Quick Action Buttons inside Chat
                  h(
                    "div",
                    { style: { display: "flex", gap: "8px", marginTop: "14px", flexWrap: "wrap" } },
                    h(
                      "button",
                      {
                        className: "codex-btn-vscode",
                        onClick: function () {
                          openVsCode(null);
                        },
                      },
                      "Open in VS Code ↗"
                    ),
                    h(
                      "button",
                      {
                        className: "codex-btn-action",
                        onClick: function () {
                          setDrawerOpen(true);
                          setDrawerTab("preview");
                        },
                      },
                      "Preview App (:3000)"
                    ),
                    h(
                      "button",
                      {
                        className: "codex-btn-action",
                        onClick: function () {
                          setDrawerOpen(true);
                          setDrawerTab("diffs");
                        },
                      },
                      "Inspect Diffs (+" + (diffSum.total_additions || 0) + " / -" + (diffSum.total_deletions || 0) + ")"
                    )
                  )
                )
              ),

          // ==================================================================
          // 3. FLOATING CODEX COMPOSER WIDGET (The Exact Pill Box)
          // ==================================================================
          h(
            "div",
            { className: "codex-composer-container" },

            // Top Capsule Bar: Folder + Local + Branch
            h(
              "div",
              { className: "codex-composer-capsule" },
              h(
                "div",
                {
                  className: "codex-capsule-item",
                  onClick: function () {
                    setFolderModalOpen(true);
                  },
                },
                h(IconFolder),
                h("span", null, projectName)
              ),
              h(
                "div",
                { className: "codex-capsule-item" },
                h(IconLaptop),
                h("span", null, "Local")
              ),
              h(
                "div",
                {
                  className: "codex-capsule-item",
                  onClick: function () {
                    setDrawerOpen(true);
                    setDrawerTab("git");
                  },
                },
                h(IconBranch),
                h("span", null, gh.branch || "main")
              )
            ),

            // Inner Input Card
            h(
              "div",
              { className: "codex-composer-card" },
              h("textarea", {
                className: "codex-composer-textarea",
                placeholder: "Do anything",
                rows: 1,
                value: promptText,
                onChange: function (e) {
                  setPromptText(e.target.value);
                },
                onKeyDown: function (e) {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    executeTask(promptText);
                  }
                },
              }),

              // Bottom Toolbar inside Composer
              h(
                "div",
                { className: "codex-composer-bottom-bar" },
                h(
                  "div",
                  { className: "codex-bottom-left" },
                  // Plus button with context popover
                  h(
                    "button",
                    {
                      className: "codex-plus-btn",
                      onClick: function () {
                        setContextMenuOpen(!contextMenuOpen);
                      },
                      title: "Add context or trigger specific actions",
                    },
                    "+"
                  ),

                  // Context popover
                  contextMenuOpen
                    ? h(
                        "div",
                        { className: "codex-context-popover" },
                        h(
                          "div",
                          {
                            className: "codex-context-item",
                            onClick: function () {
                              setContextMenuOpen(false);
                              setDrawerOpen(true);
                              setDrawerTab("vscode");
                            },
                          },
                          h(IconFolder),
                          h("span", null, "Inspect Workspace File")
                        ),
                        h(
                          "div",
                          {
                            className: "codex-context-item",
                            onClick: function () {
                              setContextMenuOpen(false);
                              executeTask("Run full OWASP IDOR and RBAC security audit");
                            },
                          },
                          h(IconShield),
                          h("span", null, "Run OWASP Security Audit")
                        ),
                        h(
                          "div",
                          {
                            className: "codex-context-item",
                            onClick: function () {
                              setContextMenuOpen(false);
                              setDrawerOpen(true);
                              setDrawerTab("gate");
                            },
                          },
                          h("span", null, "🛡"),
                          h("span", null, "Check Pre-Prod Gates")
                        )
                      )
                    : null,

                  // Orange permission badge: Full access vs Review steps
                  h(
                    "div",
                    {
                      className: "codex-access-badge",
                      onClick: function () {
                        setAccessLevel(accessLevel === "full" ? "ask" : "full");
                      },
                      title: "Click to toggle between autonomous execution and review mode",
                    },
                    h(IconShield),
                    h("span", null, accessLevel === "full" ? "Full access" : "Review steps")
                  )
                ),

                h(
                  "div",
                  { className: "codex-bottom-right" },
                  // Model badge with popover
                  h(
                    "div",
                    { style: { position: "relative" } },
                    h(
                      "div",
                      {
                        className: "codex-model-badge",
                        onClick: function (e) {
                          e.stopPropagation();
                          setOpenMenu(openMenu === "model" ? null : "model");
                        },
                        title: "Select reasoning model tier",
                      },
                      selectedModel,
                      h("span", { style: { fontSize: "10px", marginLeft: "4px" } }, "⌵")
                    ),
                    openMenu === "model"
                      ? h(
                          "div",
                          { className: "codex-model-popover" },
                          ((state && state.available_models) || DEFAULT_MODELS).map(function (m) {
                            var isCur = selectedModel === m.name || selectedModel === m.id;
                            return h(
                              "div",
                              {
                                key: m.id || m.name,
                                className: "codex-model-option " + (isCur ? "active" : ""),
                                onClick: function () {
                                  setSelectedModel(m.name);
                                  setOpenMenu(null);
                                  setNotice("Model selected: " + m.name + " (" + (m.provider || "Local") + ")");
                                },
                              },
                              h(
                                "div",
                                { style: { display: "flex", justifyContent: "space-between", alignItems: "center" } },
                                h("strong", { style: { fontSize: "12px", color: isCur ? "#2563eb" : "#18181b" } }, (isCur ? "✓ " : "") + m.name),
                                h("span", { style: { fontSize: "10px", color: isCur ? "#2563eb" : "#64748b", background: isCur ? "#eff6ff" : "#f1f5f9", padding: "1px 6px", borderRadius: "10px", fontWeight: 600 } }, m.tag || m.provider)
                              ),
                              h("span", { style: { fontSize: "11px", color: "#71717a", marginTop: "2px" } }, m.desc)
                            );
                          })
                        )
                      : null
                  ),
                  // Round submit circle with white up-arrow ↑
                  h(
                    "button",
                    {
                      className: "codex-submit-circle",
                      disabled: busy,
                      onClick: function () {
                        executeTask(promptText);
                      },
                      title: "Run in SamAgent (Enter)",
                    },
                    busy ? h("span", { className: "codex-spin" }, "⟳") : h(IconArrowUp)
                  )
                )
              )
            )
          )
        ),

        // ==================================================================
        // 4. RIGHT SLIDE-OUT INSPECTOR DRAWER (Diffs, VS Code, Dev App)
        // ==================================================================
        drawerOpen
          ? h(
              "aside",
              { className: "codex-right-drawer" },

              // Drawer Header
              h(
                "div",
                { className: "codex-drawer-header" },
                h("strong", { style: { fontSize: "13px", color: "#18181b" } }, "Inspector & Studio"),
                h(
                  "button",
                  {
                    style: { background: "none", border: "none", cursor: "pointer", fontSize: "16px", color: "#71717a" },
                    onClick: function () {
                      setDrawerOpen(false);
                    },
                  },
                  "✕"
                )
              ),

              // Drawer Tabs
              h(
                "div",
                { className: "codex-drawer-tabs" },
                h(
                  "button",
                  {
                    className: "codex-drawer-tab " + (drawerTab === "diffs" ? "active" : ""),
                    onClick: function () {
                      setDrawerTab("diffs");
                    },
                  },
                  "Diffs (" + (diffSum.files || []).length + ")"
                ),
                h(
                  "button",
                  {
                    className: "codex-drawer-tab " + (drawerTab === "vscode" ? "active" : ""),
                    onClick: function () {
                      setDrawerTab("vscode");
                    },
                  },
                  "VS Code"
                ),
                h(
                  "button",
                  {
                    className: "codex-drawer-tab " + (drawerTab === "preview" ? "active" : ""),
                    onClick: function () {
                      setDrawerTab("preview");
                    },
                  },
                  "Live App (:3000)"
                ),
                h(
                  "button",
                  {
                    className: "codex-drawer-tab " + (drawerTab === "gate" ? "active" : ""),
                    onClick: function () {
                      setDrawerTab("gate");
                    },
                  },
                  "Gate (7/7)"
                ),
                h(
                  "button",
                  {
                    className: "codex-drawer-tab " + (drawerTab === "git" ? "active" : ""),
                    onClick: function () {
                      setDrawerTab("git");
                    },
                  },
                  "Git / PR"
                )
              ),

              // Drawer Tab Bodies
              h(
                "div",
                { className: "codex-drawer-body" },

                // Tab: Diffs
                drawerTab === "diffs"
                  ? h(
                      "div",
                      { className: "codex-panel-card" },
                      h(
                        "div",
                        { className: "codex-panel-card-title" },
                        h("span", null, "Changed Files"),
                        h(
                          "span",
                          { style: { fontSize: "11px" } },
                          h("span", { style: { color: "#16a34a", fontWeight: 700 } }, "+" + (diffSum.total_additions || 0)),
                          " ",
                          h("span", { style: { color: "#dc2626", fontWeight: 700 } }, "-" + (diffSum.total_deletions || 0))
                        )
                      ),
                      h(
                        "div",
                        { style: { display: "flex", flexDirection: "column", gap: "6px" } },
                        (diffSum.files || []).map(function (f) {
                          return h(
                            "div",
                            {
                              key: f.path,
                              style: {
                                padding: "6px 8px",
                                background: "#ffffff",
                                border: "1px solid #e5e5e7",
                                borderRadius: "6px",
                                fontSize: "11px",
                                display: "flex",
                                justifyContent: "space-between",
                                cursor: "pointer",
                              },
                              onClick: function () {
                                loadFile(f.path);
                                setDrawerTab("vscode");
                              },
                            },
                            h("code", { style: { color: "#2563eb" } }, f.path),
                            h("span", { style: { color: "#71717a" } }, "+" + f.additions + " / -" + f.deletions)
                          );
                        })
                      ),
                      rawGitDiff
                        ? h(
                            "div",
                            { style: { marginTop: "10px" } },
                            h("div", { style: { fontSize: "11px", fontWeight: 600, color: "#71717a", marginBottom: "4px" } }, "Unified Git Diff:"),
                            h(
                              "pre",
                              { className: "diff-code-pre" },
                              rawGitDiff.split("\n").map(function (line, lIdx) {
                                var cls = line.startsWith("+") ? "diff-line-add" : (line.startsWith("-") ? "diff-line-del" : "");
                                return h("span", { key: lIdx, className: cls }, line + "\n");
                              })
                            )
                          )
                        : null
                    )
                  : null,

                // Tab: VS Code Studio
                drawerTab === "vscode"
                  ? h(
                      "div",
                      { className: "codex-panel-card" },
                      h(
                        "div",
                        { className: "codex-panel-card-title" },
                        h("span", null, "VS Code Studio: " + selectedFile.split("/").pop()),
                        h(
                          "button",
                          {
                            className: "codex-btn-vscode",
                            onClick: function () {
                              openVsCode(selectedFile);
                            },
                          },
                          "Open in VS Code ↗"
                        )
                      ),
                      // Quick file pills
                      h(
                        "div",
                        { className: "codex-file-pills" },
                        ["app/main.py", "app/models.py", "tests/test_acceptance.py", "Dockerfile", "README.md"].map(function (fp) {
                          var isSel = selectedFile === fp;
                          return h(
                            "button",
                            {
                              key: fp,
                              className: "codex-file-pill " + (isSel ? "active" : ""),
                              onClick: function () {
                                loadFile(fp);
                              },
                            },
                            fp.split("/").pop()
                          );
                        })
                      ),
                      h("textarea", {
                        style: {
                          width: "100%",
                          minHeight: "260px",
                          fontFamily: "var(--codex-font-mono)",
                          fontSize: "12px",
                          padding: "10px",
                          border: "1px solid #e5e5e7",
                          borderRadius: "6px",
                          background: "#ffffff",
                          resize: "vertical",
                          lineHeight: "1.45",
                        },
                        value: fileContent,
                        onChange: function (e) {
                          setFileContent(e.target.value);
                        },
                      }),
                      h(
                        "div",
                        { style: { display: "flex", gap: "8px", justifyContent: "space-between" } },
                        h(
                          "button",
                          {
                            style: { background: "none", border: "1px solid #e5e5e7", padding: "6px 12px", borderRadius: "6px", fontSize: "11px", cursor: "pointer", color: "#71717a" },
                            onClick: function () {
                              loadFile(selectedFile);
                              setNotice("Reloaded " + selectedFile + " from disk.");
                            },
                          },
                          "Revert"
                        ),
                        h(
                          "button",
                          {
                            className: "codex-btn-action",
                            disabled: busy,
                            onClick: saveFileAndVerify,
                          },
                          "Save to Disk & Re-Verify L0–L4"
                        )
                      )
                    )
                  : null,

                // Tab: Live Dev App (:3000)
                drawerTab === "preview"
                  ? h(
                      "div",
                      { className: "codex-panel-card" },
                      h(
                        "div",
                        { className: "codex-panel-card-title" },
                        h("span", null, "Live App Preview (Port 3000)"),
                        h(
                          "button",
                          {
                            style: { background: "none", border: "1px solid #e5e5e7", padding: "2px 8px", borderRadius: "4px", cursor: "pointer", fontSize: "11px" },
                            onClick: refresh,
                          },
                          "⟳ Refresh"
                        )
                      ),
                      h("iframe", {
                        style: { width: "100%", height: "240px", border: "1px solid #e5e5e7", borderRadius: "6px" },
                        srcDoc: state.preview_html || "<html><body><p>App running on port 3000</p></body></html>",
                        title: "App Preview",
                      }),
                      h("div", { style: { fontSize: "11px", fontWeight: 600, color: "#71717a", marginTop: "8px" } }, "Interactive RBAC / IDOR Sandbox:"),
                      h(
                        "div",
                        { style: { display: "flex", gap: "4px", flexWrap: "wrap" } },
                        h(
                          "button",
                          {
                            className: "codex-btn-action",
                            style: { background: devRole === "visitor" ? "#2563eb" : "#f4f4f5", color: devRole === "visitor" ? "#fff" : "#18181b", border: "1px solid #e5e5e7", fontSize: "10px" },
                            onClick: function () {
                              setDevRole("visitor");
                            },
                          },
                          "Visitor"
                        ),
                        h(
                          "button",
                          {
                            className: "codex-btn-action",
                            style: { background: devRole === "member" ? "#2563eb" : "#f4f4f5", color: devRole === "member" ? "#fff" : "#18181b", border: "1px solid #e5e5e7", fontSize: "10px" },
                            onClick: function () {
                              setDevRole("member");
                            },
                          },
                          "Member Alice"
                        ),
                        h(
                          "button",
                          {
                            className: "codex-btn-action",
                            style: { background: devRole === "admin" ? "#2563eb" : "#f4f4f5", color: devRole === "admin" ? "#fff" : "#18181b", border: "1px solid #e5e5e7", fontSize: "10px" },
                            onClick: function () {
                              setDevRole("admin");
                            },
                          },
                          "Admin"
                        )
                      ),
                      h(
                        "div",
                        { style: { display: "flex", gap: "6px", marginTop: "6px", flexWrap: "wrap" } },
                        h(
                          "button",
                          {
                            className: "codex-btn-action",
                            style: { fontSize: "10px" },
                            onClick: function () {
                              runDevAction("list_items");
                            },
                          },
                          "List Schedule"
                        ),
                        h(
                          "button",
                          {
                            className: "codex-btn-action",
                            style: { fontSize: "10px" },
                            onClick: function () {
                              runDevAction("create_booking");
                            },
                          },
                          "Book Class"
                        ),
                        h(
                          "button",
                          {
                            className: "codex-btn-action",
                            style: { fontSize: "10px", background: "#fef2f2", color: "#dc2626", border: "1px solid #fecaca" },
                            onClick: function () {
                              runDevAction("get_booking");
                            },
                          },
                          "Test IDOR Access"
                        )
                      ),
                      devResponse
                        ? h(
                            "pre",
                            { className: "diff-code-pre", style: { maxHeight: "140px", marginTop: "8px" } },
                            JSON.stringify(devResponse.response, null, 2)
                          )
                        : null
                    )
                  : null,

                // Tab: Gate
                drawerTab === "gate"
                  ? h(
                      "div",
                      { className: "codex-panel-card" },
                      h("div", { className: "codex-panel-card-title" }, "Pre-Production Gates (7 Checks)"),
                      h(
                        "div",
                        { style: { display: "flex", flexDirection: "column", gap: "6px" } },
                        Object.keys(preProd.checks || {}).map(function (k) {
                          var ok = Boolean(preProd.checks[k]);
                          return h(
                            "div",
                            {
                              key: k,
                              style: {
                                display: "flex",
                                justifyContent: "space-between",
                                padding: "6px 8px",
                                background: "#ffffff",
                                border: "1px solid #e5e5e7",
                                borderRadius: "6px",
                                fontSize: "11px",
                              },
                            },
                            h("code", null, k),
                            h("span", { style: { color: ok ? "#16a34a" : "#dc2626", fontWeight: 700 } }, ok ? "PASS" : "FAIL")
                          );
                        })
                      ),
                      h(
                        "div",
                        { style: { display: "flex", gap: "8px", marginTop: "10px", flexWrap: "wrap" } },
                        h(
                          "button",
                          {
                            style: { background: "#ffffff", border: "1px solid #e5e5e7", padding: "6px 12px", borderRadius: "6px", fontSize: "11px", cursor: "pointer", color: "#18181b", fontWeight: 600 },
                            disabled: busy,
                            onClick: function () {
                              setBusy(true);
                              apiGet("/state")
                                .then(function (data) {
                                  setState(data);
                                  setNotice("Pre-Production gates re-verified: 7/7 checks evaluated.");
                                })
                                .finally(function () {
                                  setBusy(false);
                                });
                            },
                          },
                          "⟳ Re-Verify All Gates"
                        ),
                        h(
                          "button",
                          {
                            className: "codex-btn-action",
                            style: { flex: 1, minWidth: "180px" },
                            disabled: busy || !preProd.ready_for_production,
                            onClick: function () {
                              setBusy(true);
                              apiPost("/promote-prod", {})
                                .then(function (res) {
                                  setNotice("Promoted to Production Release: Dockerfile & RELEASE_MANIFEST.json verified!");
                                  refresh();
                                })
                                .finally(function () {
                                  setBusy(false);
                                });
                            },
                          },
                          "Promote to Production"
                        )
                      )
                    )
                  : null,

                // Tab: Git / PR
                drawerTab === "git"
                  ? h(
                      "div",
                      { className: "codex-panel-card" },
                      h("div", { className: "codex-panel-card-title" }, "Git Sync & Pull Request"),
                      h("label", { style: { fontSize: "11px", color: "#71717a" } }, "Commit Message:"),
                      h("input", {
                        style: { width: "100%", padding: "6px 8px", border: "1px solid #e5e5e7", borderRadius: "6px", fontSize: "12px" },
                        value: commitMsg,
                        onChange: function (e) {
                          setCommitMsg(e.target.value);
                        },
                      }),
                      h(
                        "button",
                        { className: "codex-btn-action", onClick: handleGitHubSync },
                        "Commit & Push to Remote"
                      ),
                      h("label", { style: { fontSize: "11px", color: "#71717a", marginTop: "8px" } }, "PR Title:"),
                      h("input", {
                        style: { width: "100%", padding: "6px 8px", border: "1px solid #e5e5e7", borderRadius: "6px", fontSize: "12px" },
                        value: prTitle,
                        onChange: function (e) {
                          setPrTitle(e.target.value);
                        },
                      }),
                      h(
                        "button",
                        { className: "codex-btn-action", onClick: handleCreatePr },
                        "Create Verified GitHub PR"
                      )
                    )
                  : null
              )
            )
          : null
        ),

      // ======================================================================
      // 5. PROJECT FOLDER PICKER MODAL
      // ======================================================================
      folderModalOpen
        ? h(
            "div",
            {
              className: "codex-modal-backdrop",
              onClick: function () {
                setFolderModalOpen(false);
              },
            },
            h(
              "div",
              {
                className: "codex-modal",
                onClick: function (e) {
                  e.stopPropagation();
                },
              },
              h("strong", { style: { fontSize: "15px", color: "#18181b" } }, "Open Local Project Folder"),
              h("p", { style: { fontSize: "12px", color: "#71717a", margin: "0" } }, "Select or enter any local directory on your machine to open with SamAgent:"),
              h("input", {
                style: { width: "100%", padding: "8px 10px", border: "1px solid #e5e5e7", borderRadius: "6px", fontSize: "13px" },
                placeholder: "Paste local directory path...",
                value: customFolderPath,
                onChange: function (e) {
                  setCustomFolderPath(e.target.value);
                },
              }),
              h(
                "div",
                { style: { display: "flex", gap: "8px", justifyContent: "flex-end" } },
                h(
                  "button",
                  {
                    style: { background: "none", border: "1px solid #e5e5e7", padding: "6px 12px", borderRadius: "6px", cursor: "pointer", fontSize: "12px" },
                    onClick: function () {
                      setFolderModalOpen(false);
                    },
                  },
                  "Cancel"
                ),
                h(
                  "button",
                  {
                    className: "codex-btn-action",
                    disabled: !customFolderPath.trim(),
                    onClick: function () {
                      handleLoadFolder(customFolderPath);
                    },
                  },
                  "Open Workspace"
                )
              ),
              h("div", { style: { fontSize: "11px", fontWeight: 600, color: "#71717a", marginTop: "8px" } }, "Discovered Local Projects:"),
              h(
                "div",
                { style: { display: "flex", flexWrap: "wrap", gap: "6px" } },
                (folderBrowser.folders || []).map(function (fd) {
                  return h(
                    "button",
                    {
                      key: fd.path,
                      style: { background: "#f4f4f5", border: "1px solid #e5e5e7", padding: "4px 8px", borderRadius: "4px", fontSize: "11px", cursor: "pointer" },
                      onClick: function () {
                        handleLoadFolder(fd.path);
                      },
                    },
                    "📁 " + fd.name
                  );
                })
              )
            )
          )
        : null,

      // ======================================================================
      // 6. SETTINGS MODAL
      // ======================================================================
      settingsOpen
        ? h(
            "div",
            {
              className: "codex-modal-backdrop",
              onClick: function () {
                setSettingsOpen(false);
              },
            },
            h(
              "div",
              {
                className: "codex-modal",
                onClick: function (e) {
                  e.stopPropagation();
                },
              },
              h("strong", { style: { fontSize: "15px", color: "#18181b" } }, "SamAgent Desktop Settings"),
              h("div", { style: { fontSize: "12px", color: "#52525b" } }, "● Platform: SamAgent Local Engine (FastAPI 127.0.0.1:8080)"),
              h("div", { style: { fontSize: "12px", color: "#52525b" } }, "● VS Code Studio Extension: Connected"),
              h("div", { style: { fontSize: "12px", color: "#52525b" } }, "● Workspace: " + (state && state.workspace)),
              h(
                "div",
                { style: { marginTop: "10px" } },
                h(
                  "button",
                  {
                    className: "codex-btn-vscode",
                    onClick: handleInstallPlatform,
                  },
                  "One-Click Install OS Desktop Launcher & VS Code Extension"
                )
              ),
              h(
                "div",
                { style: { display: "flex", justifyContent: "flex-end", marginTop: "14px" } },
                h(
                  "button",
                  {
                    className: "codex-btn-action",
                    onClick: function () {
                      setSettingsOpen(false);
                    },
                  },
                  "Close"
                )
              )
            )
          )
        : null
    );
  }

  // Register in both namespaces so it works in standalone SamAgent & plugin host
  if (window.__SAMAGENT_PLUGINS__ && typeof window.__SAMAGENT_PLUGINS__.register === "function") {
    window.__SAMAGENT_PLUGINS__.register("samagent", SamAgentCodexDesktopApp);
  }
  if (window.__HERMES_PLUGINS__ && typeof window.__HERMES_PLUGINS__.register === "function") {
    window.__HERMES_PLUGINS__.register("samagent", SamAgentCodexDesktopApp);
  }
})();

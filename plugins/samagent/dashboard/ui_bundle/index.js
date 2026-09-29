/* SamAgent Local Platform & VS Code Pre-Production Studio (Zero-CDN UMD Bundle) */
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

  function SamAgentMissionControl() {
    var _s = useState(null), state = _s[0], setState = _s[1];
    var _t = useState("vscode"), tab = _t[0], setTab = _t[1]; // vscode | live | brief | plan | ledger
    var _b = useState(false), busy = _b[0], setBusy = _b[1];
    var _err = useState(""), errMsg = _err[0], setErrMsg = _err[1];
    var _notice = useState(""), notice = _notice[0], setNotice = _notice[1];

    var _brief = useState(
      "Booking site for my yoga studio where visitors see the schedule, members book classes (no double booking, private per member), and admins add classes."
    );
    var briefInput = _brief[0], setBriefInput = _brief[1];
    var _qs = useState([]), questions = _qs[0], setQuestions = _qs[1];
    var _ans = useState({}), answers = _ans[0], setAnswers = _ans[1];
    var _pol = useState("default"), routerPolicy = _pol[0], setRouterPolicy = _pol[1];
    var _aut = useState("milestones"), autonomy = _aut[0], setAutonomy = _aut[1];

    // Local Workspace & VS Code Studio state
    var _nw = useState(""), newWsSlug = _nw[0], setNewWsSlug = _nw[1];
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
        })
        .catch(function (e) {
          setErrMsg(String(e));
        });
    }

    useEffect(function () {
      refresh();
      loadFile("app/main.py");
    }, []);

    function openVsCode(relPath) {
      apiPost("/ide/open", { rel_path: relPath || null, line: 1 })
        .then(function (res) {
          setNotice(
            "VS Code Ready: " +
              (res.vscode_uri || "") +
              " · Terminal fallback: " +
              (res.cli_fallback || "")
          );
          if (res.vscode_uri) {
            window.open(res.vscode_uri, "_blank");
          }
        })
        .catch(function (e) {
          setErrMsg(String(e));
        });
    }

    function switchWorkspace(nameOrPath) {
      if (!nameOrPath) return;
      setBusy(true);
      apiPost("/workspace/switch", { workspace_name_or_path: nameOrPath, brief: briefInput })
        .then(function (data) {
          setState(data);
          setNewWsSlug("");
          loadFile("app/main.py");
          setNotice("Switched active local workspace to " + data.workspace);
        })
        .catch(function (e) {
          setErrMsg(String(e));
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function saveFileAndVerify() {
      setBusy(true);
      apiPost("/ide/file", {
        rel_path: selectedFile,
        content: fileContent,
        auto_reverify: true,
      })
        .then(function (data) {
          setState(data);
          setNotice("Saved " + selectedFile + " to local disk and re-verified L0–L4 Pre-Prod Gate.");
        })
        .catch(function (e) {
          setErrMsg(String(e));
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function syncFromVsCodeAndVerify() {
      setBusy(true);
      apiPost("/reverify", {})
        .then(function (data) {
          setState(data);
          loadFile(selectedFile);
          setNotice("Synced edits from VS Code on disk and re-ran L0–L4 + OWASP Pre-Prod Gate.");
        })
        .catch(function (e) {
          setErrMsg(String(e));
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
            setNotice(
              "Promoted to Production Release " +
                pr.release_id +
                "! Generated Dockerfile, docker-compose.prod.yml, .env.example & signed RELEASE_MANIFEST.json."
            );
          } else {
            setErrMsg(
              "Production Promotion Blocked by Pre-Prod Gate: " +
                ((pr.blockers || []).join(", ") || "Verification failed")
            );
          }
        })
        .catch(function (e) {
          setErrMsg(String(e));
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function installPlatform() {
      setBusy(true);
      apiPost("/platform/install", {})
        .then(function (data) {
          setState(data);
          setNotice(
            "Installed Local Platform Desktop App, Background Service, ~/SamAgentProjects, and VS Code Extension!"
          );
        })
        .catch(function (e) {
          setErrMsg(String(e));
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function runDevAction(action, extra) {
      setBusy(true);
      var payload = Object.assign(
        {
          action: action,
          role: devRole,
          user_id: devUserId,
          item_id: devItemId,
          title: devNewTitle,
        },
        extra || {}
      );
      apiPost("/dev-app/action", payload)
        .then(function (res) {
          setDevOut(res);
          refresh();
        })
        .catch(function (e) {
          setErrMsg(String(e));
        })
        .finally(function () {
          setBusy(false);
        });
    }

    function controlDevServer(action) {
      setBusy(true);
      apiPost("/dev-server/control", { action: action, port: 3000 })
        .then(function (data) {
          setState(data);
          var ds = data.dev_server || {};
          setNotice(
            "Local Dev Server (Port 3000): " +
              (ds.running ? "ONLINE at " + ds.url : "STOPPED")
          );
        })
        .catch(function (e) {
          setErrMsg(String(e));
        })
        .finally(function () {
          setBusy(false);
        });
    }

    if (!state) {
      return h("div", { className: "sam-root" }, h("p", null, "Loading SamAgent Local Platform..."));
    }

    var spec = state.spec || {};
    var planCard = state.plan_card || {};
    var deliv = state.deliverable || {};
    var ver = deliv.verification || {};
    var preProd = state.pre_prod_gate || {};
    var ide = state.ide || {};
    var watcher = ide.watcher || {};
    var devSrv = state.dev_server || {};
    var devApp = state.dev_app || { items: [], bookings: [] };
    var releases = state.releases || [];
    var pInst = state.platform_install || {};
    var ledger = state.ledger || {};
    var meas = state.measurements || {};

    return h(
      "div",
      { className: "sam-root" },
      // Header
      h(
        "div",
        { className: "sam-header" },
        h(
          "div",
          null,
          h(
            "h1",
            { className: "sam-title" },
            "SamAgent Local Platform",
            h("span", { className: "sam-pill sam-pill-blue" }, "Installed Local Platform · VS Code Synced"),
            h(
              "span",
              { className: "sam-pill " + (devSrv.running ? "sam-pill-green" : "sam-pill-amber") },
              devSrv.running ? "DEV SERVER :3000 ONLINE" : "DEV SERVER :3000 IDLE"
            ),
            pill(Boolean(preProd.ready_for_production), preProd.ready_for_production ? "PRE-PROD GATE READY" : "PRE-PROD BLOCKED")
          ),
          h(
            "p",
            { className: "sam-subtitle" },
            "Active Local Workspace on Disk: ",
            h("code", { className: "sam-mono" }, state.workspace || ""),
            " · Auto-Watching ",
            h("strong", null, String(watcher.tracked_file_count || 0)),
            " files in VS Code / Cursor / ACP"
          )
        ),
        h(
          "div",
          { style: { display: "flex", gap: "8px", flexWrap: "wrap" } },
          h(
            "button",
            {
              className: "sam-btn sam-btn-primary",
              onClick: function () {
                openVsCode(null);
              },
            },
            "Open Workspace in VS Code"
          ),
          h(
            "a",
            {
              className: "sam-btn",
              style: { textDecoration: "none" },
              href: ide.cursor_workspace_uri || "#",
            },
            "Open in Cursor"
          ),
          h(
            "button",
            {
              className: "sam-btn",
              disabled: busy,
              onClick: function () {
                controlDevServer("restart");
              },
            },
            "Restart Dev Server (:3000)"
          ),
          h(
            "button",
            {
              className: "sam-btn",
              disabled: busy,
              onClick: syncFromVsCodeAndVerify,
            },
            "Sync from VS Code & Verify"
          ),
          h(
            "button",
            {
              className: "sam-btn sam-btn-success",
              disabled: busy || !preProd.ready_for_production,
              onClick: promoteToProduction,
            },
            "Promote to Production Release"
          )
        )
      ),

      // Local Workspace Switcher Bar
      h(
        "div",
        {
          className: "sam-card",
          style: {
            marginBottom: "14px",
            padding: "10px 14px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "10px",
          },
        },
        h(
          "div",
          { style: { display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap", fontSize: "12px" } },
          h("strong", null, "Local Projects (~/SamAgentProjects):"),
          (state.available_workspaces || []).map(function (w) {
            return h(
              "button",
              {
                key: w.path,
                className: "sam-btn " + (w.active ? "sam-btn-primary" : ""),
                onClick: function () {
                  switchWorkspace(w.path);
                },
              },
              w.name
            );
          })
        ),
        h(
          "div",
          { style: { display: "flex", gap: "6px", alignItems: "center" } },
          h("input", {
            className: "sam-input",
            style: { width: "190px" },
            placeholder: "New local project name...",
            value: newWsSlug,
            onChange: function (e) {
              setNewWsSlug(e.target.value);
            },
          }),
          h(
            "button",
            {
              className: "sam-btn",
              disabled: busy || !newWsSlug.trim(),
              onClick: function () {
                switchWorkspace(newWsSlug);
              },
            },
            "+ Create Local Workspace"
          )
        )
      ),

      errMsg
        ? h(
            "div",
            { className: "sam-card", style: { borderColor: "#ef4444", marginBottom: "12px", color: "#f87171" } },
            "Error: " + errMsg
          )
        : null,

      notice
        ? h(
            "div",
            {
              className: "sam-card",
              style: {
                borderColor: "#3b82f6",
                marginBottom: "12px",
                padding: "10px 14px",
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

      // Navigation Tabs
      h(
        "div",
        { className: "sam-nav" },
        h(
          "button",
          {
            className: "sam-tab " + (tab === "vscode" ? "active" : ""),
            onClick: function () {
              setTab("vscode");
            },
          },
          "1. VS Code Studio & Pre-Prod Gate"
        ),
        h(
          "button",
          {
            className: "sam-tab " + (tab === "live" ? "active" : ""),
            onClick: function () {
              setTab("live");
            },
          },
          "2. Live Local Dev & Multi-Role Sandbox"
        ),
        h(
          "button",
          {
            className: "sam-tab " + (tab === "brief" ? "active" : ""),
            onClick: function () {
              setTab("brief");
            },
          },
          "3. App Brief & Spec Interview"
        ),
        h(
          "button",
          {
            className: "sam-tab " + (tab === "plan" ? "active" : ""),
            onClick: function () {
              setTab("plan");
            },
          },
          "4. Plan Card & Contract"
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

      // TAB 1: VS Code Studio & Pre-Production Gate
      tab === "vscode"
        ? h(
            "div",
            { className: "sam-grid-2" },
            // Left card: Workspace Files + Live Bidirectional Editor
            h(
              "div",
              { className: "sam-card" },
              h(
                "div",
                { className: "sam-card-title" },
                h("span", null, "Local Workspace Files (.vscode/ configured)"),
                h(
                  "button",
                  {
                    className: "sam-btn sam-btn-primary",
                    onClick: function () {
                      openVsCode(selectedFile);
                    },
                  },
                  "Open " + selectedFile + " in VS Code"
                )
              ),
              h(
                "div",
                { className: "sam-list", style: { maxHeight: "175px", overflow: "auto", marginBottom: "12px" } },
                (ide.files || []).map(function (f) {
                  return h(
                    "div",
                    {
                      key: f.rel_path,
                      className: "sam-list-item",
                      style: {
                        cursor: "pointer",
                        borderColor: selectedFile === f.rel_path ? "#3b82f6" : "#1e293b",
                      },
                      onClick: function () {
                        loadFile(f.rel_path);
                      },
                    },
                    h(
                      "div",
                      null,
                      h("code", { className: "sam-mono" }, f.rel_path),
                      h("span", { style: { fontSize: "11px", color: "#94a3b8", marginLeft: "8px" } }, "(" + f.size_bytes + " B)")
                    ),
                    h(
                      "div",
                      { style: { display: "flex", gap: "8px", alignItems: "center" } },
                      h("span", { className: "sam-pill sam-pill-purple" }, f.category),
                      h(
                        "a",
                        {
                          href: f.vscode_uri,
                          style: { color: "#60a5fa", fontSize: "11px", textDecoration: "none" },
                          onClick: function (e) {
                            e.stopPropagation();
                          },
                        },
                        "vscode:// ↗"
                      )
                    )
                  );
                })
              ),
              h(
                "div",
                { className: "sam-card-title" },
                h("span", null, "Live File Inspector / Editor: " + selectedFile),
                h(
                  "button",
                  {
                    className: "sam-btn sam-btn-success",
                    disabled: busy,
                    onClick: saveFileAndVerify,
                  },
                  "Save to Disk & Re-Verify L0–L4"
                )
              ),
              h("textarea", {
                className: "sam-textarea sam-mono",
                style: { minHeight: "230px", fontSize: "12px" },
                value: fileContent,
                onChange: function (e) {
                  setFileContent(e.target.value);
                },
              })
            ),

            // Right card: Pre-Production Gate Checklist + Git Diff + Production Releases + Installer
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
                "p",
                { style: { fontSize: "12px", color: "#94a3b8", marginTop: 0 } },
                "Blocks production deployment until your local development build and any VS Code edits pass all 7 verification & OWASP security gates."
              ),
              h(
                "div",
                { className: "sam-list", style: { marginBottom: "14px" } },
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
                h("span", null, "Signed Production Releases (" + releases.length + ")"),
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
              releases.length > 0
                ? h(
                    "div",
                    { className: "sam-list", style: { marginBottom: "14px" } },
                    releases.slice(0, 3).map(function (rel) {
                      return h(
                        "div",
                        { key: rel.release_id, className: "sam-list-item" },
                        h(
                          "span",
                          null,
                          rel.release_id +
                            " · Commit: " +
                            rel.git_commit +
                            " · Artifacts: " +
                            (rel.artifacts || []).join(", ")
                        ),
                        pill(true, "SIGNED")
                      );
                    })
                  )
                : h(
                    "div",
                    { style: { fontSize: "12px", color: "#94a3b8", marginBottom: "14px" } },
                    "No production releases promoted yet. Test in VS Code & click 'Generate Production Bundle'."
                  ),
              h(
                "div",
                { className: "sam-card-title" },
                h(
                  "span",
                  null,
                  "Live Git Status & Diff (Branch: " +
                    ((ide.git && ide.git.branch) || "main") +
                    " · Commit: " +
                    ((ide.git && ide.git.head_commit) || "HEAD") +
                    ")"
                )
              ),
              ide.git && ide.git.changed_files && ide.git.changed_files.length > 0
                ? h("pre", { className: "sam-pre", style: { marginBottom: "14px" } }, ide.git.diff || "")
                : h(
                    "div",
                    { style: { fontSize: "12px", color: "#34d399", marginBottom: "14px" } },
                    "Working tree clean — all local VS Code edits verified and in sync."
                  ),
              h(
                "div",
                { className: "sam-card-title" },
                h("span", null, "Local Machine Platform & VS Code Extension Installer"),
                h(
                  "button",
                  {
                    className: "sam-btn sam-btn-primary",
                    disabled: busy,
                    onClick: installPlatform,
                  },
                  pInst.installed ? "Repair Local Platform & VS Code Bridge" : "Install Local Platform & VS Code Bridge"
                )
              ),
              h(
                "div",
                { className: "sam-list" },
                h(
                  "div",
                  { className: "sam-list-item" },
                  h("span", null, "Persistent Projects Folder (~/SamAgentProjects)"),
                  pill(Boolean(pInst.projects_root_exists), "Disk")
                ),
                h(
                  "div",
                  { className: "sam-list-item" },
                  h("span", null, "VS Code / Cursor Extension (samjuniors.samagent-vscode)"),
                  pill(Boolean(pInst.vscode_extension_installed), "VS Code")
                ),
                h(
                  "div",
                  { className: "sam-list-item" },
                  h("span", null, "OS Desktop App & Background Daemon (SamAgent.app / systemd)"),
                  pill(Boolean(pInst.desktop_app_installed), "Desktop App")
                )
              )
            )
          )
        : null,

      // TAB 2: Live Local Dev & Multi-Role Sandbox
      tab === "live"
        ? h(
            "div",
            { className: "sam-grid-2" },
            h(
              "div",
              { className: "sam-card" },
              h(
                "div",
                { className: "sam-card-title" },
                h("span", null, "Live Local Dev App Preview & Multi-Role Testing Sandbox"),
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
                { style: { marginTop: "12px", display: "flex", gap: "6px", flexWrap: "wrap" } },
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
                { style: { display: "flex", gap: "8px", marginTop: "10px", flexWrap: "wrap" } },
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
              h(
                "div",
                { style: { marginTop: "10px", fontSize: "12px", fontWeight: 700 } },
                "Live Local SQLite Bookings (" + (devApp.bookings || []).length + ") — Click to test IDOR protection:"
              ),
              h(
                "div",
                { className: "sam-list", style: { marginTop: "6px" } },
                (devApp.bookings || []).map(function (b) {
                  return h(
                    "div",
                    { key: b.id, className: "sam-list-item" },
                    h("span", null, b.id + " · Class: " + b.item_id + " · Owner: " + b.owner_id),
                    h(
                      "button",
                      {
                        className: "sam-btn",
                        onClick: function () {
                          runDevAction("get_booking", { booking_id: b.id });
                        },
                      },
                      "Inspect as " + devRole + " (" + (devUserId || "anon") + ")"
                    )
                  );
                })
              ),
              devOut
                ? h(
                    "pre",
                    { className: "sam-pre", style: { marginTop: "10px" } },
                    JSON.stringify(devOut.response, null, 2)
                  )
                : null
            ),

            // Right card: L0-L4 Verification Pyramid
            h(
              "div",
              { className: "sam-card" },
              h(
                "div",
                { className: "sam-card-title" },
                h("span", null, "Verification Pyramid (L0–L4) & OWASP Security"),
                pill(Boolean(ver.all_passed), ver.all_passed ? "ALL GATES GREEN" : "BLOCKED")
              ),
              h(
                "div",
                { className: "sam-kpi-row" },
                h(
                  "div",
                  { className: "sam-kpi" },
                  h("div", { className: "sam-kpi-label" }, "L0 Syntax"),
                  h("div", { className: "sam-kpi-value" }, ver.l0_compile_passed ? "PASS" : "FAIL")
                ),
                h(
                  "div",
                  { className: "sam-kpi" },
                  h("div", { className: "sam-kpi-label" }, "L1 Contract"),
                  h("div", { className: "sam-kpi-value" }, ver.l1_unit_passed ? "PASS" : "FAIL")
                ),
                h(
                  "div",
                  { className: "sam-kpi" },
                  h("div", { className: "sam-kpi-label" }, "L2 Ownership"),
                  h("div", { className: "sam-kpi-value" }, ver.l2_contract_passed ? "PASS" : "FAIL")
                ),
                h(
                  "div",
                  { className: "sam-kpi" },
                  h("div", { className: "sam-kpi-label" }, "L4 DOM Smoke"),
                  h("div", { className: "sam-kpi-value" }, ver.l4_browser_smoke && ver.l4_browser_smoke.passed ? "PASS" : "FAIL")
                )
              ),
              h(
                "div",
                { className: "sam-list" },
                (ver.story_results || []).map(function (sr) {
                  return h(
                    "div",
                    { key: sr.story_id, className: "sam-list-item" },
                    h("span", null, "[" + sr.story_id + "] (" + sr.role + ") " + sr.method + " " + sr.route + " — " + sr.accept),
                    pill(Boolean(sr.passed), sr.story_id)
                  );
                })
              )
            )
          )
        : null,

      // TAB 3: App Brief & Interview
      tab === "brief"
        ? h(
            "div",
            { className: "sam-grid-2" },
            h(
              "div",
              { className: "sam-card" },
              h("h2", { className: "sam-card-title" }, "Describe the App to Build in Your Local Workspace"),
              h("textarea", {
                className: "sam-textarea",
                value: briefInput,
                onChange: function (e) {
                  setBriefInput(e.target.value);
                },
              }),
              h(
                "div",
                { className: "sam-actions" },
                h(
                  "button",
                  {
                    className: "sam-btn",
                    disabled: busy,
                    onClick: function () {
                      setBusy(true);
                      apiPost("/interview", { brief: briefInput })
                        .then(function (res) {
                          setQuestions(res.questions || []);
                        })
                        .finally(function () {
                          setBusy(false);
                        });
                    },
                  },
                  "Ask Clarifying Questions (≤5)"
                ),
                h(
                  "button",
                  {
                    className: "sam-btn sam-btn-success",
                    disabled: busy,
                    onClick: function () {
                      setBusy(true);
                      apiPost("/plan", {
                        brief: briefInput,
                        answers: answers,
                        router_policy: routerPolicy,
                        autonomy: autonomy,
                      })
                        .then(function () {
                          return apiPost("/build", { autonomy: autonomy, router_policy: routerPolicy });
                        })
                        .then(function (built) {
                          setState(built);
                          setTab("vscode");
                        })
                        .finally(function () {
                          setBusy(false);
                        });
                    },
                  },
                  "Approve & Build to Local Workspace"
                )
              )
            ),
            h(
              "div",
              { className: "sam-card" },
              h("h2", { className: "sam-card-title" }, "Active Spec (.samagent/spec.yaml)"),
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
            )
          )
        : null,

      // TAB 4: Plan Card & Contract
      tab === "plan"
        ? h(
            "div",
            { className: "sam-grid-2" },
            h(
              "div",
              { className: "sam-card" },
              h("h2", { className: "sam-card-title" }, "1-Screen Plan Card & Isolated Worktree Waves"),
              h(
                "div",
                { className: "sam-list" },
                (planCard.modules || []).map(function (m) {
                  return h(
                    "div",
                    { key: m.module, className: "sam-list-item" },
                    h("span", null, m.module + " → owns " + (m.paths || []).join(", ")),
                    h("span", { className: "sam-pill sam-pill-purple" }, (m.depends_on || []).length ? "Wave 2" : "Wave 1")
                  );
                })
              )
            ),
            h(
              "div",
              { className: "sam-card" },
              h("h2", { className: "sam-card-title" }, "Frozen Contract (.samagent/contract/db/schema.sql)"),
              h("pre", { className: "sam-pre" }, (state.contracts && state.contracts["db/schema.sql"]) || "")
            )
          )
        : null,

      // TAB 5: Persistent Ledger & Benchmarks
      tab === "ledger"
        ? h(
            "div",
            { className: "sam-grid-2" },
            h(
              "div",
              { className: "sam-card" },
              h("h2", { className: "sam-card-title" }, "Bi-Temporal Project Ledger (.samagent/ledger.db)"),
              h(
                "div",
                { className: "sam-list" },
                (ledger.active_facts || []).map(function (f) {
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
              h("h2", { className: "sam-card-title" }, "SamBench-v0 & Ablation H1–H7 Summary"),
              h(
                "pre",
                { className: "sam-pre", style: { maxHeight: "260px" } },
                JSON.stringify(meas.ablation_h1_h7 && meas.ablation_h1_h7.hypotheses, null, 2)
              )
            )
          )
        : null
    );
  }

  if (window.__HERMES_PLUGINS__ && typeof window.__HERMES_PLUGINS__.register === "function") {
    window.__HERMES_PLUGINS__.register("samagent", SamAgentMissionControl);
  }
})();

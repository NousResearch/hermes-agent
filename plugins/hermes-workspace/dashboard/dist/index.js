(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK) return;

  const React = SDK.React;
  const hooks = SDK.hooks;
  const components = SDK.components;
  const Button = components.Button;
  const Badge = components.Badge;

  const h = React.createElement;

  const WORKSPACE_API = "/api/plugins/hermes-workspace";
  const PROMPTS = [
    {
      label: "Orchestrer",
      text: "Agis comme orchestrateur Hermes. Decoupe mon objectif en taches Kanban, assigne les bons profils, precise ready/done, risques, dependances et commandes de verification.",
    },
    {
      label: "Debug",
      text: "Diagnostique Hermes dashboard/chat/Kanban/gateway. Donne les hypotheses, commandes locales, endpoints a tester, logs a lire et corrections minimales.",
    },
    {
      label: "Plugins",
      text: "Propose les plugins Hermes utiles pour enrichir ce workspace: memoire, observabilite, providers, outils, UX dashboard, securite locale et automatisation.",
    },
    {
      label: "Workers",
      text: "Cree une strategie multi-agent: pilot orchestre, ops debug, docs documente, research cartographie, cyber audite, growth/business monetise.",
    },
  ];

  function number(value) {
    return Number.isFinite(Number(value)) ? Number(value) : 0;
  }

  function countBy(list, predicate) {
    return (list || []).filter(predicate).length;
  }

  function statusText(data) {
    const gateway = data && data.gateway && data.gateway.state;
    if (gateway && gateway.state) return gateway.state;
    if (gateway && gateway.platforms) return "running";
    return "unknown";
  }

  function nav(path) {
    window.location.href = path;
  }

  function copy(text, setMessage) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        setMessage("Prompt copie");
        setTimeout(function () { setMessage(""); }, 1800);
      }).catch(function () {
        setMessage(text);
      });
      return;
    }
    setMessage(text);
  }

  function useSummary() {
    const useState = hooks.useState;
    const useEffect = hooks.useEffect;
    const useCallback = hooks.useCallback;
    const state = useState({ loading: true, error: "", data: null });
    const current = state[0];
    const setCurrent = state[1];

    const refresh = useCallback(function () {
      setCurrent(function (prev) {
        return { loading: true, error: "", data: prev.data };
      });
      SDK.fetchJSON(WORKSPACE_API + "/summary")
        .then(function (data) {
          setCurrent({ loading: false, error: "", data: data });
        })
        .catch(function (err) {
          setCurrent({ loading: false, error: String(err && err.message ? err.message : err), data: null });
        });
    }, []);

    useEffect(function () {
      refresh();
      const timer = setInterval(refresh, 15000);
      return function () { clearInterval(timer); };
    }, [refresh]);

    return { summary: current, refresh: refresh };
  }

  function Kpi(props) {
    return h("div", { className: "hw-kpi" },
      h("span", { className: "hw-muted hw-small" }, props.label),
      h("strong", null, props.value),
      h("span", { className: "hw-muted hw-small" }, props.detail || "")
    );
  }

  function Pill(props) {
    return h("span", { className: "hw-pill", "data-state": props.state || "" },
      props.children
    );
  }

  function WorkspaceHero(props) {
    const data = props.data;
    const profiles = data.profiles || [];
    const activeProfiles = data.active_profiles || [];
    const catalogLabel = data.catalog_label || "Personal Agents";
    const dashboardPlugins = (data.plugins && data.plugins.dashboard) || [];
    const agentPlugins = (data.plugins && data.plugins.agent) || [];
    const boards = (data.kanban && data.kanban.boards) || [];
    const activeBoard = data.kanban && data.kanban.active_board;
    const currentBoard = boards.find(function (b) { return b.slug === activeBoard; }) || {};
    const installedActive = countBy(activeProfiles, function (p) { return p.installed; });
    const enabledAgentPlugins = countBy(agentPlugins, function (p) { return p.status === "enabled"; });

    return h("section", { className: "hw-hero" },
      h("div", { className: "hw-grid" },
        h("div", null,
          h("h1", { className: "hw-title" }, "Hermes Workspace"),
          h("p", { className: "hw-muted" },
            "Cockpit local pour piloter profils, plugins, memoire, debug, Kanban et workers Hermes."
          )
        ),
        h("div", { className: "hw-pills" },
          h(Pill, { state: statusText(data) === "running" ? "ok" : "warn" }, "Gateway: " + statusText(data)),
          h(Pill, { state: "ok" }, "Board: " + (activeBoard || "default")),
          h(Pill, null, "Memoire: " + ((data.memory && data.memory.provider) || "default")),
          h(Pill, null, "Workspace V2: " + (data.workspace_v2 && data.workspace_v2.exists ? "present" : "absent"))
        ),
        h("div", { className: "hw-actions" },
          h(Button, { onClick: function () { nav("/kanban"); } }, "Ouvrir Kanban"),
          h(Button, { onClick: function () { nav("/chat"); } }, "Ouvrir Chat"),
          h(Button, { onClick: function () { nav("/plugins"); } }, "Plugins"),
          h(Button, { onClick: props.refresh }, "Rafraichir")
        )
      ),
      h("div", { className: "hw-grid hw-grid-2" },
        h(Kpi, { label: "Profils actifs", value: installedActive + "/" + activeProfiles.length, detail: profiles.length + " profils installes" }),
        h(Kpi, { label: catalogLabel, value: (data.catalog_agents || []).length, detail: "catalogue local" }),
        h(Kpi, { label: "Plugins dashboard", value: dashboardPlugins.length, detail: "dont Workspace et Kanban" }),
        h(Kpi, { label: "Plugins agents", value: enabledAgentPlugins + "/" + agentPlugins.length, detail: "actifs / visibles" }),
        h(Kpi, { label: "Skills profils", value: data.skills ? data.skills.total_profile_skills : 0, detail: "SKILL.md installes" }),
        h(Kpi, { label: "Taches board", value: currentBoard.total || 0, detail: activeBoard || "default" })
      )
    );
  }

  function AgentMatrix(props) {
    const data = props.data;
    const profiles = data.profiles || [];
    const activeNames = new Set((data.active_profiles || []).map(function (p) { return p.name; }));

    return h("section", { className: "hw-section" },
      h("div", { className: "hw-section-head" },
        h("div", null,
          h("h2", null, "Agents / profils"),
          h("span", { className: "hw-muted hw-small" }, "Etat des profils Hermes et mapping du catalogue local.")
        ),
        h("div", { className: "hw-actions" },
          h(Button, { onClick: function () { nav("/profiles"); } }, "Profils"),
          h(Button, { onClick: function () { nav("/skills"); } }, "Skills")
        )
      ),
      h("div", { className: "hw-grid hw-grid-3" },
        h("div", null,
          h("h3", null, "Hermes actifs"),
          h("div", { className: "hw-pills" },
            (data.active_profiles || []).map(function (agent) {
              return h(Pill, { key: agent.name, state: agent.installed ? "ok" : "warn" }, agent.name);
            })
          )
        ),
        h("div", null,
          h("h3", null, data.catalog_label || "Personal Agents"),
          h("div", { className: "hw-pills" },
            (data.catalog_agents || []).map(function (agent) {
              return h(Pill, { key: agent.name, state: agent.installed_profile ? "ok" : "" }, agent.name);
            })
          )
        ),
        h("div", null,
          h("h3", null, "Profils installes"),
          profiles.map(function (profile) {
            return h("div", { className: "hw-row", key: profile.name },
              h("div", { className: "hw-row-main" },
                h("strong", null, profile.name),
                h("span", { className: "hw-muted hw-small" },
                  (profile.provider || "provider ?") + " / " + (profile.model || "model ?")
                )
              ),
              h("div", { className: "hw-pills" },
                h(Pill, { state: activeNames.has(profile.name) ? "ok" : "" }, profile.skill_count + " skills"),
                profile.gateway_running ? h(Pill, { state: "ok" }, "gateway") : null
              )
            );
          })
        )
      )
    );
  }

  function PluginsPanel(props) {
    const data = props.data;
    const dashboard = (data.plugins && data.plugins.dashboard) || [];
    const agent = (data.plugins && data.plugins.agent) || [];
    const providers = (data.plugins && data.plugins.providers) || {};

    return h("section", { className: "hw-section" },
      h("div", { className: "hw-section-head" },
        h("div", null,
          h("h2", null, "Plugins / providers"),
          h("span", { className: "hw-muted hw-small" }, "Inventaire runtime, dashboard, memoire et contexte.")
        ),
        h(Button, { onClick: function () { nav("/plugins"); } }, "Manager plugins")
      ),
      h("div", { className: "hw-grid hw-grid-2" },
        h("div", null,
          h("h3", null, "Dashboard"),
          dashboard.map(function (p) {
            return h("div", { className: "hw-row", key: p.name },
              h("div", { className: "hw-row-main" },
                h("strong", null, p.label || p.name),
                h("span", { className: "hw-muted hw-small" }, p.description || p.name)
              ),
              h(Pill, { state: p.has_api ? "ok" : "" }, p.has_api ? "api" : "ui")
            );
          })
        ),
        h("div", null,
          h("h3", null, "Agent plugins"),
          h("div", { className: "hw-pills" },
            h(Pill, null, "memory: " + (providers.memory_provider || "default")),
            h(Pill, null, "context: " + (providers.context_engine || "default"))
          ),
          agent.slice(0, 18).map(function (p) {
            return h("div", { className: "hw-row", key: p.name },
              h("div", { className: "hw-row-main" },
                h("strong", null, p.name),
                h("span", { className: "hw-muted hw-small" }, p.description || p.source)
              ),
              h(Pill, { state: p.status === "enabled" ? "ok" : p.status === "disabled" ? "warn" : "" }, p.status)
            );
          })
        )
      )
    );
  }

  function WorkflowLaunchers(props) {
    const useState = hooks.useState;
    const data = props.data;
    const state = useState("");
    const message = state[0];
    const setMessage = state[1];
    const busyState = useState("");
    const busy = busyState[0];
    const setBusy = busyState[1];

    function launch(key) {
      setBusy(key);
      setMessage("");
      SDK.fetchJSON(WORKSPACE_API + "/blueprints/" + encodeURIComponent(key) + "/launch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspace: "workspace_v2", priority: 9, triage: true }),
      }).then(function (result) {
        const ids = (result.created || []).map(function (x) { return x.task_id; }).join(", ");
        setMessage("Pack cree: " + ids);
        props.refresh();
      }).catch(function (err) {
        setMessage("Erreur: " + String(err && err.message ? err.message : err));
      }).finally(function () {
        setBusy("");
      });
    }

    return h("section", { className: "hw-section" },
      h("div", { className: "hw-section-head" },
        h("div", null,
          h("h2", null, "Launchers Kanban"),
          h("span", { className: "hw-muted hw-small" }, "Packs de taches pre-assignees aux workers specialises.")
        ),
        h("div", { className: "hw-actions" },
          h(Button, { onClick: function () { nav("/kanban"); } }, "Voir board"),
          message ? h("span", { className: "hw-muted hw-small" }, message) : null
        )
      ),
      h("div", { className: "hw-grid hw-grid-4" },
        (data.blueprints || []).map(function (bp) {
          return h("div", { className: "hw-blueprint", key: bp.key },
            h("div", null,
              h("h3", null, bp.label),
              h("p", { className: "hw-muted hw-small" }, bp.description),
              h("div", { className: "hw-pills" },
                h(Pill, null, "lead: " + bp.profile),
                h(Pill, null, (bp.tasks || []).length + " tasks")
              )
            ),
            h(Button, { disabled: busy === bp.key, onClick: function () { launch(bp.key); } },
              busy === bp.key ? "Creation..." : "Creer pack"
            )
          );
        })
      )
    );
  }

  function QuickTaskPanel(props) {
    const useState = hooks.useState;
    const data = props.data;
    const profiles = data.active_profiles || [];
    const formState = useState({
      title: "",
      assignee: "pilot",
      priority: 7,
      workspace: "workspace_v2",
      triage: true,
      body: "",
      skills: "",
    });
    const form = formState[0];
    const setForm = formState[1];
    const msgState = useState("");
    const msg = msgState[0];
    const setMsg = msgState[1];

    function setField(key, value) {
      setForm(function (prev) {
        const next = {};
        Object.keys(prev).forEach(function (k) { next[k] = prev[k]; });
        next[key] = value;
        return next;
      });
    }

    function submit() {
      if (!form.title.trim()) {
        setMsg("Titre requis.");
        return;
      }
      setMsg("Creation...");
      SDK.fetchJSON(WORKSPACE_API + "/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: form.title,
          body: form.body,
          assignee: form.assignee,
          priority: number(form.priority),
          workspace: form.workspace,
          triage: Boolean(form.triage),
          skills: form.skills.split(",").map(function (s) { return s.trim(); }).filter(Boolean),
        }),
      }).then(function (result) {
        setMsg("Tache creee: " + result.task_id);
        setField("title", "");
        setField("body", "");
        props.refresh();
      }).catch(function (err) {
        setMsg("Erreur: " + String(err && err.message ? err.message : err));
      });
    }

    return h("section", { className: "hw-section" },
      h("div", { className: "hw-section-head" },
        h("div", null,
          h("h2", null, "Creation rapide"),
          h("span", { className: "hw-muted hw-small" }, "Ajoute une tache au board actif sans quitter le cockpit.")
        ),
        msg ? h("span", { className: "hw-muted hw-small" }, msg) : null
      ),
      h("div", { className: "hw-form" },
        h("div", { className: "hw-field" },
          h("label", null, "Titre"),
          h("input", {
            value: form.title,
            onChange: function (event) { setField("title", event.target.value); },
            placeholder: "Ex: Ajouter vue memoire Workspace",
          })
        ),
        h("div", { className: "hw-field" },
          h("label", null, "Assignee"),
          h("select", {
            value: form.assignee,
            onChange: function (event) { setField("assignee", event.target.value); },
          },
            profiles.map(function (p) {
              return h("option", { key: p.name, value: p.name }, p.name);
            })
          )
        ),
        h("div", { className: "hw-field" },
          h("label", null, "Priorite"),
          h("input", {
            type: "number",
            value: form.priority,
            onChange: function (event) { setField("priority", event.target.value); },
          })
        ),
        h(Button, { onClick: submit }, "Creer")
      ),
      h("div", { className: "hw-grid hw-grid-2", style: { marginTop: "10px" } },
        h("div", { className: "hw-field" },
          h("label", null, "Workspace"),
          h("select", {
            value: form.workspace,
            onChange: function (event) { setField("workspace", event.target.value); },
          },
            Object.keys(data.workspaces || {}).map(function (key) {
              const item = data.workspaces[key];
              return h("option", { key: key, value: key }, item.label || key);
            })
          )
        ),
        h("div", { className: "hw-field" },
          h("label", null, "Skills forces, separes par virgule"),
          h("input", {
            value: form.skills,
            onChange: function (event) { setField("skills", event.target.value); },
            placeholder: "kanban-worker, code-security",
          })
        ),
        h("div", { className: "hw-field" },
          h("label", null, "Body"),
          h("textarea", {
            value: form.body,
            onChange: function (event) { setField("body", event.target.value); },
            placeholder: "Contexte, definition of done, contraintes, liens.",
          })
        ),
        h("div", { className: "hw-field" },
          h("label", null, "Mode"),
          h("label", { className: "hw-pill", style: { justifyContent: "flex-start" } },
            h("input", {
              type: "checkbox",
              checked: Boolean(form.triage),
              onChange: function (event) { setField("triage", event.target.checked); },
            }),
            "Triage avant execution"
          )
        )
      )
    );
  }

  function DebugPanel(props) {
    const data = props.data;
    const tasks = (data.kanban && data.kanban.recent_tasks) || [];
    const runtime = data.runtime || {};
    const workspace = data.workspace_v2 || {};

    return h("section", { className: "hw-section" },
      h("div", { className: "hw-section-head" },
        h("div", null,
          h("h2", null, "Debug / chemins"),
          h("span", { className: "hw-muted hw-small" }, "Raccourcis de verification locale.")
        ),
        h("div", { className: "hw-actions" },
          h(Button, { onClick: function () { nav("/logs"); } }, "Logs"),
          h(Button, { onClick: function () { nav("/sessions"); } }, "Sessions")
        )
      ),
      h("div", { className: "hw-grid hw-grid-3" },
        h("div", null,
          h("h3", null, "Runtime"),
          h("div", { className: "hw-code hw-muted" }, runtime.path || ""),
          h("div", { className: "hw-code" }, runtime.status || "")
        ),
        h("div", null,
          h("h3", null, "Workspace V2"),
          h("div", { className: "hw-code hw-muted" }, workspace.path || ""),
          h("div", { className: "hw-code" }, workspace.status || "")
        ),
        h("div", null,
          h("h3", null, "Taches recentes"),
          tasks.length === 0 ? h("span", { className: "hw-muted hw-small" }, "Aucune tache sur le board actif.") :
            tasks.map(function (task) {
              return h("div", { className: "hw-row", key: task.id },
                h("div", { className: "hw-row-main" },
                  h("strong", null, task.title),
                  h("span", { className: "hw-muted hw-small" }, task.id + " / " + task.assignee)
                ),
                h(Pill, null, task.status)
              );
            })
        )
      )
    );
  }

  function WorkspacePage() {
    const state = useSummary();
    const summary = state.summary;

    if (summary.loading && !summary.data) {
      return h("div", { className: "hw-root" }, h("div", { className: "hw-section" }, "Chargement Workspace..."));
    }
    if (summary.error && !summary.data) {
      return h("div", { className: "hw-root" }, h("div", { className: "hw-section" }, "Erreur Workspace: " + summary.error));
    }

    const data = summary.data;
    return h("div", { className: "hw-root" },
      h(WorkspaceHero, { data: data, refresh: state.refresh }),
      h(QuickTaskPanel, { data: data, refresh: state.refresh }),
      h(WorkflowLaunchers, { data: data, refresh: state.refresh }),
      h(AgentMatrix, { data: data }),
      h(PluginsPanel, { data: data }),
      h(DebugPanel, { data: data })
    );
  }

  function ChatTopSlot() {
    const useState = hooks.useState;
    const msgState = useState("");
    const msg = msgState[0];
    const setMsg = msgState[1];
    return h("div", { className: "hw-chat-top" },
      h("div", null,
        h("strong", null, "Workspace prompts"),
        h("span", { className: "hw-muted hw-small", style: { display: "block" } },
          "Copie un prompt puis colle-le dans le TUI."
        )
      ),
      h("div", { className: "hw-actions" },
        PROMPTS.map(function (prompt) {
          return h(Button, { key: prompt.label, onClick: function () { copy(prompt.text, setMsg); } }, prompt.label);
        }),
        h(Button, { onClick: function () { nav("/workspace"); } }, "Cockpit")
      ),
      msg ? h("span", { className: "hw-muted hw-small" }, msg) : null
    );
  }

  function HeaderBannerSlot() {
    return h("div", { className: "hw-banner" },
      h("span", { className: "hw-small" }, "Hermes Workspace actif: agents, plugins, Kanban et debug centralises."),
      h("div", { className: "hw-actions" },
        h("button", { className: "hw-pill", onClick: function () { nav("/workspace"); } }, "Workspace"),
        h("button", { className: "hw-pill", onClick: function () { nav("/kanban"); } }, "Kanban")
      )
    );
  }

  window.__HERMES_PLUGINS__.register("hermes-workspace", WorkspacePage);
  window.__HERMES_PLUGINS__.registerSlot("hermes-workspace", "chat:top", ChatTopSlot);
  window.__HERMES_PLUGINS__.registerSlot("hermes-workspace", "header-banner", HeaderBannerSlot);
})();

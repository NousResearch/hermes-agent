(function () {
  "use strict";
  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK) return;
  const React = SDK.React;
  const h = React.createElement;
  const hooks = SDK.hooks || React;
  const comps = SDK.components || {};
  const Card = comps.Card || "div";
  const CardContent = comps.CardContent || "div";
  const Button = comps.Button || "button";
  const Badge = comps.Badge || "span";
  const useState = hooks.useState;
  const useEffect = hooks.useEffect;

  const API = "/api/plugins/kanban";
  const MISSION_CONTROL_V2_SHELL = "Mission Control v2 /kanban-mission-v2 Inbox decisioni Workbench operativo Document registry Team selector System Diagnostics collapsed";

  function authHeaders() {
    const token = window.__HERMES_SESSION_TOKEN__ || "";
    return token ? { Authorization: "Bearer " + token, "X-Hermes-Session-Token": token } : {};
  }

  async function fetchJSON(path) {
    const res = await fetch(API + path, { headers: authHeaders(), credentials: "include" });
    if (!res.ok) throw new Error(res.status + ": " + (await res.text()));
    return res.json();
  }

  async function postJSON(path, body) {
    const headers = Object.assign({ "Content-Type": "application/json" }, authHeaders());
    const res = await fetch(API + path, { method: "POST", headers: headers, credentials: "include", body: JSON.stringify(body || {}) });
    if (!res.ok) throw new Error(res.status + ": " + (await res.text()));
    return res.json();
  }

  async function patchJSON(path, body) {
    const headers = Object.assign({ "Content-Type": "application/json" }, authHeaders());
    const res = await fetch(API + path, { method: "PATCH", headers: headers, credentials: "include", body: JSON.stringify(body || {}) });
    if (!res.ok) throw new Error(res.status + ": " + (await res.text()));
    return res.json();
  }

  function boardFromUrl() {
    try { return new URL(window.location.href).searchParams.get("board") || localStorage.getItem("hermes.kanban.selectedBoard") || ""; }
    catch (_e) { return ""; }
  }

  function sectionCard(title, subtitle, children) {
    return h(Card, { className: "rounded-xl border bg-card" }, h(CardContent, { className: "space-y-3 p-4" },
      h("div", null,
        h("h2", { className: "text-base font-semibold" }, title),
        subtitle ? h("p", { className: "text-sm text-muted-foreground" }, subtitle) : null
      ),
      children
    ));
  }

  function taskButton(task, onSelect) {
    return h("button", {
      key: task.id,
      className: "w-full rounded-lg border bg-background p-3 text-left hover:border-primary",
      onClick: function () { onSelect(task.id); }
    },
      h("div", { className: "flex items-center justify-between gap-2" },
        h("span", { className: "font-medium" }, task.title || task.id),
        h(Badge, { variant: "outline" }, task.status || "task")
      ),
      h("div", { className: "mt-1 text-xs text-muted-foreground" }, task.id + (task.assignee ? " · " + task.assignee : ""))
    );
  }

  function Workbench(props) {
    const wb = props.workbench || {};
    const columns = [
      ["setup", "Da impostare"],
      ["ready", "Pronte per partire"],
      ["running", "In lavorazione"],
      ["review", "Da review"],
      ["recent_done", "Completate recenti"]
    ];
    return sectionCard("Workbench operativo", "Pipeline operativa: apri una task per agire dal drawer contestuale.",
      h("div", { className: "grid gap-3 lg:grid-cols-5" }, columns.map(function (pair) {
        const key = pair[0];
        const label = pair[1];
        const tasks = (wb[key] && wb[key].tasks) || [];
        return h("div", { key: key, className: "rounded-lg border bg-background p-3" },
          h("div", { className: "mb-2 flex items-center justify-between gap-2" },
            h("h3", { className: "text-sm font-semibold" }, label),
            h(Badge, { variant: "outline" }, String(tasks.length))
          ),
          tasks.length ? h("div", { className: "space-y-2" }, tasks.map(function (t) { return taskButton(t, props.onSelect); })) :
            h("p", { className: "text-xs text-muted-foreground" }, "Vuota")
        );
      }))
    );
  }

  function TaskDrawer(props) {
    const detail = props.detail;
    const board = props.board;
    const [title, setTitle] = useState("");
    const [body, setBody] = useState("");
    const [contextNote, setContextNote] = useState("");
    const [requestNote, setRequestNote] = useState("");
    const [status, setStatus] = useState("");
    if (!detail) {
      return sectionCard("Task drawer contestuale", "Seleziona una task per vedere azioni, documenti e contesto.",
        h("p", { className: "text-sm text-muted-foreground" }, "Nessuna task selezionata.")
      );
    }
    const task = detail.task || {};
    const actions = detail.primary_actions || [];
    const activation = detail.review_activation || null;
    const inputDocs = (detail.documents && detail.documents.input_attachments) || [];
    const outputs = (detail.documents && detail.documents.produced_outputs) || [];
    function createFollowup() {
      if (!title.trim() || !body.trim()) { setStatus("Inserisci titolo e istruzioni del follow-up."); return; }
      setStatus("Creo follow-up collegato…");
      const qs = board ? "?board=" + encodeURIComponent(board) : "";
      postJSON("/tasks/" + encodeURIComponent(task.id) + "/actions/create-followup" + qs, {
        author: "daniele",
        title: title,
        body: body,
        assignee: task.assignee || "default",
        confirm: true
      }).then(function (j) {
        setStatus("Follow-up creato in todo: " + j.task_id + ". Dispatch separato: nessun lavoro nascosto avviato.");
        setTitle("");
        setBody("");
        if (props.onRefresh) props.onRefresh();
      }).catch(function (e) { setStatus("Errore follow-up: " + e.message); });
    }
    function addContext() {
      if (!contextNote.trim()) { setStatus("Scrivi il contesto da aggiungere."); return; }
      setStatus("Aggiungo contesto alla task…");
      const qs = board ? "?board=" + encodeURIComponent(board) : "";
      postJSON("/tasks/" + encodeURIComponent(task.id) + "/comments" + qs, {
        author: "daniele",
        body: "MISSION_CONTROL_CONTEXT\n" + contextNote + "\n\nGuardrail: commento/contesto soltanto; no dispatch."
      }).then(function () {
        setStatus("Contesto aggiunto. Nessun dispatch avviato.");
        setContextNote("");
        if (props.onRefresh) props.onRefresh();
      }).catch(function (e) { setStatus("Errore contesto: " + e.message); });
    }
    function requestChanges() {
      const note = requestNote.trim() || body.trim();
      if (!note) { setStatus("Scrivi quali modifiche chiedere."); return; }
      setStatus("Preparo follow-up modifiche…");
      const qs = board ? "?board=" + encodeURIComponent(board) : "";
      postJSON("/tasks/" + encodeURIComponent(task.id) + "/actions/create-followup" + qs, {
        author: "daniele",
        title: "Modifiche richieste — " + (task.title || task.id),
        body: note,
        assignee: task.assignee || "default",
        confirm: true
      }).then(function (j) {
        setStatus("Richiesta modifiche preparata in todo: " + j.task_id + ". Dispatch separato.");
        setRequestNote("");
        if (props.onRefresh) props.onRefresh();
      }).catch(function (e) { setStatus("Errore richiesta modifiche: " + e.message); });
    }
    function decideReview(nextStatus, label) {
      if (nextStatus === "ready") {
        const ok = window.confirm("Preview sblocco / prepara dispatch? La task diventa ready, ma il dispatch resta separato e richiede DISPATCH_ONE_TICK.");
        if (!ok) {
          setStatus("Sblocco annullato. Nessun dispatch avviato.");
          return;
        }
      }
      const qs = board ? "?board=" + encodeURIComponent(board) : "";
      setStatus(label + "…");
      const payload = nextStatus === "done" ? {
        status: "done",
        summary: "Accepted from Mission Control v2 review activation.",
        result: "Accepted by Daniele from Mission Control v2. No dispatch and no external send were started.",
        metadata: { accepted_from: "mission-control-v2", dispatch_started: false, external_send_started: false }
      } : { status: nextStatus };
      patchJSON("/tasks/" + encodeURIComponent(task.id) + qs, payload).then(function () {
        setStatus(label + " completato. Dispatch resta separato e confermato.");
        if (props.onRefresh) props.onRefresh();
      }).catch(function (e) { setStatus("Errore decisione: " + e.message); });
    }
    return sectionCard("Task drawer contestuale", "Azioni primarie cambiano per stato task; dispatch sempre separato e confermato.",
      h("div", { className: "space-y-4" },
        h("div", { className: "rounded-lg border bg-background p-3" },
          h("div", { className: "flex flex-wrap items-center justify-between gap-2" },
            h("h3", { className: "text-lg font-semibold" }, task.title || task.id),
            h(Badge, { variant: "outline" }, (detail.state || task.status || "state"))
          ),
          h("p", { className: "mt-1 text-xs text-muted-foreground" }, task.id + " · " + (task.assignee || "unassigned")),
          task.latest_summary ? h("p", { className: "mt-2 text-sm" }, task.latest_summary) : null
        ),
        activation ? h("div", { className: "rounded-lg border border-amber-500/40 bg-amber-500/10 p-3" },
          h("div", { className: "flex flex-wrap items-center justify-between gap-2" },
            h("h4", { className: "text-sm font-semibold" }, activation.state || "Decisione richiesta"),
            h(Badge, { variant: "outline" }, "Serve Daniele")
          ),
          h("p", { className: "mt-2 text-sm" }, activation.why_it_matters),
          h("p", { className: "mt-1 text-xs text-muted-foreground" }, activation.recommended_next_action),
          h("div", { className: "mt-3 flex flex-wrap gap-2" },
            h(Button, { onClick: function () { decideReview("done", "Accetta e chiudi"); } }, "Accetta e chiudi"),
            h(Button, { onClick: function () { decideReview("ready", "Preview sblocco / prepara dispatch"); } }, "Preview sblocco / prepara dispatch"),
            h(Button, { onClick: function () { setStatus("Compila la richiesta modifiche qui sotto, poi premi Prepara richiesta modifiche."); } }, "Chiedi modifiche"),
            h(Button, { onClick: function () { setStatus("Parcheggiata: la task resta bloccata finché non scegli un'azione."); } }, "Parcheggia")
          ),
          h("p", { className: "mt-2 text-xs text-muted-foreground" }, activation.guardrail_copy || "Non dispatcha e non invia nulla senza conferma.")
        ) : null,
        h("div", { className: "grid gap-3 md:grid-cols-2" },
          h("div", { className: "rounded-lg border bg-background p-3" },
            h("h4", { className: "text-sm font-semibold" }, "Azioni contestuali"),
            h("div", { className: "mt-2 flex flex-wrap gap-2" }, actions.map(function (a) {
              return h(Badge, { key: a.id, variant: "outline" }, (a.label || a.id) + (a.requires_confirmation ? " · conferma" : ""));
            })),
            h("p", { className: "mt-2 text-xs text-muted-foreground" }, "No auto-dispatch · no external send · no hidden work")
          ),
          h("div", { className: "rounded-lg border bg-background p-3" },
            h("h4", { className: "text-sm font-semibold" }, "Documenti task"),
            h("p", { className: "mt-2 text-xs text-muted-foreground" }, "Allegati della task: " + inputDocs.length + " · Documenti prodotti: " + outputs.length),
            outputs.length ? h("ul", { className: "mt-2 space-y-1 text-xs" }, outputs.map(function (o) { return h("li", { key: o.entry_name || o.filename }, o.filename); })) : null,
            inputDocs.length ? h("ul", { className: "mt-2 space-y-1 text-xs text-muted-foreground" }, inputDocs.map(function (o) { return h("li", { key: o.id || o.filename }, o.filename); })) : null
          )
        ),
        h("div", { className: "rounded-lg border bg-background p-3" },
          h("h4", { className: "text-sm font-semibold" }, "Aggiungi contesto"),
          h("p", { className: "text-xs text-muted-foreground" }, "Aggiunge una nota alla task. Non sblocca, non dispatcha, non invia fuori."),
          h("textarea", { className: "mt-2 w-full rounded-md border bg-background px-3 py-2 text-sm", rows: 3, placeholder: "Contesto da aggiungere alla task", value: contextNote, onChange: function (e) { setContextNote(e.target.value); } }),
          h("div", { className: "mt-2 flex flex-wrap items-center gap-2" },
            h(Button, { onClick: addContext }, "Aggiungi contesto")
          )
        ),
        h("div", { className: "rounded-lg border bg-background p-3" },
          h("h4", { className: "text-sm font-semibold" }, "Chiedi modifiche"),
          h("p", { className: "text-xs text-muted-foreground" }, "Prepara una task figlia in todo con le modifiche richieste. Non dispatcha."),
          h("textarea", { className: "mt-2 w-full rounded-md border bg-background px-3 py-2 text-sm", rows: 3, placeholder: "Modifiche richieste", value: requestNote, onChange: function (e) { setRequestNote(e.target.value); } }),
          h("div", { className: "mt-2 flex flex-wrap items-center gap-2" },
            h(Button, { onClick: requestChanges }, "Prepara richiesta modifiche")
          )
        ),
        h("div", { className: "rounded-lg border bg-background p-3" },
          h("h4", { className: "text-sm font-semibold" }, "Crea follow-up"),
          h("p", { className: "text-xs text-muted-foreground" }, "Crea una task figlia collegata in todo. Non dispatcha e non invia fuori."),
          h("input", { className: "mt-2 w-full rounded-md border bg-background px-3 py-2 text-sm", placeholder: "Titolo follow-up", value: title, onChange: function (e) { setTitle(e.target.value); } }),
          h("textarea", { className: "mt-2 w-full rounded-md border bg-background px-3 py-2 text-sm", rows: 3, placeholder: "Istruzioni / contesto follow-up", value: body, onChange: function (e) { setBody(e.target.value); } }),
          h("div", { className: "mt-2 flex flex-wrap items-center gap-2" },
            h(Button, { onClick: createFollowup }, "Crea follow-up"),
            status ? h("span", { className: "text-xs text-muted-foreground" }, status) : null
          )
        )
      )
    );
  }

  function MissionControlV2Page() {
    const [board, setBoard] = useState(boardFromUrl());
    const [data, setData] = useState(null);
    const [error, setError] = useState("");
    const [selected, setSelected] = useState("");
    const [detail, setDetail] = useState(null);
    const [loading, setLoading] = useState(false);
    const [activeTab, setActiveTab] = useState("inbox");
    const [documentActionStatus, setDocumentActionStatus] = useState("");
    const [teamActionStatus, setTeamActionStatus] = useState("");
    const [dispatchPreview, setDispatchPreview] = useState(null);
    const [dispatchStatus, setDispatchStatus] = useState("");

    function load() {
      setLoading(true);
      setError("");
      const qs = board ? "?board=" + encodeURIComponent(board) : "";
      fetchJSON("/v2/cockpit" + qs).then(function (j) {
        setData(j);
        setLoading(false);
      }).catch(function (e) { setError(e.message); setLoading(false); });
    }
    function openTask(id) {
      setSelected(id);
      setActiveTab("workbench");
      const qs = board ? "?board=" + encodeURIComponent(board) : "";
      fetchJSON("/v2/tasks/" + encodeURIComponent(id) + qs).then(setDetail).catch(function (e) { setError(e.message); });
    }
    useEffect(load, [board]);
    const inbox = data && data.inbox ? data.inbox : { items: [] };
    const docs = data && data.documents ? data.documents : { items: [] };
    const teams = data && data.teams ? data.teams : { presets: [] };
    const system = data && data.system ? data.system : { counts: {} };
    const tabs = [
      ["inbox", "Inbox"],
      ["workbench", "Workbench"],
      ["documents", "Documents"],
      ["teams", "Teams"],
      ["system", "System"]
    ];
    function documentUrl(d) {
      const url = d.download_url || ("/api/plugins/kanban/v2/documents/" + encodeURIComponent(d.entry_name || ""));
      const qs = board && url.indexOf("?") === -1 ? "?board=" + encodeURIComponent(board) : "";
      return url + qs;
    }
    function downloadDocument(d) {
      const url = documentUrl(d);
      setError("");
      fetch(url, { headers: authHeaders(), credentials: "include" }).then(async function (res) {
        if (!res.ok) throw new Error(res.status + ": " + (await res.text()));
        const blob = await res.blob();
        const objectUrl = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = objectUrl;
        a.download = d.filename || d.entry_name || "document";
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(function () { URL.revokeObjectURL(objectUrl); }, 30000);
      }).catch(function (e) { setError("Download documento non riuscito: " + e.message); });
    }
    function prepareDocumentAction(d, template) {
      if (!template || !template.id) return;
      const ok = window.confirm("Preparare una task non-dispatchable in triage per: " + (template.label || template.id) + "? Nessun dispatch e nessun invio esterno verranno avviati.");
      if (!ok) return;
      setDocumentActionStatus("Preparo task…");
      const qs = board ? "?board=" + encodeURIComponent(board) : "";
      postJSON("/v2/document-actions/prepare-followup" + qs, {
        filename: d.filename,
        action_id: template.id,
        author: "daniele",
        assignee: "default",
        confirm: true
      }).then(function (j) {
        setDocumentActionStatus("Task preparata in triage: " + j.task_id + ". Nessun dispatch avviato.");
        load();
      }).catch(function (e) { setDocumentActionStatus("Errore preparazione task: " + e.message); });
    }
    function prepareTeamAction(preset) {
      if (!preset || !preset.id) return;
      const ok = window.confirm("Preparare il flow team: " + (preset.label || preset.id) + "? Verrà creata una task non-dispatchable in triage, senza dispatch.");
      if (!ok) return;
      setTeamActionStatus("Preparo flow team…");
      const qs = board ? "?board=" + encodeURIComponent(board) : "";
      postJSON("/v2/team-actions/prepare" + qs, {
        preset_id: preset.id,
        author: "daniele",
        assignee: "default",
        confirm: true
      }).then(function (j) {
        setTeamActionStatus("Flow team preparato in triage: " + j.task_id + ". Nessun dispatch avviato.");
        load();
      }).catch(function (e) { setTeamActionStatus("Errore flow team: " + e.message); });
    }
    function previewDispatch() {
      setDispatchStatus("Calcolo preview…");
      const qs = board ? "?board=" + encodeURIComponent(board) : "";
      fetchJSON("/v2/dispatch/preview" + qs).then(function (j) {
        setDispatchPreview(j);
        setDispatchStatus("Preview pronta: " + ((j.would_spawn || []).length) + " task partirebbe ora.");
      }).catch(function (e) { setDispatchStatus("Errore preview dispatch: " + e.message); });
    }
    function confirmDispatchOneTick() {
      const phrase = window.prompt("Per avviare un solo tick controllato scrivi DISPATCH_ONE_TICK. Max 1 worker, no external send, no auto-decompose.");
      if (phrase !== "DISPATCH_ONE_TICK") { setDispatchStatus("Dispatch annullato: frase non confermata."); return; }
      setDispatchStatus("Avvio un solo dispatch tick…");
      const qs = board ? "?board=" + encodeURIComponent(board) : "";
      postJSON("/v2/dispatch/confirm-one-tick" + qs, { confirmation: phrase, author: "daniele" }).then(function (j) {
        setDispatchPreview(j.before_preview || null);
        const spawned = (j.dispatch_result && j.dispatch_result.spawned) || [];
        setDispatchStatus("Dispatch tick eseguito. Worker avviati: " + spawned.length + ". Osserva subito Workbench/System.");
        load();
      }).catch(function (e) { setDispatchStatus("Errore dispatch: " + e.message); });
    }
    function DispatchControl() {
      const preview = dispatchPreview || {};
      const wouldSpawn = preview.would_spawn || [];
      const skipped = preview.skipped || [];
      return sectionCard("Dispatch controllato", "Preview → conferma esplicita → un solo tick → osserva. Default sicuro: max 1 worker.",
        h("div", { className: "space-y-3" },
          h("div", { className: "rounded-lg border border-amber-500/40 bg-amber-500/10 p-3" },
            h("div", { className: "text-sm font-semibold" }, "Guardrail"),
            h("p", { className: "mt-1 text-sm text-muted-foreground" }, "Non è automatico: prima fai Preview dispatch, poi confermi scrivendo DISPATCH_ONE_TICK. Limiti server: max_spawn=1, max_in_progress=2, max_in_progress_per_profile=1.")
          ),
          h("div", { className: "flex flex-wrap gap-2" },
            h(Button, { onClick: previewDispatch }, "Preview dispatch"),
            h(Button, { onClick: confirmDispatchOneTick }, "Conferma one-tick")
          ),
          dispatchStatus ? h("p", { className: "text-xs text-muted-foreground" }, dispatchStatus) : null,
          dispatchPreview ? h("div", { className: "grid gap-3 md:grid-cols-2" },
            h("div", { className: "rounded-lg border bg-background p-3" },
              h("div", { className: "text-sm font-semibold" }, "Partirebbe ora"),
              wouldSpawn.length ? h("ul", { className: "mt-2 space-y-1 text-xs" }, wouldSpawn.map(function (t) { return h("li", { key: t.task_id }, t.task_id + " · " + t.assignee + " · " + (t.title || "task")); })) : h("p", { className: "mt-2 text-xs text-muted-foreground" }, "Nessuna task dispatchabile ora.")
            ),
            h("div", { className: "rounded-lg border bg-background p-3" },
              h("div", { className: "text-sm font-semibold" }, "Saltate / cap"),
              skipped.length ? h("ul", { className: "mt-2 space-y-1 text-xs text-muted-foreground" }, skipped.slice(0, 6).map(function (t) { return h("li", { key: t.task_id + ':' + t.reason }, t.task_id + " · " + t.reason); })) : h("p", { className: "mt-2 text-xs text-muted-foreground" }, "Nessuna task saltata.")
            )
          ) : null
        )
      );
    }
    return h("div", { className: "space-y-6", "data-smoke": MISSION_CONTROL_V2_SHELL },
      h("div", { className: "flex flex-wrap items-start justify-between gap-3" },
        h("div", null,
          h("h1", { className: "text-2xl font-semibold" }, "Mission Control v2"),
          h("p", { className: "text-sm text-muted-foreground" }, "Cockpit operativo parallelo: Inbox, Workbench, Documents, Teams, System. Legacy /kanban-mission resta intatto.")
        ),
        h("div", { className: "flex flex-wrap items-center gap-2" },
          h("input", { className: "rounded-md border bg-background px-3 py-2 text-sm", placeholder: "board", value: board, onChange: function (e) { setBoard(e.target.value); } }),
          h(Button, { onClick: load, disabled: loading }, loading ? "Loading…" : "Refresh"),
          h(Badge, { variant: "outline" }, "no auto-dispatch")
        )
      ),
      h("div", { className: "flex flex-wrap gap-2 rounded-xl border bg-card p-2" }, tabs.map(function (tab) {
        return h(Button, {
          key: tab[0],
          onClick: function () { setActiveTab(tab[0]); },
          variant: activeTab === tab[0] ? "default" : "outline"
        }, tab[1]);
      })),
      error ? h("div", { className: "rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-500" }, error) : null,
      activeTab === "inbox" ? sectionCard("Inbox decisioni", "Max 3–5 elementi che richiedono Daniele ora.",
        inbox.items && inbox.items.length ? h("div", { className: "grid gap-2 md:grid-cols-2 xl:grid-cols-3" }, inbox.items.slice(0, inbox.max_items || 5).map(function (item) {
          return h("button", { key: item.task_id, className: "rounded-lg border bg-background p-3 text-left hover:border-primary", onClick: function () { openTask(item.task_id); } },
            h("div", { className: "font-medium" }, item.title),
            h("div", { className: "mt-1 text-xs text-muted-foreground" }, item.status + " · " + item.why_it_matters),
            h("div", { className: "mt-2 text-xs" }, item.recommended_action || "open_task_drawer")
          );
        })) : h("p", { className: "text-sm text-muted-foreground" }, inbox.empty_state || "Nessuna decisione urgente.")
      ) : null,
      activeTab === "workbench" ? h("div", { className: "space-y-6" },
        h(Workbench, { workbench: data && data.workbench, onSelect: openTask }),
        h(DispatchControl),
        h(TaskDrawer, { detail: detail, selected: selected, board: board, onRefresh: load })
      ) : null,
      activeTab === "documents" ? sectionCard("Document registry", "Output, Vault/Drive links, versioni e review come cittadini di prima classe.",
        h("div", { className: "space-y-3" },
          h("div", { className: "rounded-lg border border-primary/30 bg-primary/5 p-3" },
            h("div", { className: "text-sm font-semibold" }, "Cosa fare ora"),
            h("p", { className: "mt-1 text-sm text-muted-foreground" }, "Apri solo i documenti CURRENT_REVIEW principali. Per Kania: PDD, Evidence Register e Check finale. Non inviare fuori: questa vista serve per review interna."),
            h("div", { className: "mt-2 flex flex-wrap gap-2" },
              h(Badge, { variant: "outline" }, "review interna"),
              h(Badge, { variant: "outline" }, "no external send"),
              h(Badge, { variant: "outline" }, "no dispatch")
            ),
            documentActionStatus ? h("p", { className: "mt-2 text-xs text-muted-foreground" }, documentActionStatus) : null
          ),
          docs.items && docs.items.length ? h("div", { className: "grid gap-2 md:grid-cols-2" }, docs.items.map(function (d) {
            return h("div", { key: (d.source || "doc") + ":" + (d.task_id || d.pack || "vault") + ":" + d.entry_name, className: "rounded-lg border bg-background p-3" },
              h("div", { className: "flex items-start justify-between gap-2" },
                h("div", { className: "font-medium" }, d.filename),
                h(Badge, { variant: "outline" }, d.review_status || "review pending")
              ),
              h("div", { className: "mt-1 text-xs text-muted-foreground" },
                (d.task_id ? "task " + d.task_id : (d.project || "Vault")) +
                " · " + (d.document_type || d.source || "document") +
                (d.code ? " · " + d.code : "")
              ),
              d.pack ? h("div", { className: "mt-1 text-xs text-muted-foreground" }, "pack " + d.pack + (d.sha256_short ? " · sha256 " + d.sha256_short : "")) : null,
              h("div", { className: "mt-3 flex flex-wrap gap-2" },
                h(Button, { onClick: function () { downloadDocument(d); } }, "Scarica"),
                d.task_id ? h(Button, { onClick: function () { openTask(d.task_id); setActiveTab("workbench"); } }, "Apri task") : null,
                h(Badge, { variant: "outline" }, d.source || "document")
              ),
              d.action_templates && d.action_templates.length ? h("div", { className: "mt-3 border-t pt-3" },
                h("div", { className: "mb-2 text-xs font-semibold text-muted-foreground" }, "Azioni guidate"),
                h("div", { className: "flex flex-wrap gap-2" }, d.action_templates.map(function (template) {
                  return h(Button, { key: template.id, onClick: function () { prepareDocumentAction(d, template); } }, template.label || template.id);
                }))
              ) : null
            );
          })) : h("p", { className: "text-sm text-muted-foreground" }, "Nessun documento prodotto indicizzato.")
        )
      ) : null,
      activeTab === "teams" ? sectionCard("Team selector", "Scelta team semplificata; dispatch resta separato.",
        h("div", { className: "space-y-3" },
          h("div", { className: "rounded-lg border border-primary/30 bg-primary/5 p-3" },
            h("div", { className: "text-sm font-semibold" }, "Cosa fa questo tab"),
            h("p", { className: "mt-1 text-sm text-muted-foreground" }, "Prepara una task di coordinamento team in todo. Non parte nessun agente: il dispatch resta un passo separato e confermato."),
            teamActionStatus ? h("p", { className: "mt-2 text-xs text-muted-foreground" }, teamActionStatus) : null
          ),
          h("div", { className: "grid gap-2 md:grid-cols-2 xl:grid-cols-5" }, (teams.presets || []).map(function (p) {
            return h("div", { key: p.id, className: "rounded-lg border bg-background p-3" },
              h("div", { className: "font-medium" }, p.label),
              h("div", { className: "mt-1 text-xs text-muted-foreground" }, p.description || "Preset team"),
              h("div", { className: "mt-3 flex flex-wrap gap-2" },
                h(Button, { onClick: function () { prepareTeamAction(p); } }, "Prepara flow"),
                h(Badge, { variant: "outline" }, "no dispatch")
              )
            );
          }))
        )
      ) : null,
      activeTab === "system" ? h("details", { className: "rounded-xl border bg-card p-4", open: true },
        h("summary", { className: "cursor-pointer font-semibold" }, "System / Diagnostics collassato"),
        h("pre", { className: "mt-3 overflow-auto rounded-lg bg-background p-3 text-xs" }, JSON.stringify(system.counts || {}, null, 2))
      ) : null
    );
  }

  if (window.__HERMES_PLUGINS__ && typeof window.__HERMES_PLUGINS__.register === "function") {
    window.__HERMES_PLUGINS__.register("kanban-dashboard-v2", MissionControlV2Page);
  }
})();

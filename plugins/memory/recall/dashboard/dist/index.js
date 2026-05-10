(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const React = SDK.React;
  const h = React.createElement;
  const C = SDK.components;
  const Card = C.Card, CardHeader = C.CardHeader, CardTitle = C.CardTitle, CardContent = C.CardContent;
  const Badge = C.Badge;
  const useEffect = SDK.hooks.useEffect, useState = SDK.hooks.useState;

  function Button(props) {
    const className = "inline-flex cursor-pointer items-center justify-center border border-border bg-background px-3 py-2 text-xs font-bold uppercase tracking-widest transition hover:bg-foreground/10 disabled:cursor-not-allowed disabled:opacity-50 " + (props.className || "");
    return h("button", { type: "button", onClick: props.onClick, disabled: props.disabled, className: className, title: props.title }, props.children);
  }

  function pill(text) { return h(Badge, { variant: "outline", className: "font-courier text-xs", key: text }, text); }
  function fmt(n) { return typeof n === "number" ? n.toLocaleString() : String(n || 0); }
  function score(n) { return typeof n === "number" ? Math.round(n * 100) + "%" : "—"; }
  function short(id) { return String(id || "").slice(0, 12); }
  function labelStatus(status) { return status === "all" ? "all rows" : status; }

  function StatButton(props) {
    const active = props.active ? "bg-foreground/10 ring-1 ring-foreground/30" : "";
    return h("button", {
      type: "button",
      onClick: props.onClick,
      title: "Click to view " + props.label + " rows",
      className: "group cursor-pointer border border-border p-3 text-left transition hover:bg-foreground/10 hover:ring-1 hover:ring-foreground/30 " + active,
    },
      h("div", { className: "flex items-center justify-between gap-2" },
        h("div", { className: "text-xs uppercase text-muted-foreground" }, props.label),
        h("div", { className: "text-[10px] uppercase tracking-widest text-muted-foreground opacity-70 group-hover:opacity-100" }, "View")),
      h("div", { className: "font-courier text-2xl" }, fmt(props.value)));
  }

  function HowToUse(props) {
    const candidateCount = props.candidateCount || 0;
    return h(Card, { className: "border-dashed" }, h(CardContent, { className: "py-4" },
      h("div", { className: "grid gap-4 lg:grid-cols-[1.2fr_1fr]" },
        h("div", null,
          h("h3", { className: "mb-2 text-base font-bold uppercase tracking-wide" }, "How to use this page"),
          h("ol", { className: "ml-5 list-decimal space-y-1 text-sm text-muted-foreground" },
            h("li", null, h("strong", { className: "text-foreground" }, "Click a count tile"), " — Active, Candidate, Rejected, or All Rows changes the queue below."),
            h("li", null, h("strong", { className: "text-foreground" }, "Open Details"), " to inspect the raw archived row."),
            h("li", null, h("strong", { className: "text-foreground" }, "Review / promote"), " to edit a compact durable fact before writing it to built-in memory."),
            h("li", null, h("strong", { className: "text-foreground" }, "Accept / Reject"), " only changes the Recall archive row status; it does not write built-in memory."))),
        h("div", { className: "border border-border bg-background/30 p-3 text-sm" },
          h("div", { className: "mb-2 font-bold uppercase" }, candidateCount ? "Pending review" : "No candidate inbox right now"),
          h("p", { className: "mb-3 text-muted-foreground" }, candidateCount ? "Candidate rows are the normal approval queue. Start there." : "This is why the previous screen looked read-only: there are currently zero candidate rows. Use Active or All Rows to browse existing memory."),
          h("div", { className: "flex flex-wrap gap-2" },
            h(Button, { onClick: function () { props.setStatus(candidateCount ? "candidate" : "active"); } }, candidateCount ? "Open candidates" : "Open active rows"),
            h(Button, { onClick: function () { props.setStatus("all"); } }, "Show all rows"))))));
  }

  function EmptyState(props) {
    return h("div", { className: "border border-dashed border-border bg-background/30 p-4" },
      h("p", { className: "mb-2 font-bold uppercase" }, "No rows visible for " + labelStatus(props.status) + "."),
      h("p", { className: "mb-3 text-sm text-muted-foreground" }, props.status === "candidate"
        ? "The approval inbox is empty. This is a healthy state, not a read-only mode. Browse Active or All Rows to inspect existing Recall memory."
        : "Your current status/search/filter combination returned nothing. Try another status tile or clear filters."),
      h("div", { className: "flex flex-wrap gap-2" },
        props.counts.active ? h(Button, { onClick: function () { props.setStatus("active"); } }, "View active rows") : null,
        props.counts.rejected ? h(Button, { onClick: function () { props.setStatus("rejected"); } }, "View rejected rows") : null,
        h(Button, { onClick: function () { props.setStatus("all"); } }, "Show all rows"),
        h(Button, { onClick: props.clearFilters }, "Clear filters")));
  }

  function RowCard(props) {
    const row = props.row;
    return h("div", { className: "border border-border bg-background/30 p-3" },
      h("div", { className: "mb-2 flex flex-wrap items-center gap-2" },
        pill(row.status || "unknown"), pill(row.type || "fact"), pill("quality " + score(row.quality_score)),
        h("span", { className: "font-courier text-xs text-muted-foreground" }, short(row.id))),
      h("p", { className: "mb-3 whitespace-pre-wrap text-sm" }, row.content || ""),
      row.quality_reasons && row.quality_reasons.length ? h("p", { className: "mb-3 text-xs text-muted-foreground" }, row.quality_reasons.join(" · ")) : null,
      h("div", { className: "flex flex-wrap gap-2" },
        h(Button, { onClick: function () { props.onDetail(row); } }, "Open details"),
        h(Button, { onClick: function () { props.onSelect(row); } }, "Review / promote"),
        h(Button, { onClick: function () { props.onMark(row, "active"); } }, "Accept archive row"),
        h(Button, { onClick: function () { props.onMark(row, "rejected"); } }, "Reject archive row")));
  }

  function RecallPage() {
    const [overview, setOverview] = useState(null), [rows, setRows] = useState([]), [consolidations, setConsolidations] = useState([]), [audit, setAudit] = useState([]);
    const [status, setStatus] = useState("active"), [query, setQuery] = useState(""), [selected, setSelected] = useState(null), [detail, setDetail] = useState(null);
    const [factOnly, setFactOnly] = useState(false), [hideEpisodes, setHideEpisodes] = useState(true), [minQuality, setMinQuality] = useState("0");
    const [allowRejected, setAllowRejected] = useState(false);
    const [target, setTarget] = useState("memory"), [edited, setEdited] = useState(""), [loading, setLoading] = useState(false), [message, setMessage] = useState("");

    function load() {
      setLoading(true);
      const q = query.trim();
      const filters = "&exclude_episode=" + encodeURIComponent(hideEpisodes ? "true" : "false") + "&min_quality_score=" + encodeURIComponent(minQuality || "0") + (factOnly ? "&type=fact" : "");
      Promise.all([
        SDK.fetchJSON("/api/plugins/recall/overview"),
        SDK.fetchJSON("/api/plugins/recall/observations?status=" + encodeURIComponent(status) + "&limit=80" + filters + (q ? "&q=" + encodeURIComponent(q) : "")),
        SDK.fetchJSON("/api/plugins/recall/consolidations?limit=10"),
        SDK.fetchJSON("/api/plugins/recall/audit?limit=12")
      ]).then(function (parts) {
        setOverview(parts[0]); setRows(parts[1].results || []); setConsolidations(parts[2].results || []); setAudit(parts[3].events || []);
      }).catch(function (err) { setMessage("Load failed: " + (err && err.message ? err.message : err)); }).finally(function () { setLoading(false); });
    }
    function clearFilters() {
      setQuery(""); setMinQuality("0"); setFactOnly(false); setHideEpisodes(true);
      setTimeout(load, 0);
    }

    useEffect(load, [status]);
    useEffect(function () { if (selected) setEdited(selected.content || ""); }, [selected && selected.id]);

    const stats = (overview && overview.stats) || {}, counts = stats.observations_by_status || {}, build = (overview && overview.build_info) || {};
    const diagnoseOK = !!(overview && overview.diagnose && overview.diagnose.ok), auditOK = !!(overview && overview.audit && overview.audit.ok);
    const candidateCount = counts.candidate || 0;

    function mark(row, next) {
      setMessage("");
      SDK.fetchJSON("/api/plugins/recall/observations/" + encodeURIComponent(row.id) + "/mark", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: next, reason: "dashboard curation" }) })
        .then(function () { setMessage("Marked " + next + ": " + short(row.id)); load(); })
        .catch(function (err) { setMessage("Mark failed: " + (err.message || err)); });
    }
    function fetchDetail(row) {
      SDK.fetchJSON("/api/plugins/recall/observations/" + encodeURIComponent(row.id)).then(setDetail).catch(function (err) { setMessage("Detail failed: " + (err.message || err)); });
    }
    function promote() {
      if (!selected) return;
      setMessage("");
      SDK.fetchJSON("/api/plugins/recall/observations/" + encodeURIComponent(selected.id) + "/promote", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target: target, content: edited, confirm: true, allow_rejected: allowRejected, reason: "dashboard reviewed" }) })
        .then(function (res) { setMessage(res.requires_confirm ? "Promotion needs confirmation." : "Promoted to built-in " + target + " memory."); if (!res.requires_confirm) { setSelected(null); load(); } })
        .catch(function (err) { setMessage("Promotion failed: " + (err.message || err)); });
    }
    function applyGroup(group) {
      setMessage("");
      SDK.fetchJSON("/api/plugins/recall/consolidations/apply", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ canonical_id: group.canonical_id, duplicate_ids: group.duplicate_ids || [], confirm: true, reason: "dashboard reviewed" }) })
        .then(function (res) { setMessage("Applied consolidation: rejected " + res.duplicates_rejected + " duplicate row(s)."); load(); })
        .catch(function (err) { setMessage("Consolidation failed: " + (err.message || err)); });
    }

    const header = h(Card, null, h(CardHeader, null, h("div", { className: "flex items-start justify-between gap-4" },
      h("div", null, h(CardTitle, { className: "text-lg" }, "Recall Memory Curation"), h("p", { className: "mt-1 text-sm text-muted-foreground" }, "Browse the lower-trust Recall archive, review candidate rows, promote verified durable facts, and apply audited consolidations.")),
      h("div", { className: "flex flex-wrap items-center justify-end gap-2" }, pill("v" + (build.version || "?")), pill(diagnoseOK ? "diagnose OK" : "diagnose ?"), pill(auditOK ? "audit OK" : "audit ?"), h(Button, { onClick: load, disabled: loading }, loading ? "Refreshing…" : "Refresh")))));

    const statsCard = h(Card, null, h(CardContent, { className: "pt-6" },
      h("div", { className: "mb-4 grid gap-2 lg:grid-cols-[1fr_auto_auto_auto_auto]" },
        h("input", { value: query, onChange: function (e) { setQuery(e.target.value); }, onKeyDown: function (e) { if (e.key === "Enter") load(); }, placeholder: "Search archive rows…", className: "w-full border border-border bg-background px-3 py-2 font-courier text-sm" }),
        h("label", { className: "flex items-center gap-2 border border-border px-3 py-2 text-xs" }, h("input", { type: "checkbox", checked: factOnly, onChange: function (e) { setFactOnly(e.target.checked); } }), "Fact rows"),
        h("label", { className: "flex items-center gap-2 border border-border px-3 py-2 text-xs" }, h("input", { type: "checkbox", checked: hideEpisodes, onChange: function (e) { setHideEpisodes(e.target.checked); } }), "Hide episodes"),
        h("label", { className: "flex items-center gap-2 border border-border px-3 py-2 text-xs" }, "Minimum quality", h("input", { value: minQuality, onChange: function (e) { setMinQuality(e.target.value); }, className: "w-16 bg-background font-courier" })),
        h("div", { className: "flex gap-2" }, h(Button, { onClick: load }, "Search / apply filters"), h(Button, { onClick: clearFilters }, "Clear"))),
      h("p", { className: "mb-3 text-xs text-muted-foreground" }, "The tiles below are navigation buttons, not static counters."),
      h("div", { className: "grid gap-3 md:grid-cols-5" }, ["active", "candidate", "promoted", "rejected"].map(function (k) { return h(StatButton, { key: k, label: k, value: counts[k] || 0, active: status === k, onClick: function () { setStatus(k); } }); }), h(StatButton, { label: "all rows", value: Object.keys(counts).reduce(function (sum, key) { return sum + (counts[key] || 0); }, 0), active: status === "all", onClick: function () { setStatus("all"); } }))));

    const queue = h(Card, null,
      h(CardHeader, null, h(CardTitle, { className: "text-base" }, "Rows — " + labelStatus(status) + (query ? " — search" : ""))),
      h(CardContent, null, h("div", { className: "flex flex-col gap-3" },
        rows.length === 0 ? h(EmptyState, { status: status, counts: counts, setStatus: setStatus, clearFilters: clearFilters }) : null,
        rows.map(function (row) { return h(RowCard, { key: row.id, row: row, onDetail: fetchDetail, onSelect: setSelected, onMark: mark }); }))));

    const promoteCard = h(Card, null, h(CardHeader, null, h(CardTitle, { className: "text-base" }, "Promote to built-in memory")), h(CardContent, null, selected ? h("div", { className: "flex flex-col gap-3" }, h("div", { className: "flex gap-2" }, h(Button, { onClick: function () { setTarget("memory"); }, className: target === "memory" ? "bg-foreground/10" : "" }, "MEMORY.md"), h(Button, { onClick: function () { setTarget("user"); }, className: target === "user" ? "bg-foreground/10" : "" }, "USER.md")), h("textarea", { value: edited, onChange: function (e) { setEdited(e.target.value); }, className: "min-h-40 w-full border border-border bg-background p-3 font-courier text-sm", spellCheck: false }), h("p", { className: "text-xs text-muted-foreground" }, "Promotion is explicit and audited. Keep entries compact, durable, and safe for system-prompt injection."), selected.status === "rejected" ? h("label", { className: "flex items-center gap-2 text-xs" }, h("input", { type: "checkbox", checked: allowRejected, onChange: function (e) { setAllowRejected(e.target.checked); } }), "Allow rejected override") : null, h(Button, { onClick: promote }, "Promote reviewed entry")) : h("p", { className: "text-sm text-muted-foreground" }, "Click Review / promote on any row first. Nothing is written to built-in memory until you press Promote reviewed entry.")));

    const detailCard = h(Card, null, h(CardHeader, null, h(CardTitle, { className: "text-base" }, "Row detail")), h(CardContent, null, detail ? h("div", { className: "flex flex-col gap-2 text-sm" }, h("div", { className: "font-courier text-xs text-muted-foreground" }, detail.id), h("div", null, pill(detail.status), pill(detail.type), pill("quality " + score(detail.quality_score))), h("pre", { className: "whitespace-pre-wrap border border-border bg-background/40 p-2 font-courier text-xs" }, JSON.stringify(detail, null, 2))) : h("p", { className: "text-sm text-muted-foreground" }, "Click Open details on any row to inspect raw metadata, quality signals, and audit-relevant fields.")));

    const consolidationCard = h(Card, null, h(CardHeader, null, h(CardTitle, { className: "text-base" }, "Consolidation suggestions")), h(CardContent, null, h("div", { className: "flex flex-col gap-2" }, consolidations.length === 0 ? h("p", { className: "text-sm text-muted-foreground" }, "No useful duplicate groups right now.") : null, consolidations.map(function (g, i) { return h("div", { key: i, className: "border border-border p-2 text-sm" }, h("div", { className: "mb-1 flex flex-wrap gap-2" }, pill(g.subject_key || "group"), pill("quality " + score(g.canonical_quality_score)), pill("dupes " + (g.duplicate_count || 0))), h("p", { className: "mb-2 text-muted-foreground" }, g.recommended_action || "review"), h(Button, { onClick: function () { applyGroup(g); } }, "Apply reviewed consolidation")); }))));

    const auditCard = h(Card, null, h(CardHeader, null, h(CardTitle, { className: "text-base" }, "Recent audit")), h(CardContent, null, h("div", { className: "flex flex-col gap-2" }, audit.map(function (event) { return h("div", { key: event.event_id || event.seq, className: "border border-border p-2 font-courier text-xs" }, h("div", null, "#" + event.seq + " " + event.operation), h("div", { className: "text-muted-foreground" }, event.created_at)); }))));

    return h("div", { className: "flex flex-col gap-6" }, header, h(HowToUse, { candidateCount: candidateCount, setStatus: setStatus }), statsCard, message ? h(Card, { className: "border-dashed" }, h(CardContent, { className: "py-3 text-sm font-courier" }, message)) : null, h("div", { className: "grid gap-6 xl:grid-cols-[1.5fr_1fr]" }, queue, h("div", { className: "flex flex-col gap-6" }, promoteCard, detailCard, consolidationCard, auditCard)));
  }

  window.__HERMES_PLUGINS__.register("recall", RecallPage);
})();

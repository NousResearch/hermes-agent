import { useMemo, useState } from "react";
import {
  getSkills,
  getToolsets,
  installSkillFromHub,
  searchSkillsHub,
  toggleSkill,
  toggleToolset,
  type SkillHubResult,
  type SkillHubSearchResponse,
  type SkillInfo,
  type ToolsetInfo,
} from "@/api/skills";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";
import { ApiError, pollAction } from "@/api/client";

type Note = { kind: "ok" | "err"; msg: string } | null;

function errMsg(e: unknown): string {
  return e instanceof ApiError ? e.message : String(e);
}

export default function SkillsPage() {
  const skills = useResource(getSkills);
  const toolsets = useResource(getToolsets);
  const [filter, setFilter] = useState("");
  const [note, setNote] = useState<Note>(null);

  return (
    <ManagementPage
      title="Skills & Toolsets"
      actions={
        <input
          className="ht-input ht-input--sm"
          placeholder="Filter…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="Filter skills"
        />
      }
    >
      {note && (
        <p
          className={note.kind === "ok" ? "ht-ok" : "ht-error-inline"}
          style={{ margin: "0 0 12px" }}
        >
          {note.msg}
        </p>
      )}

      <section className="ht-card">
        <h2 className="ht-card__title">Skills</h2>
        <ResourceView resource={skills} empty={(d) => d.length === 0}>
          {(list) => (
            <SkillList
              skills={list}
              filter={filter}
              onChanged={skills.reload}
              onError={(msg) => setNote({ kind: "err", msg })}
            />
          )}
        </ResourceView>
      </section>

      <section className="ht-card">
        <h2 className="ht-card__title">Toolsets</h2>
        <ResourceView resource={toolsets} empty={(d) => d.length === 0}>
          {(list) => (
            <ToolsetList
              toolsets={list}
              filter={filter}
              onChanged={toolsets.reload}
              onError={(msg) => setNote({ kind: "err", msg })}
            />
          )}
        </ResourceView>
      </section>

      <HubSection onInstalled={skills.reload} />
    </ManagementPage>
  );
}

// ── Installed skills (grouped by category) ──────────────────────────
function SkillList({
  skills,
  filter,
  onChanged,
  onError,
}: {
  skills: SkillInfo[];
  filter: string;
  onChanged: () => void;
  onError: (msg: string) => void;
}) {
  const groups = useMemo(() => {
    const q = filter.trim().toLowerCase();
    const matched = skills.filter(
      (s) =>
        !q ||
        s.name.toLowerCase().includes(q) ||
        s.description.toLowerCase().includes(q) ||
        s.category.toLowerCase().includes(q),
    );
    const byCat = new Map<string, SkillInfo[]>();
    for (const s of matched) {
      const cat = s.category || "other";
      const arr = byCat.get(cat) ?? [];
      arr.push(s);
      byCat.set(cat, arr);
    }
    return [...byCat.entries()]
      .map(([cat, items]) => [cat, items.sort((a, b) => a.name.localeCompare(b.name))] as const)
      .sort(([a], [b]) => a.localeCompare(b));
  }, [skills, filter]);

  if (groups.length === 0) return <p className="ht-muted">No matching skills.</p>;

  return (
    <div className="ht-providers">
      {groups.map(([cat, items]) => (
        <div key={cat} className="ht-provider">
          <div className="ht-provider__head">
            <span className="ht-provider__name">{cat}</span>
            <span className="ht-provider__count">{items.length}</span>
          </div>
          <ul className="ht-model-list">
            {items.map((s) => (
              <SkillRow key={s.name} skill={s} onChanged={onChanged} onError={onError} />
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

function SkillRow({
  skill,
  onChanged,
  onError,
}: {
  skill: SkillInfo;
  onChanged: () => void;
  onError: (msg: string) => void;
}) {
  const [busy, setBusy] = useState(false);

  const toggle = async () => {
    setBusy(true);
    try {
      await toggleSkill(skill.name, !skill.enabled);
      onChanged();
    } catch (e) {
      onError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <li className="ht-model">
      <div style={{ flex: 1 }}>
        <code>{skill.name}</code>
        {skill.description && <div className="ht-muted ht-sm">{skill.description}</div>}
      </div>
      <span className={skill.enabled ? "ht-chip ht-chip--ok" : "ht-chip"}>
        {skill.enabled ? "enabled" : "disabled"}
      </span>
      <button
        type="button"
        className={`ht-btn ht-btn--sm${skill.enabled ? " ht-btn--ghost" : ""}`}
        disabled={busy}
        onClick={toggle}
        aria-label={`${skill.enabled ? "Disable" : "Enable"} ${skill.name}`}
      >
        {skill.enabled ? "Disable" : "Enable"}
      </button>
    </li>
  );
}

// ── Installed toolsets ──────────────────────────────────────────────
function ToolsetList({
  toolsets,
  filter,
  onChanged,
  onError,
}: {
  toolsets: ToolsetInfo[];
  filter: string;
  onChanged: () => void;
  onError: (msg: string) => void;
}) {
  const rows = useMemo(() => {
    const q = filter.trim().toLowerCase();
    return toolsets
      .filter(
        (t) =>
          !q ||
          t.name.toLowerCase().includes(q) ||
          t.label.toLowerCase().includes(q) ||
          t.description.toLowerCase().includes(q),
      )
      .sort((a, b) => a.label.localeCompare(b.label));
  }, [toolsets, filter]);

  if (rows.length === 0) return <p className="ht-muted">No matching toolsets.</p>;

  return (
    <ul className="ht-model-list">
      {rows.map((t) => (
        <ToolsetRow key={t.name} toolset={t} onChanged={onChanged} onError={onError} />
      ))}
    </ul>
  );
}

function ToolsetRow({
  toolset,
  onChanged,
  onError,
}: {
  toolset: ToolsetInfo;
  onChanged: () => void;
  onError: (msg: string) => void;
}) {
  const [busy, setBusy] = useState(false);

  const toggle = async () => {
    setBusy(true);
    try {
      await toggleToolset(toolset.name, !toolset.enabled);
      onChanged();
    } catch (e) {
      onError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <li className="ht-model">
      <div style={{ flex: 1 }}>
        <code>{toolset.label}</code>
        {toolset.description && <div className="ht-muted ht-sm">{toolset.description}</div>}
        <div className="ht-chips">
          <span className="ht-chip">{toolset.tools.length} tools</span>
          {!toolset.configured && <span className="ht-chip ht-chip--warn">not configured</span>}
        </div>
      </div>
      <span className={toolset.enabled ? "ht-chip ht-chip--ok" : "ht-chip"}>
        {toolset.enabled ? "enabled" : "disabled"}
      </span>
      <button
        type="button"
        className={`ht-btn ht-btn--sm${toolset.enabled ? " ht-btn--ghost" : ""}`}
        disabled={busy}
        onClick={toggle}
        aria-label={`${toolset.enabled ? "Disable" : "Enable"} ${toolset.label}`}
      >
        {toolset.enabled ? "Disable" : "Enable"}
      </button>
    </li>
  );
}

// ── Skills hub (collapsible; guarded so a hub outage never blanks the page) ──
function HubSection({ onInstalled }: { onInstalled: () => void }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<SkillHubSearchResponse | null>(null);
  const [hubError, setHubError] = useState<string | null>(null);
  const [note, setNote] = useState<Note>(null);
  const [installing, setInstalling] = useState<string | null>(null);
  const [progress, setProgress] = useState<string[]>([]);

  const runSearch = async () => {
    const q = query.trim();
    if (!q) return;
    setSearching(true);
    setHubError(null);
    setNote(null);
    try {
      const res = await searchSkillsHub(q);
      setResults(res);
    } catch (e) {
      // A hub failure is contained here — the installed sections stay usable.
      setHubError(errMsg(e));
      setResults(null);
    } finally {
      setSearching(false);
    }
  };

  const install = async (r: SkillHubResult) => {
    setInstalling(r.identifier);
    setProgress([]);
    setNote(null);
    try {
      const action = await installSkillFromHub(r.identifier);
      if (!action.ok || !action.name) {
        setNote({ kind: "err", msg: action.error ?? "Install could not be started." });
        return;
      }
      const status = await pollAction(action.name, {
        onProgress: (s) => setProgress(s.lines),
      });
      if (status.exit_code === 0) {
        setNote({ kind: "ok", msg: `Installed ${r.name}.` });
        onInstalled();
      } else {
        setNote({ kind: "err", msg: `Install of ${r.name} failed (exit ${status.exit_code}).` });
      }
    } catch (e) {
      setNote({ kind: "err", msg: errMsg(e) });
    } finally {
      setInstalling(null);
    }
  };

  return (
    <section className="ht-card">
      <button
        type="button"
        className="ht-provider__head"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="ht-provider__name">Skills hub</span>
        <span className="ht-muted ht-sm">search &amp; install from the community hub</span>
        <span aria-hidden>{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div>
          <div className="ht-row-actions" style={{ margin: "12px 0" }}>
            <input
              className="ht-input ht-input--sm"
              placeholder="Search the hub…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void runSearch();
              }}
              aria-label="Search skills hub"
            />
            <button
              type="button"
              className="ht-btn ht-btn--sm"
              disabled={searching || !query.trim()}
              onClick={() => void runSearch()}
            >
              {searching ? "Searching…" : "Search"}
            </button>
          </div>

          {hubError && (
            <p className="ht-error-inline">Hub unavailable: {hubError}</p>
          )}
          {note && (
            <p className={note.kind === "ok" ? "ht-ok" : "ht-error-inline"}>{note.msg}</p>
          )}
          {installing && progress.length > 0 && (
            <pre className="ht-sm ht-muted" style={{ maxHeight: 160, overflow: "auto" }}>
              {progress.join("\n")}
            </pre>
          )}

          {results && (
            <HubResults
              results={results}
              installing={installing}
              onInstall={(r) => void install(r)}
            />
          )}
        </div>
      )}
    </section>
  );
}

function HubResults({
  results,
  installing,
  onInstall,
}: {
  results: SkillHubSearchResponse;
  installing: string | null;
  onInstall: (r: SkillHubResult) => void;
}) {
  if (results.results.length === 0) return <p className="ht-muted">No hub results.</p>;

  return (
    <ul className="ht-model-list">
      {results.results.map((r) => {
        const alreadyInstalled = Boolean(results.installed[r.identifier]);
        const busy = installing === r.identifier;
        return (
          <li key={r.identifier} className="ht-model">
            <div style={{ flex: 1 }}>
              <code>{r.name}</code>
              {r.description && <div className="ht-muted ht-sm">{r.description}</div>}
              <div className="ht-chips">
                <span className="ht-chip">{r.source}</span>
                <span className="ht-chip">{r.trust_level}</span>
                {r.tags.slice(0, 3).map((tag) => (
                  <span key={tag} className="ht-chip">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
            {alreadyInstalled ? (
              <span className="ht-chip ht-chip--ok">installed</span>
            ) : (
              <button
                type="button"
                className="ht-btn ht-btn--sm"
                disabled={busy || installing !== null}
                onClick={() => onInstall(r)}
              >
                {busy ? "Installing…" : "Install"}
              </button>
            )}
          </li>
        );
      })}
    </ul>
  );
}

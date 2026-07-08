import { useMemo, useState } from "react";
import {
  deleteEnvVar,
  getEnvVars,
  revealEnvVar,
  setEnvVar,
  type EnvVarInfo,
} from "@/api/endpoints";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";

export default function EnvPage() {
  const env = useResource(getEnvVars);
  const [filter, setFilter] = useState("");

  return (
    <ManagementPage
      title="Environment & Keys"
      actions={
        <input
          className="ht-input ht-input--sm"
          placeholder="Filter…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="Filter keys"
        />
      }
    >
      <ResourceView resource={env}>
        {(vars) => <EnvTable vars={vars} filter={filter} onChanged={env.reload} />}
      </ResourceView>
    </ManagementPage>
  );
}

function EnvTable({
  vars,
  filter,
  onChanged,
}: {
  vars: Record<string, EnvVarInfo>;
  filter: string;
  onChanged: () => void;
}) {
  const rows = useMemo(() => {
    const q = filter.trim().toUpperCase();
    return Object.entries(vars)
      .filter(([k]) => !q || k.toUpperCase().includes(q))
      .sort(([a], [b]) => a.localeCompare(b));
  }, [vars, filter]);

  if (rows.length === 0) return <p className="ht-muted">No matching keys.</p>;

  return (
    <table className="ht-table">
      <thead>
        <tr>
          <th>Key</th>
          <th>Value</th>
          <th>Category</th>
          <th />
        </tr>
      </thead>
      <tbody>
        {rows.map(([key, info]) => (
          <EnvRow key={key} name={key} info={info} onChanged={onChanged} />
        ))}
      </tbody>
    </table>
  );
}

function EnvRow({
  name,
  info,
  onChanged,
}: {
  name: string;
  info: EnvVarInfo;
  onChanged: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState("");
  const [revealed, setRevealed] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const save = async () => {
    setBusy(true);
    try {
      await setEnvVar(name, value);
      setEditing(false);
      setValue("");
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    setBusy(true);
    try {
      await deleteEnvVar(name);
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  const reveal = async () => {
    const res = await revealEnvVar(name);
    setRevealed(res.value);
  };

  return (
    <tr>
      <td>
        <code>{name}</code>
        {info.description && <div className="ht-muted ht-sm">{info.description}</div>}
      </td>
      <td>
        {editing ? (
          <input
            className="ht-input ht-input--sm"
            type={info.is_password ? "password" : "text"}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="New value"
            autoFocus
          />
        ) : revealed != null ? (
          <code>{revealed}</code>
        ) : info.is_set ? (
          <span className="ht-mono">{info.redacted_value ?? "••••"}</span>
        ) : (
          <span className="ht-dim">not set</span>
        )}
      </td>
      <td>
        <span className="ht-chip">{info.category}</span>
      </td>
      <td className="ht-row-actions">
        {editing ? (
          <>
            <button type="button" className="ht-btn ht-btn--sm" disabled={busy} onClick={save}>
              Save
            </button>
            <button
              type="button"
              className="ht-btn ht-btn--sm ht-btn--ghost"
              onClick={() => setEditing(false)}
            >
              Cancel
            </button>
          </>
        ) : (
          <>
            {info.is_set && info.is_password && revealed == null && (
              <button type="button" className="ht-btn ht-btn--sm ht-btn--ghost" onClick={reveal}>
                Reveal
              </button>
            )}
            <button type="button" className="ht-btn ht-btn--sm" onClick={() => setEditing(true)}>
              {info.is_set ? "Edit" : "Set"}
            </button>
            {info.is_set && (
              <button
                type="button"
                className="ht-btn ht-btn--sm ht-btn--stop"
                disabled={busy}
                onClick={remove}
              >
                Delete
              </button>
            )}
          </>
        )}
      </td>
    </tr>
  );
}

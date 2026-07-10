import { useEffect, useState } from "react";
import {
  createProfile,
  deleteProfile,
  getActiveProfile,
  getProfiles,
  getProfileSoul,
  renameProfile,
  setActiveProfile,
  updateProfileSoul,
  type ProfileInfo,
} from "@/api/profiles";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";
import { ApiError } from "@/api/client";

interface ProfilesData {
  profiles: ProfileInfo[];
  active: string;
}

export default function ProfilesPage() {
  const profiles = useResource<ProfilesData>(async () => {
    const [list, active] = await Promise.all([getProfiles(), getActiveProfile()]);
    return { profiles: list.profiles, active: active.active };
  });
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <ManagementPage
      title="Profiles"
      actions={
        <button type="button" className="ht-btn ht-btn--sm" onClick={profiles.reload}>
          Refresh
        </button>
      }
    >
      <ResourceView resource={profiles} empty={(d) => d.profiles.length === 0}>
        {(d) => (
          <>
            <CreateProfileForm onCreated={profiles.reload} />
            <div className="ht-card">
              <div className="ht-card__title">Profiles</div>
              <table className="ht-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Model</th>
                    <th>Description</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {d.profiles.map((p) => (
                    <ProfileRow
                      key={p.name}
                      p={p}
                      active={d.active === p.name}
                      selected={selected === p.name}
                      onSelect={() => setSelected(p.name)}
                      onChanged={profiles.reload}
                    />
                  ))}
                </tbody>
              </table>
            </div>
            {selected && <SoulEditor key={selected} name={selected} />}
          </>
        )}
      </ResourceView>
    </ManagementPage>
  );
}

function CreateProfileForm({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<{ kind: "ok" | "err"; msg: string } | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    setNote(null);
    try {
      await createProfile({
        name: name.trim(),
        description: description.trim() || undefined,
      });
      setName("");
      setDescription("");
      setNote({ kind: "ok", msg: "Profile created." });
      onCreated();
    } catch (err) {
      setNote({ kind: "err", msg: err instanceof ApiError ? err.message : String(err) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="ht-card" onSubmit={submit}>
      <div className="ht-card__title">Create profile</div>
      <div className="ht-chips">
        <input
          className="ht-input ht-input--sm"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Name"
          aria-label="Profile name"
        />
        <input
          className="ht-input ht-input--sm"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Description (optional)"
          aria-label="Profile description"
        />
        <button type="submit" className="ht-btn ht-btn--sm" disabled={busy || !name.trim()}>
          {busy ? "Creating…" : "Create"}
        </button>
        {note && (
          <span className={note.kind === "ok" ? "ht-ok" : "ht-error-inline"}>{note.msg}</span>
        )}
      </div>
    </form>
  );
}

function ProfileRow({
  p,
  active,
  selected,
  onSelect,
  onChanged,
}: {
  p: ProfileInfo;
  active: boolean;
  selected: boolean;
  onSelect: () => void;
  onChanged: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(p.name);
  const [busy, setBusy] = useState(false);

  // The active profile and the default profile can't be removed — the API
  // rejects both, so the button is disabled to make the constraint visible.
  const canDelete = !active && !p.is_default;

  const activate = async () => {
    setBusy(true);
    try {
      await setActiveProfile(p.name);
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  const rename = async () => {
    setBusy(true);
    try {
      await renameProfile(p.name, name.trim());
      setEditing(false);
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!window.confirm(`Delete profile "${p.name}"? This cannot be undone.`)) return;
    setBusy(true);
    try {
      await deleteProfile(p.name);
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  return (
    <tr>
      <td>
        {editing ? (
          <input
            className="ht-input ht-input--sm"
            value={name}
            onChange={(e) => setName(e.target.value)}
            autoFocus
          />
        ) : (
          <>
            <span>{p.name}</span>{" "}
            {active && <span className="ht-chip ht-chip--ok">active</span>}
            {p.is_default && <span className="ht-chip">default</span>}
          </>
        )}
      </td>
      <td>
        {p.model ? (
          <span className="ht-sm">{p.model}</span>
        ) : (
          <span className="ht-dim">—</span>
        )}
      </td>
      <td>
        {p.description ? (
          <span className="ht-muted ht-sm">{p.description}</span>
        ) : (
          <span className="ht-dim">—</span>
        )}
      </td>
      <td className="ht-row-actions">
        {editing ? (
          <>
            <button type="button" className="ht-btn ht-btn--sm" disabled={busy} onClick={rename}>
              Save
            </button>
            <button
              type="button"
              className="ht-btn ht-btn--sm ht-btn--ghost"
              onClick={() => {
                setName(p.name);
                setEditing(false);
              }}
            >
              Cancel
            </button>
          </>
        ) : (
          <>
            <button
              type="button"
              className="ht-btn ht-btn--sm"
              disabled={busy || active}
              onClick={activate}
            >
              {active ? "Active" : "Activate"}
            </button>
            <button
              type="button"
              className="ht-btn ht-btn--sm ht-btn--ghost"
              onClick={onSelect}
            >
              {selected ? "Editing SOUL" : "Edit SOUL"}
            </button>
            <button type="button" className="ht-btn ht-btn--sm" onClick={() => setEditing(true)}>
              Rename
            </button>
            <button
              type="button"
              className="ht-btn ht-btn--sm ht-btn--stop"
              disabled={busy || !canDelete}
              title={canDelete ? undefined : "The active/default profile can't be deleted"}
              onClick={remove}
            >
              Delete
            </button>
          </>
        )}
      </td>
    </tr>
  );
}

function SoulEditor({ name }: { name: string }) {
  const soul = useResource(() => getProfileSoul(name), [name]);

  return (
    <div className="ht-card">
      <div className="ht-card__title">SOUL.md — {name}</div>
      <ResourceView resource={soul}>
        {(data) => (
          <SoulEditorForm
            name={name}
            initial={data.content}
            exists={data.exists}
            onSaved={soul.reload}
          />
        )}
      </ResourceView>
    </div>
  );
}

function SoulEditorForm({
  name,
  initial,
  exists,
  onSaved,
}: {
  name: string;
  initial: string;
  exists: boolean;
  onSaved: () => void;
}) {
  const [text, setText] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<{ kind: "ok" | "err"; msg: string } | null>(null);
  const dirty = text !== initial;

  useEffect(() => setText(initial), [initial]);

  const save = async () => {
    setBusy(true);
    setNote(null);
    try {
      await updateProfileSoul(name, text);
      setNote({ kind: "ok", msg: "Saved. SOUL.md updated." });
      onSaved();
    } catch (e) {
      setNote({ kind: "err", msg: e instanceof ApiError ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="ht-editor">
      <p className="ht-muted">
        The <code>SOUL.md</code> persona for <strong>{name}</strong>.
        {!exists && <span className="ht-dim"> (no file yet — saving creates it)</span>}
      </p>
      <textarea
        className="ht-code-editor"
        value={text}
        onChange={(e) => setText(e.target.value)}
        spellCheck={false}
        aria-label={`SOUL.md for ${name}`}
      />
      <div className="ht-editor__bar">
        <button type="button" className="ht-btn" disabled={busy || !dirty} onClick={save}>
          {busy ? "Saving…" : "Save"}
        </button>
        {dirty && (
          <button type="button" className="ht-btn ht-btn--ghost" onClick={() => setText(initial)}>
            Revert
          </button>
        )}
        {note && (
          <span className={note.kind === "ok" ? "ht-ok" : "ht-error-inline"}>{note.msg}</span>
        )}
      </div>
    </div>
  );
}

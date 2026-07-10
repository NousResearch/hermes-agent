import { useRef, useState } from "react";
import {
  createDirectory,
  deleteFile,
  listFiles,
  uploadFile,
  type ManagedFileEntry,
  type ManagedFilesResponse,
} from "@/api/files";
import { ApiError } from "@/api/client";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";
import { fmtBytes, relTime } from "./mgmtFormat";

// A functional managed-files browser: breadcrumb navigation, per-entry delete,
// a "New folder" action and a streamed upload. Not a full file manager (no
// rename/move/preview) — just enough to inspect and curate the managed tree.
export default function FilesPage() {
  const [path, setPath] = useState<string | undefined>(undefined);
  const files = useResource<ManagedFilesResponse>(() => listFiles(path), [path]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const uploadRef = useRef<HTMLInputElement>(null);

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      files.reload();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onNewFolder = (cwd: string) => {
    const name = window.prompt("New folder name");
    if (!name || !name.trim()) return;
    const base = cwd.replace(/\/+$/, "");
    const target = base ? `${base}/${name.trim()}` : name.trim();
    void run(() => createDirectory(target));
  };

  const onUpload = (cwd: string, file: File | undefined) => {
    if (!file) return;
    void run(() => uploadFile(cwd, file));
  };

  const onDelete = (entry: ManagedFileEntry) => {
    const msg = entry.is_directory
      ? `Delete folder "${entry.name}" and all its contents?`
      : `Delete file "${entry.name}"?`;
    if (!window.confirm(msg)) return;
    void run(() => deleteFile(entry.path, entry.is_directory));
  };

  return (
    <ManagementPage
      title="Files"
      actions={
        <button
          type="button"
          className="ht-btn ht-btn--sm"
          onClick={files.reload}
          disabled={busy}
        >
          Refresh
        </button>
      }
    >
      {error && <div className="ht-error-inline">{error}</div>}
      <ResourceView resource={files}>
        {(d) => (
          <>
            <Breadcrumb data={d} onNavigate={setPath} />

            <div className="ht-row-actions" style={{ margin: "0.5rem 0" }}>
              <button
                type="button"
                className="ht-btn ht-btn--sm"
                disabled={busy}
                onClick={() => onNewFolder(d.path)}
              >
                New folder
              </button>
              <button
                type="button"
                className="ht-btn ht-btn--sm ht-btn--ghost"
                disabled={busy}
                onClick={() => uploadRef.current?.click()}
              >
                Upload
              </button>
              <input
                ref={uploadRef}
                type="file"
                hidden
                onChange={(e) => {
                  onUpload(d.path, e.target.files?.[0]);
                  e.target.value = "";
                }}
              />
            </div>

            <div className="ht-chips">
              {d.root && (
                <span className="ht-chip" title="Browse root">
                  root: {d.root}
                </span>
              )}
              {d.locked_root && (
                <span className="ht-chip ht-chip--warn" title="Locked to this root">
                  locked: {d.locked_root}
                </span>
              )}
              {!d.can_change_path && <span className="ht-chip">path locked</span>}
            </div>

            {d.entries.length === 0 ? (
              <p className="ht-muted">Empty folder.</p>
            ) : (
              <table className="ht-table">
                <thead>
                  <tr>
                    <th />
                    <th>Name</th>
                    <th>Size</th>
                    <th>Modified</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {d.entries.map((entry) => (
                    <tr key={entry.path}>
                      <td aria-hidden>{entry.is_directory ? "📁" : "📄"}</td>
                      <td>
                        {entry.is_directory ? (
                          <button
                            type="button"
                            className="ht-btn ht-btn--sm ht-btn--ghost"
                            onClick={() => setPath(entry.path)}
                          >
                            {entry.name}/
                          </button>
                        ) : (
                          <span>{entry.name}</span>
                        )}
                      </td>
                      <td>{entry.is_directory ? "—" : fmtBytes(entry.size)}</td>
                      <td className="ht-sm ht-muted">{relTime(entry.mtime)}</td>
                      <td className="ht-row-actions">
                        <button
                          type="button"
                          className="ht-btn ht-btn--sm ht-btn--stop"
                          disabled={busy}
                          onClick={() => onDelete(entry)}
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}
      </ResourceView>
    </ManagementPage>
  );
}

// Breadcrumb built from the absolute path segments. Each ancestor segment is
// clickable to jump up; the "Up" chip uses the server-provided parent so it
// respects the locked root.
function Breadcrumb({
  data,
  onNavigate,
}: {
  data: ManagedFilesResponse;
  onNavigate: (path: string | undefined) => void;
}) {
  const segments = data.path.split("/").filter(Boolean);
  return (
    <nav className="ht-row-actions" aria-label="Breadcrumb">
      <button
        type="button"
        className="ht-btn ht-btn--sm ht-btn--ghost"
        disabled={data.parent === null}
        onClick={() => onNavigate(data.parent ?? undefined)}
      >
        ↑ Up
      </button>
      <span className="ht-sm ht-mono ht-muted">{data.path || "/"}</span>
      {segments.length > 1 && (
        <span className="ht-chips">
          {segments.map((seg, i) => {
            const segPath = "/" + segments.slice(0, i + 1).join("/");
            const isLast = i === segments.length - 1;
            return isLast ? (
              <span key={segPath} className="ht-chip">
                {seg}
              </span>
            ) : (
              <button
                key={segPath}
                type="button"
                className="ht-btn ht-btn--sm ht-btn--ghost"
                onClick={() => onNavigate(segPath)}
              >
                {seg}
              </button>
            );
          })}
        </span>
      )}
    </nav>
  );
}

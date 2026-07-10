import { useEffect, useState } from "react";
import { getConfigRaw, saveConfigRaw } from "@/api/endpoints";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";
import { ApiError } from "@/api/client";

export default function ConfigPage() {
  const config = useResource(getConfigRaw);

  return (
    <ManagementPage title="Config">
      <ResourceView resource={config}>
        {(data) => <ConfigEditor initial={data.yaml} onSaved={config.reload} />}
      </ResourceView>
    </ManagementPage>
  );
}

function ConfigEditor({ initial, onSaved }: { initial: string; onSaved: () => void }) {
  const [text, setText] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<{ kind: "ok" | "err"; msg: string } | null>(null);
  const dirty = text !== initial;

  // Re-seed the editor when a reload delivers fresh server content.
  useEffect(() => setText(initial), [initial]);

  const save = async () => {
    setBusy(true);
    setNote(null);
    try {
      await saveConfigRaw(text);
      setNote({ kind: "ok", msg: "Saved. config.yaml updated." });
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
        Raw <code>config.yaml</code>. Edits are validated server-side on save.
      </p>
      <textarea
        className="ht-code-editor"
        value={text}
        onChange={(e) => setText(e.target.value)}
        spellCheck={false}
        aria-label="config.yaml"
      />
      <div className="ht-editor__bar">
        <button type="button" className="ht-btn" disabled={busy || !dirty} onClick={save}>
          {busy ? "Saving…" : "Save"}
        </button>
        {dirty && (
          <button
            type="button"
            className="ht-btn ht-btn--ghost"
            onClick={() => setText(initial)}
          >
            Revert
          </button>
        )}
        {note && <span className={note.kind === "ok" ? "ht-ok" : "ht-error-inline"}>{note.msg}</span>}
      </div>
    </div>
  );
}

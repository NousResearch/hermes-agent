/* One knowledge corpus: what is in it, and putting documents into it.
 *
 * A corpus is a folder of the customer's own documents. Until now the only way a document
 * reached one was somebody putting a file on the host's disk.
 *
 * **Uploading indexes.** A document on disk that the index has never seen is invisible to
 * every agent, so the screen reports both facts separately — stored, and indexed. A
 * failed index is not a failed upload, and saying so is the difference between "try again"
 * and "the file is there, the search is not".
 *
 * **The corpus decides what it takes.** The accepted patterns and the size limit come from
 * the source's own declaration and are shown, so a rejection is predictable rather than a
 * surprise at the end of an upload.
 */

import * as React from "react";
import {
  AlertTriangle, Check, FileText, Loader2, RefreshCw, Trash2, Upload,
} from "lucide-react";

import { GlassPanel, SectionHeader, StatusPill } from "@/components/glass";
import { post } from "@/lib/api";
import { usePanel } from "@/lib/hooks";
import { sinceIso } from "@/lib/state";

type Doc = { name: string; bytes: number; modified_at: number; too_large: boolean };
type Corpus = {
  id: string; title: string; root: string; classification: string;
  accepts: string[]; excludes: string[]; max_file_bytes: number;
  documents: Doc[];
  indexed: { documents: number; chunks: number; detail: string };
  readable_by: string[];
};

type Result =
  | { kind: "idle" }
  | { kind: "busy"; what: string }
  | { kind: "done"; message: string; warning: string }
  | { kind: "error"; message: string };

function kb(bytes: number) {
  return bytes < 1000 ? `${bytes} B` : `${(bytes / 1000).toFixed(bytes < 100_000 ? 1 : 0)} kB`;
}

export function CorpusPanel({ sourceId, onChanged }: { sourceId: string; onChanged?: () => void }) {
  const [nonce, setNonce] = React.useState(0);
  const loaded = usePanel<Corpus>(
    `/knowledge/${encodeURIComponent(sourceId)}/documents`, 0, nonce,
  );
  const [result, setResult] = React.useState<Result>({ kind: "idle" });
  const [confirm, setConfirm] = React.useState<string | null>(null);

  const refresh = () => { setNonce((n) => n + 1); onChanged?.(); };
  const busy = result.kind === "busy";

  async function act(what: string, path: string, body: Record<string, unknown>) {
    if (busy) return;
    setResult({ kind: "busy", what });
    try {
      const response: any = await post(`/knowledge/${encodeURIComponent(sourceId)}/${path}`, body);
      const index = response?.index ?? {};
      setResult({
        kind: "done",
        message:
          path === "upload" ? `Stored ${response?.stored?.name ?? ""}`
          : path === "remove" ? `Removed ${response?.removed ?? ""}`
          : "Index rebuilt",
        warning: index.ok === false
          ? `Saved, but the index was not rebuilt: ${index.error ?? "unknown error"}. No agent can find it yet.`
          : (index.skipped ?? []).length
            ? `${index.skipped.length} file(s) the ingester skipped — see below.`
            : "",
      });
      refresh();
    } catch (cause) {
      setResult({
        kind: "error",
        message: cause instanceof Error ? cause.message : "the request did not complete",
      });
    }
  }

  async function upload(file: File, replace = false) {
    let data: string;
    try {
      data = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result ?? ""));
        reader.onerror = () => reject(new Error("the file could not be read"));
        reader.readAsDataURL(file);
      });
    } catch (cause) {
      // A file the browser could not read never reaches the control plane, and saying so
      // here is better than a request that fails for a reason the server cannot explain.
      setResult({
        kind: "error",
        message: cause instanceof Error ? cause.message : "the file could not be read",
      });
      return;
    }
    await act("upload", "upload", { filename: file.name, data, replace });
  }

  if (loaded.state === "loading") {
    return <GlassPanel className="p-5"><p className="text-ink-faint text-[13px]">reading…</p></GlassPanel>;
  }
  if (loaded.state === "forbidden") {
    return <GlassPanel className="p-5"><p className="text-ink-muted text-[13px]">Not visible to your role.</p></GlassPanel>;
  }
  if (loaded.state === "error") {
    return (
      <GlassPanel className="p-5">
        <p className="text-blocked text-[13px]"><b>This corpus could not be read.</b> {loaded.message}</p>
      </GlassPanel>
    );
  }

  const c = loaded.data;
  const notIndexed = c.documents.length > 0 && c.indexed.documents === 0;

  return (
    <GlassPanel className="p-5">
      <SectionHeader
        icon={FileText} title={c.title}
        detail={`${c.documents.length} document(s) on disk · ${c.indexed.documents} indexed · ${c.indexed.chunks} chunks`}
        action={
          <button type="button" onClick={refresh}
            className="text-ink-faint hover:text-ink inline-flex items-center gap-1.5 text-[11.5px]">
            <RefreshCw className="size-3" /> Refresh
          </button>
        }
      />

      {c.indexed.detail || notIndexed ? (
        <div className="border-waiting/30 mb-4 rounded-lg border p-3">
          <p className="text-waiting text-[12.5px] font-medium">
            {notIndexed ? "Nothing in this corpus is searchable." : "Not indexed."}
          </p>
          <p className="text-ink-muted mt-1 text-[12px] leading-relaxed">
            {c.indexed.detail || "These documents are on disk but the index has never seen them, so no agent can find them."}
          </p>
          <button
            type="button" disabled={busy}
            onClick={() => void act("reindex", "reindex", {})}
            className="glass-solid text-ink mt-2 inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12px] font-medium disabled:opacity-40"
          >
            {busy && result.what === "reindex" ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
            Rebuild the index
          </button>
        </div>
      ) : null}

      <div className="border-glass-border mb-4 flex flex-wrap items-center gap-3 rounded-lg border p-3">
        <label className="glass-solid text-ink inline-flex cursor-pointer items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium">
          {busy && result.what === "upload" ? <Loader2 className="size-3.5 animate-spin" /> : <Upload className="size-3.5" />}
          {busy && result.what === "upload" ? "Uploading…" : "Upload a document"}
          <input
            type="file" className="sr-only" disabled={busy}
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) void upload(file);
            }}
          />
        </label>
        <p className="text-ink-faint min-w-0 flex-1 text-[11.5px] leading-relaxed">
          This corpus takes <span className="font-mono">{c.accepts.join(", ")}</span>
          {c.excludes.length ? <> and excludes <span className="font-mono">{c.excludes.join(", ")}</span></> : null}
          , up to {kb(c.max_file_bytes)} per document. Uploading rebuilds the index so agents
          can find it.
        </p>
      </div>

      {result.kind === "error" ? (
        <div className="border-blocked/30 bg-blocked/5 mb-4 rounded-lg border p-3">
          <p className="text-blocked flex items-start gap-1.5 text-[12.5px]">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
            <span><b>Nothing was stored.</b> {result.message}</span>
          </p>
        </div>
      ) : null}
      {result.kind === "done" ? (
        <div className="border-glass-border mb-4 rounded-lg border p-3">
          <p className="text-running flex items-center gap-1.5 text-[12.5px]">
            <Check className="size-3.5" /> {result.message}
          </p>
          {result.warning ? (
            <p className="text-waiting mt-1 flex items-start gap-1.5 text-[12px]">
              <AlertTriangle className="mt-0.5 size-3 shrink-0" /> {result.warning}
            </p>
          ) : null}
        </div>
      ) : null}

      {c.documents.length === 0 ? (
        <p className="text-ink-muted text-[12.5px]">
          This corpus is empty. Upload a document, or put files in{" "}
          <span className="font-mono">{c.root}</span> on the host and rebuild the index.
        </p>
      ) : (
        <ul className="divide-glass-border divide-y">
          {c.documents.map((doc) => (
            <li key={doc.name} className="flex flex-wrap items-center gap-3 py-2 first:pt-0 last:pb-0">
              <FileText className="text-ink-faint size-3.5 shrink-0" />
              <span className="text-ink min-w-0 flex-1 truncate font-mono text-[12px]">{doc.name}</span>
              {doc.too_large ? (
                <StatusPill state="blocked">Too large to index</StatusPill>
              ) : null}
              <span className="text-ink-faint text-[11.5px]">{kb(doc.bytes)}</span>
              <span className="text-ink-faint text-[11.5px]">
                {sinceIso(new Date(doc.modified_at * 1000).toISOString())}
              </span>
              <button
                type="button" disabled={busy}
                onClick={() => setConfirm(confirm === doc.name ? null : doc.name)}
                aria-label={`Remove ${doc.name}`}
                className="text-ink-faint hover:text-blocked rounded-md p-1 transition-colors disabled:opacity-40"
              >
                <Trash2 className="size-3.5" />
              </button>
              {confirm === doc.name ? (
                <div className="border-blocked/30 w-full rounded-lg border p-2.5">
                  <p className="text-ink-muted text-[11.5px]">
                    Removing takes it off the host and out of the index. Agents stop being
                    able to cite it.
                  </p>
                  <button
                    type="button" disabled={busy}
                    onClick={() => { setConfirm(null); void act("remove", "remove", { name: doc.name }); }}
                    className="border-blocked/40 bg-blocked/10 text-blocked mt-2 inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-[11.5px] font-medium disabled:opacity-40"
                  >
                    <Trash2 className="size-3" /> Remove {doc.name}
                  </button>
                </div>
              ) : null}
            </li>
          ))}
        </ul>
      )}

      <p className="text-ink-faint border-glass-border mt-4 border-t pt-3 text-[11.5px] leading-relaxed">
        Stored at <span className="font-mono">{c.root}</span> on the host running NOVA.
        {c.readable_by.length
          ? <> Readable by <span className="font-mono">{c.readable_by.join(", ")}</span>.</>
          : <> No agent is granted this corpus yet, so nothing here is reachable.</>}
      </p>
    </GlassPanel>
  );
}

/* The save behaviour every editable section shares.
 *
 * Extracted from the Soul editor rather than invented: that one was driven in a browser
 * and got the awkward parts right, and re-implementing them per form is how six sections
 * end up with six subtly different ideas of "unsaved".
 *
 * Three properties are worth naming because they are easy to get wrong:
 *
 * **A null draft means "showing the server's copy".** Only once someone types does the
 * component hold an opinion. Panels poll on an interval, so a draft initialised from the
 * server would be silently replaced by the next refresh, and a half-typed edit would
 * vanish while somebody was looking at it.
 *
 * **Saved and applied are different facts.** A bundle edit is committed to disk and then
 * pushed into the runtime. The second can fail while the first succeeded, and the screen
 * has to be able to say so — one combined tick lets a half-finished change read as done.
 *
 * **A submission in flight blocks another.** Not cosmetic: two saves racing on the same
 * agent would have the second overwrite the first's validation result.
 */

import * as React from "react";
import { AlertTriangle, Check, Loader2, RotateCcw, Save } from "lucide-react";

export type SaveState =
  | { kind: "idle" }
  | { kind: "saving" }
  | { kind: "saved"; applied: boolean; detail: string; files: string[] }
  | { kind: "error"; message: string };

function same(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

export function useEditable<V>(server: V | undefined, save: (value: V) => Promise<any>) {
  const [draft, setDraft] = React.useState<V | null>(null);
  const [state, setState] = React.useState<SaveState>({ kind: "idle" });

  const value = (draft ?? server) as V;
  const dirty = draft !== null && server !== undefined && !same(draft, server);
  const busy = state.kind === "saving";

  const edit = React.useCallback(
    (next: V | ((current: V) => V)) => {
      setDraft((current) => {
        const base = (current ?? server) as V;
        return typeof next === "function" ? (next as (c: V) => V)(base) : next;
      });
      setState((s) => (s.kind === "idle" ? s : { kind: "idle" }));
    },
    [server],
  );

  const discard = React.useCallback(() => {
    setDraft(null);
    setState({ kind: "idle" });
  }, []);

  const submit = React.useCallback(async () => {
    if (busy || !dirty) return false;
    setState({ kind: "saving" });
    try {
      const result: any = await save(value);
      // A route that does not apply anything (a pure runtime action) reports no `runtime`
      // key. Treating that as "applied" is correct — there was nothing to apply — and is
      // different from a bundle write whose apply failed, which reports applied: false.
      const runtime = result?.runtime;
      const applied = runtime === undefined ? true : Boolean(runtime.applied);
      setState({
        kind: "saved",
        applied,
        detail: applied ? "" : String(runtime?.error ?? "the runtime was not updated"),
        files: result?.files_changed ?? [],
      });
      setDraft(null);
      return true;
    } catch (cause) {
      setState({
        kind: "error",
        message: cause instanceof Error ? cause.message : "the request did not complete",
      });
      return false;
    }
  }, [busy, dirty, save, value]);

  return { value, edit, discard, submit, dirty, busy, state, setState };
}

/** The control row and result message. Identical everywhere on purpose. */
export function SaveBar({
  dirty, busy, state, onSave, onDiscard, label = "Save", disabled = false, note,
}: {
  dirty: boolean; busy: boolean; state: SaveState;
  onSave: () => void; onDiscard: () => void;
  label?: string; disabled?: boolean; note?: React.ReactNode;
}) {
  return (
    <div className="mt-3">
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button" onClick={onSave} disabled={!dirty || busy || disabled}
          className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5
                     text-[12.5px] font-medium transition-opacity disabled:opacity-40"
        >
          {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Save className="size-3.5" />}
          {busy ? "Saving…" : label}
        </button>
        {dirty && !busy ? (
          <button
            type="button" onClick={onDiscard}
            className="text-ink-faint hover:text-ink inline-flex items-center gap-1.5 text-[12.5px]"
          >
            <RotateCcw className="size-3.5" /> Discard
          </button>
        ) : null}
        {dirty && !busy ? (
          <span className="text-waiting text-[11.5px]">Unsaved changes</span>
        ) : null}
        {note ? <span className="text-ink-faint ml-auto text-[11.5px]">{note}</span> : null}
      </div>
      <SaveResult state={state} />
    </div>
  );
}

export function SaveResult({ state }: { state: SaveState }) {
  if (state.kind === "error") {
    return (
      <div className="border-blocked/30 bg-blocked/5 mt-3 rounded-lg border p-3">
        <p className="text-blocked flex items-start gap-1.5 text-[12.5px]">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
          <span>
            <b>Nothing was changed.</b> {state.message}
          </span>
        </p>
      </div>
    );
  }
  if (state.kind !== "saved") return null;
  return (
    <div className="border-glass-border mt-3 space-y-1 rounded-lg border p-3 text-[12.5px]">
      <p className="text-running flex items-center gap-1.5">
        <Check className="size-3.5" /> Saved{state.files.length ? ` to ${state.files.join(", ")}` : ""}
      </p>
      {state.applied ? (
        <p className="text-running flex items-center gap-1.5">
          <Check className="size-3.5" /> Applied — the runtime has it now
        </p>
      ) : (
        <p className="text-waiting flex items-start gap-1.5">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
          <span>
            Saved, but not applied: {state.detail}. The declaration is stored; the runtime
            still has the previous one.
          </span>
        </p>
      )}
    </div>
  );
}

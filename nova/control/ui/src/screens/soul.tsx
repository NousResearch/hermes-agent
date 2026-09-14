/* Editing an agent's persona — against the bundle, which is what survives.
 *
 * The runtime reads an agent's identity from <profile>/SOUL.md, and that file is derived:
 * the materialiser composes it from the bundle's instructions plus tenant branding and a
 * knowledge briefing, and rewrites it on every apply. So this editor loads and saves the
 * *bundle's* copy. Editing the materialised file would look identical right up to the next
 * apply, which is when it would silently revert.
 *
 * The screen says two separate things after a save, because they are two separate facts:
 * the declaration was saved, and the runtime was updated. A save that applied cleanly shows
 * both; a save whose apply failed shows the first as done and the second as the reason. One
 * combined tick would let a half-finished change read as finished.
 */

import * as React from "react";
import { AlertTriangle, Check, Loader2, RotateCcw, Save } from "lucide-react";

import { GlassPanel, SectionHeader } from "@/components/glass";
import { post } from "@/lib/api";
import { usePanel } from "@/lib/hooks";

type SoulPayload = {
  agent_id: string;
  instructions: string;
  path: string;
  inline: boolean;
};

type SaveState =
  | { kind: "idle" }
  | { kind: "saving" }
  | { kind: "saved"; applied: boolean; detail: string; files: string[] }
  | { kind: "error"; message: string };

const MAX_CHARS = 20000;

export function SoulEditor({ agentId }: { agentId: string }) {
  const [nonce, setNonce] = React.useState(0);
  const loaded = usePanel<SoulPayload>(`/agents/${encodeURIComponent(agentId)}/soul`, 120000, nonce);

  const [draft, setDraft] = React.useState<string | null>(null);
  const [state, setState] = React.useState<SaveState>({ kind: "idle" });

  const original = loaded.state === "ok" ? loaded.data.instructions : "";
  // Null draft means "showing what the server has". Only once someone types does the
  // component hold an opinion, so a background refresh cannot clobber an unsaved edit.
  const text = draft ?? original;
  const dirty = draft !== null && draft !== original;
  const tooLong = text.length > MAX_CHARS;

  async function save() {
    setState({ kind: "saving" });
    try {
      const result: any = await post(`/agents/${encodeURIComponent(agentId)}/soul`, {
        instructions: text,
      });
      const applied = Boolean(result?.runtime?.applied);
      setState({
        kind: "saved",
        applied,
        detail: applied ? "" : String(result?.runtime?.error ?? "the runtime was not updated"),
        files: result?.files_changed ?? [],
      });
      setDraft(null);
      setNonce((n) => n + 1);
    } catch (cause) {
      setState({ kind: "error", message: cause instanceof Error ? cause.message : "save failed" });
    }
  }

  if (loaded.state === "loading") {
    return (
      <GlassPanel className="p-5">
        <p className="text-ink-faint text-[13px]">reading…</p>
      </GlassPanel>
    );
  }
  if (loaded.state === "forbidden") {
    return (
      <GlassPanel className="p-5">
        <p className="text-ink-muted text-[13px]">
          A persona is an agent's standing instruction on every turn, so it is not visible to
          your role.
        </p>
      </GlassPanel>
    );
  }
  if (loaded.state === "error") {
    return (
      <GlassPanel className="p-5">
        <p className="text-blocked text-[13px]">{loaded.message}</p>
      </GlassPanel>
    );
  }

  return (
    <GlassPanel className="p-5">
      <SectionHeader
        title="Soul"
        detail="Who this agent is. Prepended to every turn it takes."
        action={
          <span className="text-ink-faint font-mono text-[11.5px]">
            {loaded.data.inline ? "declared inline" : loaded.data.path}
          </span>
        }
      />

      <label className="sr-only" htmlFor={`soul-${agentId}`}>
        Agent persona
      </label>
      <textarea
        id={`soul-${agentId}`}
        value={text}
        spellCheck={false}
        onChange={(e) => {
          setDraft(e.target.value);
          if (state.kind !== "idle") setState({ kind: "idle" });
        }}
        rows={18}
        className="border-glass-border bg-glass text-ink focus-visible:ring-info/50 mt-3 w-full
                   resize-y rounded-xl border p-3 font-mono text-[12.5px] leading-relaxed
                   outline-none focus-visible:ring-2"
        placeholder="You are…"
      />

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={save}
            disabled={!dirty || tooLong || state.kind === "saving"}
            className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5
                       text-[12.5px] font-medium transition-opacity disabled:opacity-40"
          >
            {state.kind === "saving" ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Save className="size-3.5" />
            )}
            {state.kind === "saving" ? "Saving…" : "Save and apply"}
          </button>
          {dirty ? (
            <button
              type="button"
              onClick={() => {
                setDraft(null);
                setState({ kind: "idle" });
              }}
              className="text-ink-faint hover:text-ink inline-flex items-center gap-1.5 text-[12.5px]"
            >
              <RotateCcw className="size-3.5" /> Discard
            </button>
          ) : null}
        </div>

        <span className={`text-[11.5px] ${tooLong ? "text-blocked" : "text-ink-faint"}`}>
          {text.length.toLocaleString()} / {MAX_CHARS.toLocaleString()} characters
        </span>
      </div>

      {tooLong ? (
        <p className="text-blocked mt-2 text-[12px]">
          Too long to save. This text is prepended to every turn, so its length is a cost on
          every request.
        </p>
      ) : null}

      {state.kind === "error" ? (
        <p className="text-blocked mt-3 flex items-start gap-1.5 text-[12.5px]">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
          <span>{state.message}</span>
        </p>
      ) : null}

      {state.kind === "saved" ? (
        <div className="border-glass-border mt-3 space-y-1 rounded-lg border p-3 text-[12.5px]">
          <p className="text-running flex items-center gap-1.5">
            <Check className="size-3.5" /> Saved to {state.files.join(", ") || "the bundle"}
          </p>
          {state.applied ? (
            <p className="text-running flex items-center gap-1.5">
              <Check className="size-3.5" /> Applied — the agent is using it now
            </p>
          ) : (
            <p className="text-waiting flex items-start gap-1.5">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>
                Saved, but not applied: {state.detail}. The declaration is stored; the running
                agent still has the previous one.
              </span>
            </p>
          )}
        </div>
      ) : null}
    </GlassPanel>
  );
}

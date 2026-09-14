/* Duplicate, archive, restore, delete.
 *
 * Every one calls the backend route that already enforces the role and writes the audit
 * pair; none of them changes anything in the browser first. The frontend's whole job here
 * is to ask clearly and report truthfully.
 *
 * Archive and delete are deliberately not the same control. Archiving flips `enabled` and
 * is reversible from this screen. Deleting removes the declaration and cannot be undone
 * from here — so it asks for the agent's id to be typed, which is the one confirmation
 * that cannot be clicked through by accident.
 */

import * as React from "react";
import { AlertTriangle, Archive, ArchiveRestore, Copy, Loader2, Trash2 } from "lucide-react";

import { GlassPanel, SectionHeader } from "@/components/glass";
import { TextInput } from "@/components/form";
import { post } from "@/lib/api";
import { SaveResult, type SaveState } from "@/lib/editing";

type Props = {
  agentId: string;
  displayName: string;
  enabled: boolean;
  onChanged: () => void;
  onDeleted: () => void;
};

export function AgentLifecycle({ agentId, displayName, enabled, onChanged, onDeleted }: Props) {
  const [state, setState] = React.useState<SaveState>({ kind: "idle" });
  const [pending, setPending] = React.useState<string | null>(null);

  const [dupOpen, setDupOpen] = React.useState(false);
  const [newId, setNewId] = React.useState("");
  const [newName, setNewName] = React.useState("");

  const [confirmId, setConfirmId] = React.useState("");
  const [deleteOpen, setDeleteOpen] = React.useState(false);

  async function run(action: string, body: Record<string, unknown>, after?: () => void) {
    if (pending) return; // one at a time: two lifecycle calls racing is never what was meant
    setPending(action);
    setState({ kind: "saving" });
    try {
      const result: any = await post(`/agents/${encodeURIComponent(agentId)}/${action}`, body);
      const runtime = result?.runtime;
      const applied = runtime === undefined ? true : Boolean(runtime.applied);
      setState({
        kind: "saved", applied,
        detail: applied ? "" : String(runtime?.error ?? "the runtime was not updated"),
        files: result?.files_changed ?? [],
      });
      after?.();
    } catch (cause) {
      setState({
        kind: "error",
        message: cause instanceof Error ? cause.message : "the request did not complete",
      });
    } finally {
      setPending(null);
    }
  }

  const busy = (action: string) => pending === action;

  return (
    <GlassPanel className="p-5">
      <SectionHeader title="Lifecycle" detail="What can be done to this agent's declaration." />

      <div className="space-y-4">
        {/* Duplicate */}
        <div className="border-glass-border rounded-lg border p-3.5">
          <div className="flex flex-wrap items-center gap-3">
            <div className="min-w-0 flex-1">
              <p className="text-ink text-[13px] font-medium">Duplicate</p>
              <p className="text-ink-faint mt-0.5 text-[11.5px] leading-relaxed">
                Copies this agent's declaration and persona under a new id. The copy starts
                disabled — an exact clone joining the workforce unannounced, on the same
                channels with the same permissions, is not what duplicating should mean.
              </p>
            </div>
            <button
              type="button" onClick={() => setDupOpen((o) => !o)}
              className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium"
            >
              <Copy className="size-3.5" /> Duplicate
            </button>
          </div>
          {dupOpen ? (
            <div className="mt-3.5 grid gap-3 sm:grid-cols-2">
              <TextInput
                id={`dup-id-${agentId}`} label="New agent id" value={newId} mono
                onChange={setNewId} placeholder="night-ops"
                hint="Lowercase letters, digits and hyphens."
              />
              <TextInput
                id={`dup-name-${agentId}`} label="New name" value={newName}
                onChange={setNewName} placeholder={`${displayName} (copy)`}
                hint="Optional — defaults to the original plus (copy)."
              />
              <div className="sm:col-span-2">
                <button
                  type="button"
                  disabled={!newId.trim() || busy("duplicate")}
                  onClick={() =>
                    run("duplicate", { new_id: newId.trim(), name: newName.trim() }, () => {
                      setDupOpen(false); setNewId(""); setNewName(""); onChanged();
                    })
                  }
                  className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium disabled:opacity-40"
                >
                  {busy("duplicate") ? <Loader2 className="size-3.5 animate-spin" /> : <Copy className="size-3.5" />}
                  {busy("duplicate") ? "Duplicating…" : "Create the copy"}
                </button>
              </div>
            </div>
          ) : null}
        </div>

        {/* Archive / restore */}
        <div className="border-glass-border flex flex-wrap items-center gap-3 rounded-lg border p-3.5">
          <div className="min-w-0 flex-1">
            <p className="text-ink text-[13px] font-medium">{enabled ? "Archive" : "Restore"}</p>
            <p className="text-ink-faint mt-0.5 text-[11.5px] leading-relaxed">
              {enabled
                ? "Stops this agent being scheduled or routed to. Its declaration, persona and history stay, and this is reversible."
                : "This agent is archived. Restoring makes it schedulable and routable again."}
            </p>
          </div>
          <button
            type="button"
            disabled={busy("archive") || busy("restore")}
            onClick={() => run(enabled ? "archive" : "restore", {}, onChanged)}
            className="glass-solid text-ink inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-medium disabled:opacity-40"
          >
            {busy("archive") || busy("restore") ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : enabled ? (
              <Archive className="size-3.5" />
            ) : (
              <ArchiveRestore className="size-3.5" />
            )}
            {busy("archive") ? "Archiving…" : busy("restore") ? "Restoring…" : enabled ? "Archive" : "Restore"}
          </button>
        </div>

        {/* Delete */}
        <div className="border-blocked/30 rounded-lg border p-3.5">
          <div className="flex flex-wrap items-center gap-3">
            <div className="min-w-0 flex-1">
              <p className="text-blocked text-[13px] font-medium">Delete</p>
              <p className="text-ink-faint mt-0.5 text-[11.5px] leading-relaxed">
                Removes this agent's declaration and its persona file from the bundle. It
                does <b>not</b> delete the runtime profile — conversation history, memories
                and the agent's <span className="font-mono">.env</span> are left alone,
                because retiring an agent and destroying its record are different acts.
                This cannot be undone from here.
              </p>
            </div>
            <button
              type="button" onClick={() => setDeleteOpen((o) => !o)}
              className="border-blocked/40 text-blocked inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-[12.5px] font-medium"
            >
              <Trash2 className="size-3.5" /> Delete…
            </button>
          </div>
          {deleteOpen ? (
            <div className="mt-3.5 space-y-3">
              <TextInput
                id={`confirm-${agentId}`} label={`Type ${agentId} to confirm`} mono
                value={confirmId} onChange={setConfirmId} placeholder={agentId}
                hint="Typed rather than clicked, so this cannot happen by accident."
              />
              <button
                type="button"
                disabled={confirmId.trim() !== agentId || busy("delete")}
                onClick={() => run("delete", {}, onDeleted)}
                className="border-blocked/40 bg-blocked/10 text-blocked inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-[12.5px] font-medium disabled:opacity-40"
              >
                {busy("delete") ? <Loader2 className="size-3.5 animate-spin" /> : <AlertTriangle className="size-3.5" />}
                {busy("delete") ? "Deleting…" : `Delete ${agentId}`}
              </button>
            </div>
          ) : null}
        </div>
      </div>

      <SaveResult state={state} />
    </GlassPanel>
  );
}

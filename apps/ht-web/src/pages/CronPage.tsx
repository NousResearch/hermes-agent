import { useState } from "react";
import {
  createCronJob,
  deleteCronJob,
  getCronDeliveryTargets,
  getCronJobs,
  pauseCronJob,
  resumeCronJob,
  triggerCronJob,
  type CronDeliveryTarget,
  type CronJob,
} from "@/api/cron";
import { ManagementPage, ResourceView, useResource } from "@/components/PageScaffold";
import { ApiError } from "@/api/client";

interface CronData {
  jobs: CronJob[];
  targets: CronDeliveryTarget[];
}

/** Human-readable schedule from whichever field the backend populated. */
function scheduleText(job: CronJob): string {
  return (
    job.schedule_display ||
    job.schedule?.display ||
    job.schedule?.expr ||
    job.schedule?.run_at ||
    "—"
  );
}

/** The profile a job lives under — actions must target it, not always "default". */
function jobProfile(job: CronJob): string {
  return job.profile || job.profile_name || "default";
}

export default function CronPage() {
  const cron = useResource<CronData>(async () => {
    const [jobs, targets] = await Promise.all([getCronJobs(), getCronDeliveryTargets()]);
    return { jobs, targets: targets.targets };
  });

  return (
    <ManagementPage
      title="Cron"
      actions={
        <button type="button" className="ht-btn ht-btn--sm" onClick={cron.reload}>
          Refresh
        </button>
      }
    >
      <ResourceView resource={cron}>
        {(d) => (
          <>
            <CreateCronForm targets={d.targets} onCreated={cron.reload} />
            <div className="ht-card">
              <div className="ht-card__title">Jobs</div>
              {d.jobs.length === 0 ? (
                <p className="ht-muted">No cron jobs.</p>
              ) : (
                <table className="ht-table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Schedule</th>
                      <th>Next run</th>
                      <th>State</th>
                      <th>Deliver</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {d.jobs.map((job) => (
                      <CronRow key={job.id} job={job} onChanged={cron.reload} />
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </>
        )}
      </ResourceView>
    </ManagementPage>
  );
}

function CreateCronForm({
  targets,
  onCreated,
}: {
  targets: CronDeliveryTarget[];
  onCreated: () => void;
}) {
  const [name, setName] = useState("");
  const [schedule, setSchedule] = useState("");
  const [prompt, setPrompt] = useState("");
  const [deliver, setDeliver] = useState("local");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<{ kind: "ok" | "err"; msg: string } | null>(null);

  const valid = name.trim() && schedule.trim() && prompt.trim();

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!valid) return;
    setBusy(true);
    setNote(null);
    try {
      await createCronJob({
        name: name.trim(),
        schedule: schedule.trim(),
        prompt: prompt.trim(),
        deliver: deliver.trim() || "local",
      });
      setName("");
      setSchedule("");
      setPrompt("");
      setDeliver("local");
      setNote({ kind: "ok", msg: "Job created." });
      onCreated();
    } catch (err) {
      setNote({ kind: "err", msg: err instanceof ApiError ? err.message : String(err) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="ht-card" onSubmit={submit}>
      <div className="ht-card__title">Create job</div>
      <div className="ht-kv">
        <label htmlFor="cron-name" className="ht-sm">
          Name
        </label>
        <input
          id="cron-name"
          className="ht-input ht-input--sm"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Daily digest"
        />
        <label htmlFor="cron-schedule" className="ht-sm">
          Schedule
        </label>
        <input
          id="cron-schedule"
          className="ht-input ht-input--sm"
          value={schedule}
          onChange={(e) => setSchedule(e.target.value)}
          placeholder="0 9 * * *  (cron expression)"
        />
        <label htmlFor="cron-deliver" className="ht-sm">
          Deliver to
        </label>
        <select
          id="cron-deliver"
          className="ht-select"
          value={deliver}
          onChange={(e) => setDeliver(e.target.value)}
        >
          {targets.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
              {t.id !== "local" && !t.home_target_set ? " (no home channel)" : ""}
            </option>
          ))}
        </select>
        <label htmlFor="cron-prompt" className="ht-sm">
          Prompt
        </label>
        <textarea
          id="cron-prompt"
          className="ht-code-editor"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="What the agent should do on each run…"
          aria-label="Job prompt"
        />
      </div>
      <div className="ht-editor__bar">
        <button type="submit" className="ht-btn" disabled={busy || !valid}>
          {busy ? "Creating…" : "Create job"}
        </button>
        {note && (
          <span className={note.kind === "ok" ? "ht-ok" : "ht-error-inline"}>{note.msg}</span>
        )}
      </div>
    </form>
  );
}

function CronRow({ job, onChanged }: { job: CronJob; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const profile = jobProfile(job);

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setErr(null);
    try {
      await fn();
      onChanged();
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const remove = () => {
    if (!window.confirm(`Delete cron job "${job.name || job.id}"?`)) return;
    void run(() => deleteCronJob(job.id, profile));
  };

  return (
    <tr>
      <td>
        <div>{job.name || <span className="ht-dim">Untitled</span>}</div>
        {err && <div className="ht-error-inline ht-sm">{err}</div>}
      </td>
      <td className="ht-sm">{scheduleText(job)}</td>
      <td className="ht-sm">{job.next_run_at || <span className="ht-dim">—</span>}</td>
      <td>
        {job.enabled ? (
          <span className="ht-chip ht-chip--ok">enabled</span>
        ) : (
          <span className="ht-chip ht-chip--warn">paused</span>
        )}
      </td>
      <td>
        <span className="ht-chip">{job.deliver || "local"}</span>
      </td>
      <td className="ht-row-actions">
        {job.enabled ? (
          <button
            type="button"
            className="ht-btn ht-btn--sm ht-btn--ghost"
            disabled={busy}
            onClick={() => run(() => pauseCronJob(job.id, profile))}
          >
            Pause
          </button>
        ) : (
          <button
            type="button"
            className="ht-btn ht-btn--sm"
            disabled={busy}
            onClick={() => run(() => resumeCronJob(job.id, profile))}
          >
            Resume
          </button>
        )}
        <button
          type="button"
          className="ht-btn ht-btn--sm"
          disabled={busy}
          onClick={() => run(() => triggerCronJob(job.id, profile))}
        >
          Trigger
        </button>
        <button
          type="button"
          className="ht-btn ht-btn--sm ht-btn--stop"
          disabled={busy}
          onClick={remove}
        >
          Delete
        </button>
      </td>
    </tr>
  );
}

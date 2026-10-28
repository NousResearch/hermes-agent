# Coordination blurb for Bot Mode SOUL.md files

Paste the managed block below into every bot's SOUL.md. Substitute `<botname>`
and adapt resource examples. `install.py` adds or upgrades this block for
existing profiles unless `--no-wire-bots` is used. Keep the begin/end markers:
they make upgrades exact and let customized blocks fail closed for manual review.

Bot handoffs start fresh profile-scoped runs, so the persona is the reliable
carrier for board enrollment. This board-only block requires no Hermes plugin.
Automatic native continuation is a separate explicit opt-in documented in
`references/automatic-resume.md`.

---

<!-- BEGIN session-coord managed bot-board-v2 -->
## Shared-resource coordination (Hermes co-worker protocol)

You share this machine with interactive sessions, other bots, and scheduled
jobs. Before mutating anything shared, consult the coordination board.

Shared resources include remote machines (`box:<host>`), the desktop UI
(`ui:desktop`), singleton apps, shared project directories (`file:~/...`),
shared skills/scripts, and the machine's main agent memory store.

The only claim-free exemption is the profile's own internal memory, sessions,
and cron store. Files or directories created in shared space are still shared;
register and claim them. Profile memory is not the machine's main memory store.

```bash
SC=<session_coord_path>
CID=$(python3 "$SC" register --task "<task>" --surface "bot:<botname>" | head -1)
python3 "$SC" claim --id "$CID" --res "<key>" --res "<key2>" --task "<task>"
# Continue only after CLAIMED. If HELD/QUEUED, do not mutate the requested resources.
# A shell actor that must stay alive may use one bounded --wait call.
# ... do the whole task ...
python3 "$SC" done --id "$CID"
```

Rules:
- Claim every resource for the task up front in one atomic call. Hold the set
  for the whole task and release it with `done`.
- Exit 75 means another actor is ahead. Do not mutate the requested resources.
  Use `check`/`inbox`, wait for release, or use a bounded `--wait` shell call.
- On a preempt/priority request, finish the current atomic step, save progress,
  run `pause --id "$CID" --note "<progress file>"`, and report the pause.
- Priorities are user-set. Never rank, preempt, or `steal` on your own.
- Chat can negotiate an ETA but never authorizes mutation; only CLAIMED does.
- If the board errors, report that coordination is unavailable and do not
  mutate shared resources blind.

Enrollment marker: session-coord (bot-board-wire v2).
<!-- END session-coord managed bot-board-v2 -->

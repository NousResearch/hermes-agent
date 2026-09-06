# Approval decision audit log

Enable the local JSONL sink in `config.yaml`:

```yaml
approvals:
  audit_log:
    enabled: true
```

The default is **false**. Existing config loads publish this setting; the observer
never reads config or accesses files when disabled. The profile-aware destination
is `logs/approvals.jsonl` under `get_hermes_home()` (normally
`~/.hermes/logs/approvals.jsonl`). There is no network exporter.

Responses on the command/code approval, protected-instruction-file approval, MCP
elicitation, smart approval, and gateway wait paths are sent to the sink. Selected
plugin transports are recorded too, including request-redaction failures and
unavailable transports. Each successful append is synchronous, flushed and fsynced.
A smart denial followed by an owner response produces two records. Gateway timeout and notification failure are also
recorded. Smart escalation is not a decision; its eventual human response is logged.
Existing bypasses that do not request approval, such as YOLO and cached allowlists,
do not emit approval-response hooks and are outside this sink's coverage.

Records contain a UTC timestamp, session/turn/tool-call correlation IDs, source,
verdict, `decided_by` (`user`, `aux_llm`, `gateway`, or `timeout`) and `interactive`.
`raw_choice` retains the known response code (including transport failure codes);
unknown codes are hashed. `scope` retains `once`, `session`, or `always` for human
grants, and is null for denials/timeouts and smart decisions. The two per-call CLI
confirmations record scope `once` even if a legacy callback returns a wider grant.
`pattern_keys` retains the complete ordered list as digests, so a persisted grant
can be matched to later pattern keys even though allowlist bypasses aren't logged.
Transport sources retain `transport:<name>`; their actor and interactive flag use
the original CLI/gateway surface. Other specialized gateway surfaces retain their
surface name. Transport timeout maps to `timeout`; other transport failures map
to `denied`, with the failure preserved in `raw_choice`.

If no Hermes `session_id` is available, the routing `session_key` is hashed; raw
platform/user identifiers from that fallback are never written.
Commands, descriptions and pattern keys are stored only as SHA-256 digests,
including any diffs embedded in command text. Digests describe the hook payload,
which may already be redacted by the approval path. `tool` and `target` are null:
the current hooks do not supply reliable values, and the sink does not infer them
from shell commands. Free-form reasons and arbitrary extra hook fields are omitted.

Content digests are **unsalted**, not encryption: someone with the log can guess
low-entropy values offline and compare their hashes. Hashing removes plaintext;
it does not make short secrets or predictable platform identifiers unrecoverable.
Keep log access restricted. No salt/key file or salt generation is introduced.

Rotation uses `logging.max_size_mb` and `logging.backup_count` (defaults: 5 MiB,
3 backups), with `.1` the newest backup. Writers share a separate `.jsonl.lock`
file and close data handles before renaming, including on Windows. A record larger
than the limit occupies its own file. Retention eventually removes old records.
Directory metadata is fsynced after rotation renames on POSIX; Windows has no
portable equivalent through Python's directory file descriptors.

`prev_hash` links each record to its predecessor across rotations and restarts;
the first record uses 64 zeroes. To verify a record, remove `hash`, serialize with
Python `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True)`,
encode as UTF-8 and compare its SHA-256 hex digest to `hash`. Verify adjacent
`prev_hash` links from the oldest retained backup to the active file. The oldest
retained predecessor may have expired under retention.

The writer validates the existing tail before appending. An incomplete or corrupt
tail causes a content-free warning and prevents that append; it is not silently
repaired. Sink failures never change approval outcomes. Thread-lock and process-lock
waits share a three-second budget; contention beyond that budget skips the receipt and
logs a content-free warning. This bounds lock contention, not arbitrary filesystem
latency. An append interrupted after rotation resumes from the newest backup on
the next write; the failed decision itself is not replayed. Preserve damaged files
for inspection before moving them aside. Hash chains reveal modification relative to
a trusted digest; they do not prevent someone with write access from rewriting
an entire chain or deleting its tail. This sink provides no external trust anchor.

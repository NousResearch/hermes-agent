# Security boundary

Owner-only directories/files, same-UID socket checks, immutable conversation
identity, OMP session identity, epoch/sequence validation, bounded frames and
durable cursor advancement prevent accidental cross-session delivery and unsafe
replay. They do not sandbox same-UID code, which can already modify the process,
its files or environment.

Only the native Hermes terminal supervisor owns asynchronous delivery. Child
environment values are inherited identity, not credentials or an authenticated
cross-host protocol. The CLI never sends messages directly to a chat platform.
The calling workflow must verify that native background notification was accepted.

The journal contains normalized event kinds, run/session identifiers, sequence
numbers and lifecycle state. Never store raw prompts, model messages, tool
arguments/results, credentials or arbitrary error text. State remains private
metadata and must not enter public issues, commits or reports without review.

A monitoring failure closes observation, not the worker. No protocol message
approves tools, steers, cancels, restarts or changes a task. Use normal authorized
OMP controls separately. Session changes revoke enrollment. Cursor/epoch
mismatches and retention gaps require inspection; never replace owner IDs or
clear launch intent to retry.

A tools-disabled canary is not a sandbox and still contacts the configured model.
Do not expose the Unix socket over a network or enroll untrusted directories.
Use the repository's private security-reporting process for vulnerabilities;
do not attach journals, credentials or transcripts to a public issue.

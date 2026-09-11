# Privacy for this POC

Do not put live session data in this skill, PRs, or logs.

Strip before any public commit:

- Real profile paths, cookie databases, and session stores
- Residential / VPN / mesh IPs and internal hostnames
- Real social-graph names, group IDs, post IDs, and comment text
- Owner display names from "Comment as …" chrome

Allowed examples: `example.com`, `ws://127.0.0.1:9377/camoufox`, fictional names in `executive-highlights.md`.

Do not copy cookies from a Firefox profile into a temporary page. Inspect the hold tab over the unix socket instead.

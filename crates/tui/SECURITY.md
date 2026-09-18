# Security

This crate is **experimental**. It is a local terminal client. It draws a TUI and talks to a Python `tui_gateway` child over newline JSON-RPC stdio. It does not take inbound network connections and it does not call model APIs.

## Report

Use GitHub private vulnerability reporting on the parent repository:

https://github.com/NousResearch/hermes-agent/security/advisories/new

Do not open a public issue for a secret, an RCE, or a way to escape the worktree jail.

## Scope

In scope: the Rust binary in `crates/tui/`, launch env handling, RPC framing, log redaction, worktree path confinement.

Hermes Agent (`tui_gateway`, tools, skills) is the same repo; still use private reporting, not a public issue.

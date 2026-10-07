# Architecture decision records

This directory contains Architecture Decision Records (ADRs) for durable Hermes
architecture choices that are narrower than a full design specification but
important enough to preserve for future contributors.

ADRs should describe the context, accepted decision, alternatives considered, and
implementation consequences. They are historical records: prefer adding a new ADR
that supersedes an old one over rewriting an accepted decision in place.

## Records

- [ADR 0001: Keep Kanban state row-oriented with an append-only event log](0001-kanban-event-log-not-event-sourced.md)

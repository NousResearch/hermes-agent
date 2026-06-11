# AlphaHunt Pipeline vs Hermes Research

Hermes currently has an AlphaHunt event decision pipeline in
`gateway/platforms/alphahunt_stage.py`. That file is a per-event analysis stage
handler for cleaner, screener, sentinel, packager, and fast-triage decisions.

The new `gateway.alphahunt.research_yaml` module is different. It is a generic
Hermes research output generator and validator for proactive project research.

Do not mix the two surfaces:

- `alphahunt_stage.py` handles event decision objects.
- `gateway.alphahunt.research_yaml` handles project research notes.
- An event decision object is not a project research note.
- A project research YAML envelope is not a per-event decision packet.

The research YAML is intended for AlphaHunt P08:

```bash
python3 /home/ubuntu/alphahunt/scripts/create_project_from_research.py docs/alphahunt/samples/protocol_ethena.yaml
```

This handoff remains dry-run/operator-controlled unless a user separately
authorizes a live workflow. Hermes must not write AlphaHunt production DBs,
call AlphaHunt live APIs, place trades, place wagers, sign transactions, manage
funds, or send notifications as part of this research output path.

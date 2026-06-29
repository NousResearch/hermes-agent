# hermes-agent BMad Planning Handoff

This folder is the local planning input package for isolated hermes-agent implementation.
The implementation environment may not be able to read the parent workspace.
Parent planning must materialize hermes-agent-owned implementation inputs here before hermes-agent BMad implementation starts.

Expected files:

```text
prd.md
architecture.md
epics.md
```

Do not place implementation artifacts here.
hermes-agent implementation artifacts are generated locally under:

```text
hermes-agent/_bmad-output/implementation-artifacts/
```

Run implementation workflows from inside `hermes-agent/`.
Do not rely on parent workspace paths during implementation.

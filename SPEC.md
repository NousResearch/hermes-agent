# Hermes Operator Grammar — Specification

**Version:** 1.1.0
**Status:** Stable
**Scope:** Conductor dispatch, viewport computer, private client runtime, omniverse surface mesh

## Abstract

This specification defines the operator URI grammar and scheme dispatcher used by
the Hermes conductor. It is designed to be transport-agnostic, provider-neutral,
UI-surface-agnostic, and omniverse-deployable via mesh addressing.

## 1. Terminology

- **private client:** bounded, sovereign, local-first autonomous execution surface
- **conductor:** intent router / scheme dispatcher above all surfaces
- **viewport computer:** global HTML shell runtime for Hermes agents
- **bridge:** backend daemon exposing private client capabilities over HTTP
- **governance layer:** issue/route intent tokens across agents
- **omniverse mesh:** QR-addressable `/conductor` surfaces across local-first viewports

## 2. URI Grammar

```
scheme     = "c://" / "H://" / "hermes://" / "pc://" / ...
path       = ...
```

### 2.1 Scheme Registry

| Scheme                | Canonical Form                   | Dispatcher Priority |
|-----------------------|----------------------------------|---------------------|
| `c://cc <target>`     | change context                   | primary             |
| `H:// <tail>`         | global agentic domain            | primary             |
| `hermes://`           | default runtime                  | fallback            |
| `pc://`               | private client                   | primary             |
| `pc://run <name>`     | private client run               | primary             |
| `mcp://`              | MCP tools surface                | primary             |
| `vscode://`           | viewport host (VS Code = runtime surface for the local HTML/CSS/WASM viewport) | primary             |
| `reachy://`           | robot/operator surface           | primary             |
| `NOUS://`             | provider/runtime                 | primary             |
| `llc://`              | business surface                 | primary             |
| `daollc://`           | governance/identity              | primary             |
| `intent://`           | intent/permission token          | primary             |
| `+æ://`               | DAO/intent token                 | primary             |
| `+æ://media^ffmpeg`   | media pipeline primitive         | primary             |

Longest-prefix wins. Overlapping prefixes are resolved by depth, not registration order.

## 3. Conductor Contract

Every `run_hermes(payload)` call returns:

```
{
  "ok": boolean,
  "rc": integer,
  "stdout": string,
  "stderr": string,
  "surface": object,
  "scheme": string   // absent on cli-verb passthrough
}
```

## 4. Dispatching Rules

1. If input matches a registered scheme prefix → scheme dispatch
2. If `"run pc://<name>"` → normalize to `pc://run <name>`
3. If `"H://cc <tail>"` / `"hermes://cc <tail>"` → normalize to `c://cc <tail>`
4. Else → passthrough to `hermes_cli` CLI verb subprocess

## 5. Extensibility

```python
dispatcher = SchemeDispatcher()
dispatcher.register("custom://", handler)
```

`handler(raw: str) -> dict` must return the conductor contract shape.

## 6. Security Considerations

- URI surfaces are local dispatchers unless explicitly bridged
- No scheme handler shells to OS without explicit subprocess contract
- `pc://run` accepts bounded client names; empty defaults to `default`
- HTML viewport is a control surface, not an auth boundary

## 7. Non-Goals

- This spec does not define UI rendering, transport serialization,
  or provider load-balancing. Those are separate tiers.

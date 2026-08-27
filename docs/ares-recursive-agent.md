# Ares and Recursive Agent

## Scope

This guide documents the boundary between the **Ares** downstream distribution of Hermes Agent and the separately maintained **Recursive Agent** service. It is an operator guide, not a claim that the service is bundled, automatically configured, or safe to expose on a network.

## Ownership

| Component | Canonical owner | What it does not own |
|---|---|---|
| Ares | this `RecursiveIntell/hermes-agent` fork | Recursive Agent run state, receipts, or daemon policy |
| Hermes plugin loader | Hermes-compatible runtime | evidence generation or daemon lifecycle |
| `recursive-agent-native` plugin | `RecursiveIntell/recursive-agent` integration package | Hermes core behavior and configuration |
| Recursive Agent daemon | `RecursiveIntell/recursive-agent` | provider secrets, operator approval, or Ares persistence |

The plugin makes one bounded native request. It does not expose a general remote-command transport, a direct evidence writer, or an MCP substitute.

## Preconditions

Before installing the plugin, independently establish all of these conditions:

1. You have a local checkout of `RecursiveIntell/recursive-agent`.
2. Its own build, policy, and daemon lifecycle gates have passed.
3. The daemon is running on its private local Unix-domain socket.
4. You have reviewed the plugin source and accept agent-process plugin authority.
5. You are using the intended Ares/Hermes home and have no existing plugin directory at `plugins/recursive-agent-native` unless you deliberately removed it.

The Ares bootstrap can install the plugin payload but does not satisfy conditions 2 or 3.

## Install the plugin

```bash
cd /path/to/ares
bash install.sh --with-recursive-agent-source /path/to/recursive-agent
```

The upstream integration installer copies exactly these plugin runtime files:

```text
__init__.py
client.py
schema.py
plugin.yaml
pyproject.toml
```

It writes a manifest alongside the plugin for deterministic removal. The plugin is discovered at next Ares/Hermes process start; an already-open conversation may have an older tool schema.

## What a successful invocation means

A locally exercised plugin call returns daemon-derived facts such as the terminal state, run directory, receipt-chain length, final chain hash, and verification result. This demonstrates a narrow execution path:

```text
Ares/Hermes session
  -> native plugin registration
  -> canonical bounded envelope
  -> authenticated local IPC
  -> Recursive Agent daemon
  -> terminal state + receipt-chain verification facts
```

It does **not** establish a claim about any unrelated repository, remote machine, customer workload, security certification, or future run. Every run has to be verified on its own evidence.

## Verify by layer

Use distinct terms when reporting readiness:

| State | Meaning | Example evidence |
|---|---|---|
| Selected | Ares/Hermes profile enables the plugin/toolset | typed profile configuration |
| Registered | A fresh plugin loader found the manifest and registration hook | fresh loader smoke |
| Exposed | the tool is in the active session schema | new-session tool inventory |
| Exercised | one bounded request reached the daemon and returned verified terminal facts | daemon response plus receipt verification |

Do not collapse these into “installed.” A copied plugin with no running daemon is installed but not exercised.

## Rollback

Remove the plugin using the installer in the Recursive Agent checkout:

```bash
/path/to/recursive-agent/scripts/uninstall-hermes-plugin.sh
```

Then start a fresh Ares/Hermes session. This removes the plugin payload; it does not alter daemon data, receipts, or any separately configured services.

## Security boundary

The Unix socket limits the transport’s reach; it is not containment. The plugin runs in the agent process and inherits that process’s authority. Keep the daemon local, use restrictive filesystem permissions, and rely on OS-level or whole-process isolation for hostile-input workloads.

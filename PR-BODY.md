## What this plugin does

**GoneBy — retroactive project memory for LLM agents.** Capture is deliberately dumb and free: every turn appends to an encrypted (ChaCha20-Poly1305) per-project write-ahead log — zero tokens, zero judgment, disk I/O only. At project close, GoneBy walks that log **backward from the shipped product**, using the product as the oracle to *verify* which entries actually contributed (rather than predicting mid-project what might matter), embeds the survivors (Qwen3-Embedding-0.6B or any OpenAI-compatible endpoint; MRL truncation 1024→512), and upserts them into Valkey (default) or Qdrant for semantic search in later sessions. The backward-walk framing is not present in Mem0 / MemGPT / Cognee / Letta — that is the thesis. Project-scoped by default, cross-project recall opt-in, full per-project/per-gateway capture policy including a disable-by-default opt-in mode.

## Hermes surfaces used

- **Hooks:** `pre_llm_call` (per-turn capture, `message_uid` dedup, `system`/`tool` roles skipped; optional recall injection, off by default), `post_compaction` (boundary marker only).
- **Tools (7):** `wall_append`, `wall_read`, `walk_backward`, `store_memory`, `query_memory`, `goneby_status`, `goneby_project`.
- **Slash command:** `/goneby` (status / enable / disable / gate / mode / platform).
- **Host LLM:** `ctx.llm.complete_structured` for the walk judge — the user's active model and auth, nothing hardcoded. The judge fails open: an unparseable verdict keeps every entry.
- General plugin only: no memory-provider slot claimed (coexists with Mem0/Honcho/built-in), no middleware, no desktop surface, **no core overrides of any kind**.

## Rule-13 disclosure

- **Network:** HTTP only to **user-configured endpoints** — an OpenAI-compatible `/v1/embeddings` server (`GONEBY_EMBEDDING_BASE_URL`, default `http://localhost:8081/v1`) and the user's Valkey/Qdrant host. No third-party services, no telemetry, no update checks (rule 3: none, by design).
- **Reads outside plugin data:** one **read-only** SQLite query of Hermes' `state.db` (`SELECT cwd, git_repo_root FROM sessions WHERE id = ?`) to namespace the WAL per project; opened `mode=ro`, never writes, and any failure silently falls back to `GONEBY_PROJECT`. No other Hermes internals are touched.
- **Writes:** `~/.goneby/wal/<project>.wal` (per-entry ciphertext) and `~/.goneby/config.json` (capture-policy state: mode, gate, platform kills, per-project overrides). The plugin never writes keys; the operator supplies `GONEBY_ENCRYPTION_KEY`.
- **Background:** one daemon-thread event loop bridging sync hooks to the async embedder/store; hook bodies are bounded and never raise.
- **Secrets:** `GONEBY_ENCRYPTION_KEY` (declared in `requires_env`) read via `agent.secret_scope.get_secret`, failing closed when multiplexing has no bound scope — one profile's key can never decrypt another's WAL.
- **Dependency graph:** the plugin depends on the `goneby` library via a **relative `tool.uv.sources` path inside the same repository** — no registry, no network fetch at lock time; the admission rewrite points it at the clone (library pin == plugin pin). Verified in a clean-room container (no other GoneBy present): `uv lock` resolves 15 packages, sync installs, imports succeed.

## Validation (same command as the CI gate)

At the pinned commit, on Linux:

```
hermes plugins validate <clone>/plugin --install-deps
```

All 16 checks pass, including the **security scan (safe)**, capability probe (register() in isolation), declared-vs-actual tools/hooks/env match, `requires_hermes` floor `>=0.23.0` (developed and verified on 0.23.1), and **no core override**. Repo-side: 158 tests, ruff clean.

## Links

- Repo (public, Apache-2.0): https://github.com/rahlquist/goneby
- Pinned tree: https://github.com/rahlquist/goneby/tree/c7cc997ee7fde0629daf0ddf79cd65093b6e98b6
- Mechanism doc: https://github.com/rahlquist/goneby/blob/main/docs/OVERVIEW.md
- Idea/lineage doc: https://github.com/rahlquist/goneby/blob/main/docs/IDEATION.md
- 100 s narrated explainer: https://github.com/rahlquist/goneby/releases/download/v1.0.0/goneby-explainer.mp4

## Maintainer

rahlquist — repository owner (rule 5). Screenshots and banner are frames from our own explainer video, committed at the pinned SHA.

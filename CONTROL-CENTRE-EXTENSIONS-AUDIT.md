# Audit — S3 knowledge sources, MCP servers, and plugins in the Control Centre

What the runtime already has, what NOVA has, and what therefore needed building. Written
before the code, and corrected against the code afterwards. Every count below was produced
by running the named function, not by reading a README.

---

## 1. S3 for knowledge

### What existed

`nova/knowledge/sources.py` declares a corpus as a **directory** plus glob filters, a size
cap and a classification. `iter_documents()` is the security boundary — no symlinks, no
paths resolving outside the root, nothing over the cap. `nova/knowledge/store.py` (built in
the previous phase) adds `store_document` / `remove_document` for browser uploads.

Nothing read from object storage. `boto3==1.42.89` is already a declared extra
(`pyproject.toml`, the `bedrock` extra) and is installed in this checkout — so an S3 path
needs no new dependency decision, only an honest one about whether it is *required*.

### What was built

`nova/knowledge/origin.py`. A corpus may declare an `origin:` — today one kind, `s3` —
and NOVA mirrors the bucket **into the same root the local corpora use**. The ingest that
follows is the existing one, unchanged.

That choice is the whole design. A second ingestion path for remote documents would
eventually disagree with the first about chunking, provenance, or what counts as a
document, and the disagreement would show up as an agent citing something that is not there.

Three properties are enforced rather than assumed, because object keys come from a bucket
and a bucket is not necessarily one the operator controls end to end:

| Property | Where |
|---|---|
| A key that resolves outside the corpus root writes nothing | `_safe_target()` |
| An object the corpus's own globs do not accept is skipped **and named** | `accepts()`, reused from the upload path |
| An object over `max_file_bytes` is refused *before* it is downloaded | the `Size` check, from the listing |

**NOVA holds no credentials for this.** boto3 resolves them the way it always does — the
instance role on EC2, the environment elsewhere. This is the same posture as every other
integration and it is what keeps the standing constraint intact: no hard-coded account, no
permanent root credentials asked of the customer. The IAM permission needed is
`s3:ListBucket` + `s3:GetObject` on one prefix.

**A mirrored corpus is read-only from NOVA's side.** Upload and remove return `409` and say
why. This is not a limitation to work around: a document uploaded into a mirror survives
exactly until the next sync deletes it, and a control that quietly loses data later is worse
than one that refuses now.

`prune` defaults to **on**. A mirror that only ever added would keep answering from a
document the customer deleted, which is the worse of the two failures.

---

## 2. MCP

### What the runtime has — measured

```
hermes_cli.mcp_catalog.list_catalog()  ->  65 entries
  transport=http,  auth=oauth    54
  transport=http,  auth=none     10
  transport=stdio, auth=api_key   1
```

Each entry is a pinned manifest under `optional-mcps/`. Presence there *is* the approval
signal — the module's own docstring says there is no community tier.

Storage is `mcp_servers.<name>` in the profile's `config.yaml`, via
`hermes_cli.mcp_config._get_mcp_servers` / `_save_mcp_server`. `_save_mcp_server` refuses
entries that `hermes_cli.mcp_security.validate_mcp_server_entry` flags.

Registration: `tools/mcp_tool_registration.py:315` puts a server's tools into a toolset
named **`mcp-<name>`**. That matters more than anything else here — it means an MCP server
is not a new governance surface. It is a toolset, and NOVA already models toolsets.

### The problem that decided the design

`nova apply` **rewrites `<profile>/config.yaml` wholesale** (`materialize.py:583`). Writing
`mcp_servers` straight into that file from the Control Centre would work until the next
apply erased it — the identical trap as editing `SOUL.md` directly, which this project has
already refused once.

So MCP servers are **declared in the bundle** and compiled by `materialize`, exactly like
everything else NOVA governs. The Control Centre writes the bundle through
`nova/spec/writer.py` and then applies. Saved and applied stay two separate facts.

### What is honestly true about authorization

54 of 65 catalogue entries are OAuth. OAuth needs an interactive browser consent, once per
host, and the Control Centre cannot perform it — `hermes mcp login <name>` can. NOVA
therefore reports an OAuth server as **declared and enabled, not yet authorized**, and says
which command completes it. That is the same `declared → wired → enforced → live-proven`
ladder used everywhere else, and it is not something to paper over with a green tick.

The 10 no-auth entries work the moment they are applied. The one `api_key` entry
(`stdio`) takes its key through the existing allowlisted credential path — never the bundle,
never the browser.

### What is deliberately NOT built

**Arbitrary MCP servers.** The catalogue is the allowlist. A stdio server is a command line;
accepting one from a browser form would hand anyone with admin on the Control Centre remote
code execution on the NOVA host. The same reasoning as "do not invent channels", applied to
a much sharper edge.

---

## 3. Plugins

### What the runtime has — measured

```
hermes_cli.plugins_discovery.collect_directory_manifests()  ->  58 manifests
  kind=backend    source=bundled   30
  kind=platform   source=bundled   22
  kind=standalone source=bundled    6
```

`gate_manifest()` (`plugins_discovery.py:173`) decides what loads, in this order:

1. `plugins.disabled` — always wins.
2. `kind: exclusive` / `model-provider` — their own activation paths, not this one.
3. `source: bundled` + `kind: backend` → **auto-loads**; selection among them is
   `<category>.provider`, not an enable list.
4. `source: bundled` + `kind: platform` → **deferred**, loaded on first use. These are the
   22 channels the Control Centre already exposes.
5. Everything else → must appear in `plugins.enabled`.

So the per-agent control surface is smaller and more specific than "58 plugins":

| Group | Count | What the Control Centre can honestly do |
|---|---|---|
| standalone | 6 | Enable (adds to `plugins.enabled`) or leave off |
| bundled backend | 30 | **Disable** (adds to `plugins.disabled`); enabling is a no-op, they already load |
| bundled platform | 22 | Nothing here — these are channels, and have their own screen |

Presenting all 58 with an on/off switch would be the "beautiful mock dashboard" failure:
thirty of those switches would report a state change that changed nothing.

`materialize.plugins_section()` already writes `plugins.enabled` for NOVA's own two plugins
(policy, knowledge). The new grants merge into that block rather than replacing it — losing
the policy plugin's enable would silently disarm governance, which is the worst state a
control can be in because it passes review by inspection.

---

## 4. Governance — what actually enforces anything

MCP tools reach an agent as ordinary tools, so they are decided by the same thing every
other tool call is: the compiled policy, at `nova/policy/decide.py`.

- A tenant running `defaults.unlisted_tool: deny` denies MCP tools unless granted. Real.
- A tenant on the default `allow` permits them. Also real, and the screen says so rather
  than implying a gate that is not there.
- `tools.deny` compiles to the runtime's unconditional deny list and covers `mcp-*` names
  like any other.

Positive toolset scoping is still **not** compiled into `config.yaml` — the pre-existing
limitation documented at `materialize.py:357`. Nothing here changes that, and nothing here
claims otherwise.

---

## 5. Summary of what was added

| Area | Module | Kind |
|---|---|---|
| S3 mirror | `nova/knowledge/origin.py` | new capability |
| Sync route | `nova/control/api.py` `_sync_origin` | new route, admin-only |
| Extension inventory | `nova/runtime/hermes/extensions.py` | discovery, no invention |
| Extension model | `nova/extensions/` | pure model + injection point |
| Declaration | `nova/spec/agent.py` `ExtensionsSpec` | bundle schema |
| Compilation | `nova/runtime/hermes/materialize.py` | `mcp_servers`, merged `plugins` |
| Editing | `nova/extensions/manage.py` | bundle writes, via the existing writer |
| UI | `screens/agent-extensions.tsx`, `screens/corpus.tsx` | |

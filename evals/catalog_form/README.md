# Deferred-catalog form A/B/C

Does the model need one `name: description` line per deferred plugin/MCP tool inside the `tool_search`
description, or do names alone, or one line per server, do the job for fewer tokens?

Arms (same tree, same tools, only the listing embedded in the `tool_search` description differs):

| arm | `tools.tool_search.listing_servers` | plugin/MCP servers render as | deferred built-ins |
|---|---|---|---|
| `full` | `auto` (today) | one `- name: description` line per tool | full line |
| `names` | `names` | one comma-separated line of tool names per server | full line |
| `summary` | `summary` | `blender (30 tools — names not listed; discover via tool_search)` | full line |

The catalog is a real one (`catalog.txt`: 79 deferred tools; blender 30, meshy 28, elevenlabs_music 8,
nous-support 6, 7 single tools). The 73 plugin/MCP tools are registry stubs with the catalog's names and
descriptions, small hand-written parameter schemas and canned results (`tasks.py`); no Blender, Meshy or
ElevenLabs is needed. Terminal and file tools are real. The eager surface is terminal + file tools in every arm.

## Battery (`tasks.py`)

- 11 `deferred` tasks that need one specific deferred tool the prompt does not name, several per server,
  including the blender `_for_cli` look-alike twins in both directions.
- 1 `continuation` task that is under-specified; the harness answers the agent's question once.
- 2 `control` tasks that need only eager tools, 2 `distractor` tasks that are domain-adjacent but need no
  deferred tool (any bridge call there is false discovery).

Per cell the worker records: task done (oracle = the stub was really invoked with valid arguments AND the planted
fact is in the answer), right tool reached, wrong-tool calls (deferred tools executed that are neither the target
nor a harmless read-only lookup), the model call on which the right tool was first invoked, bridge calls, and
session usage split into fresh input, cache read, cache write and output, plus dollars.

## Run

```bash
export CF_TREE=<this checkout> CF_PYTHON=<hermes env python> PYTHONPATH=$CF_TREE
CF_DUMP_PREFIX=1 $CF_PYTHON worker.py full <model> core_sfx 0 /path/dump.json   # prefix + budget-ladder dump, no model call
$CF_PYTHON run.py <model> 1 --only=bl_cli_datablocks                           # smoke one cell per arm first
$CF_PYTHON run.py <model> 3 --parallel=6
$CF_PYTHON run.py --report
```

Inference defaults to the Nous Portal agent key, read fresh from `$HERMES_HOME/auth.json` per cell and passed as a
`custom` provider. Each cell gets a temp `HERMES_HOME`, `isolate_host()` from `../core_tool_deferral/sandbox.py`,
and a PATH with any real `blender` removed.

## Reading the result

- Every arm runs the same tasks and reps; errors, timeouts and noise cells score 0 and stay in the denominator.
- Cells of one arm share a byte-identical prefix, so after the first cell the prefix is billed as a cache read.
  Compare arms on tokens per cell and model calls; dollars under-weight the prefix relative to a cold session.
- Stub descriptions are the catalog's clipped first sentences, so `tool_search` hits carry no more text than the
  `full` listing does. With real servers a search hit returns up to 500 characters per tool.

# graphify — Subcommand Reference

Full implementation for all subcommands. Before running any of these, check that `graphify-out/.graphify_python` exists. If missing, re-resolve using the detection block in [pipeline.md](pipeline.md) Step 1.

```bash
if [ ! -f graphify-out/.graphify_python ]; then
    PYTHON=""
    GRAPHIFY_BIN=$(which graphify 2>/dev/null)
    if [ -z "$PYTHON" ] && command -v uv >/dev/null 2>&1; then
        _UV_PY=$(uv tool run graphifyy python -c "import sys; print(sys.executable)" 2>/dev/null)
        if [ -n "$_UV_PY" ]; then PYTHON="$_UV_PY"; fi
    fi
    if [ -z "$PYTHON" ] && [ -n "$GRAPHIFY_BIN" ]; then
        _SHEBANG=$(head -1 "$GRAPHIFY_BIN" | tr -d '#!')
        case "$_SHEBANG" in *[!a-zA-Z0-9/_.-]*) ;; *) "$_SHEBANG" -c "import graphify" 2>/dev/null && PYTHON="$_SHEBANG" ;; esac
    fi
    if [ -z "$PYTHON" ]; then PYTHON="python3"; fi
    mkdir -p graphify-out
    "$PYTHON" -c "import sys; open('graphify-out/.graphify_python', 'w').write(sys.executable)"
fi
```

---

## --update

Use when you've added or modified files since the last run. Only re-extracts changed files.

```bash
$(cat graphify-out/.graphify_python) -c "
import sys, json
from graphify.detect import detect_incremental, save_manifest
from pathlib import Path

result = detect_incremental(Path('INPUT_PATH'))
new_total = result.get('new_total', 0)
Path('graphify-out/.graphify_incremental.json').write_text(json.dumps(result))
if new_total == 0:
    print('No files changed since last run. Nothing to update.')
    raise SystemExit(0)
print(f'{new_total} new/changed file(s) to re-extract.')
"
```

Check if all changed files are code-only:

```bash
$(cat graphify-out/.graphify_python) -c "
import json
from pathlib import Path
result = json.loads(open('graphify-out/.graphify_incremental.json').read())
code_exts = {'.py','.ts','.js','.go','.rs','.java','.cpp','.c','.rb','.swift','.kt','.cs','.scala','.php','.cc','.cxx','.hpp','.h','.kts','.lua','.toc'}
all_changed = [f for files in result.get('new_files', {}).values() for f in files]
code_only = all(Path(f).suffix.lower() in code_exts for f in all_changed)
print('code_only:', code_only)
"
```

- **code_only = True**: print `[graphify update] Code-only changes — skipping semantic extraction`, run only Step 3A (AST) on changed files, skip Step 3B subagents.
- **code_only = False**: run full Steps 3A–3C pipeline.

Before merging, save old graph backup: `cp graphify-out/graph.json graphify-out/.graphify_old.json`

Merge new extraction into existing graph:

```bash
$(cat graphify-out/.graphify_python) -c "
import sys, json
from graphify.build import build_from_json
from graphify.export import to_json
from networkx.readwrite import json_graph
import networkx as nx
from pathlib import Path

existing_data = json.loads(Path('graphify-out/graph.json').read_text())
G_existing = json_graph.node_link_graph(existing_data, edges='links')

new_extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text())
G_new = build_from_json(new_extraction)

incremental = json.loads(Path('graphify-out/.graphify_incremental.json').read_text())
deleted = set(incremental.get('deleted_files', []))
if deleted:
    to_remove = [n for n, d in G_existing.nodes(data=True) if d.get('source_file') in deleted]
    G_existing.remove_nodes_from(to_remove)
    print(f'Pruned {len(to_remove)} ghost nodes from {len(deleted)} deleted file(s)')

G_existing.update(G_new)

merged_out = {
    'nodes': [{'id': n, **d} for n, d in G_existing.nodes(data=True)],
    'edges': [{'source': u, 'target': v, **d} for u, v, d in G_existing.edges(data=True)],
    'hyperedges': new_extraction.get('hyperedges', []),
    'input_tokens': new_extraction.get('input_tokens', 0),
    'output_tokens': new_extraction.get('output_tokens', 0),
}
Path('graphify-out/.graphify_extract.json').write_text(json.dumps(merged_out))
print(f'Merged: {len(merged_out[\"nodes\"])} nodes, {len(merged_out[\"edges\"])} edges')
"
```

Then run Steps 4–8 on the merged graph. After Step 4, show the graph diff:

```bash
$(cat graphify-out/.graphify_python) -c "
import json
from graphify.analyze import graph_diff
from graphify.build import build_from_json
from networkx.readwrite import json_graph
from pathlib import Path

old_data = json.loads(Path('graphify-out/.graphify_old.json').read_text()) if Path('graphify-out/.graphify_old.json').exists() else None
G_new = build_from_json(json.loads(Path('graphify-out/.graphify_extract.json').read_text()))

if old_data:
    G_old = json_graph.node_link_graph(old_data, edges='links')
    diff = graph_diff(G_old, G_new)
    print(diff['summary'])
    if diff['new_nodes']:
        print('New nodes:', ', '.join(n['label'] for n in diff['new_nodes'][:5]))
    if diff['new_edges']:
        print('New edges:', len(diff['new_edges']))
"
```

Cleanup: `rm -f graphify-out/.graphify_old.json`

---

## --cluster-only

Skip Steps 1–3. Load the existing graph and re-run clustering:

```bash
$(cat graphify-out/.graphify_python) -c "
from pathlib import Path
if not Path('graphify-out/graph.json').exists():
    print('ERROR: No graph found. Run /graphify <path> first.')
    raise SystemExit(1)
"
```

```bash
$(cat graphify-out/.graphify_python) -c "
import sys, json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections
from graphify.report import generate
from graphify.export import to_json
from networkx.readwrite import json_graph
from pathlib import Path

data = json.loads(Path('graphify-out/graph.json').read_text())
G = json_graph.node_link_graph(data, edges='links')

detection = {'total_files': 0, 'total_words': 99999, 'needs_graph': True, 'warning': None, 'files': {'code': [], 'document': [], 'paper': []}}
tokens = {'input': 0, 'output': 0}

communities = cluster(G)
cohesion = score_all(G, communities)
gods = god_nodes(G)
surprises = surprising_connections(G, communities)
labels = {cid: 'Community ' + str(cid) for cid in communities}

report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, '.')
Path('graphify-out/GRAPH_REPORT.md').write_text(report)
to_json(G, communities, 'graphify-out/graph.json')

analysis = {'communities': {str(k): v for k, v in communities.items()}, 'cohesion': {str(k): v for k, v in cohesion.items()}, 'gods': gods, 'surprises': surprises}
Path('graphify-out/.graphify_analysis.json').write_text(json.dumps(analysis, indent=2))
print(f'Re-clustered: {len(communities)} communities')
"
```

Then run Steps 5–9 (label, visualize, benchmark, finalize).

---

## query

Two traversal modes:

| Mode | Flag | Best for |
|------|------|----------|
| BFS (default) | _(none)_ | "What is X connected to?" — broad context, nearest neighbors first |
| DFS | `--dfs` | "How does X reach Y?" — trace a specific chain |

First check the graph exists — stop if not:
```bash
$(cat graphify-out/.graphify_python) -c "
from pathlib import Path
if not Path('graphify-out/graph.json').exists():
    print('ERROR: No graph found. Run /graphify <path> first.')
    raise SystemExit(1)
"
```

```bash
$(cat graphify-out/.graphify_python) -c "
import sys, json
from networkx.readwrite import json_graph
from pathlib import Path

data = json.loads(Path('graphify-out/graph.json').read_text())
G = json_graph.node_link_graph(data, edges='links')

question = 'QUESTION'
mode = 'MODE'   # 'bfs' or 'dfs'
terms = [t.lower() for t in question.split() if len(t) > 3]

scored = sorted([(sum(1 for t in terms if t in G.nodes[n].get('label','').lower()), n) for n in G.nodes()], reverse=True)
start_nodes = [nid for _, nid in scored[:3] if _ > 0]

if not start_nodes:
    print('No matching nodes found for:', terms)
    sys.exit(0)

subgraph_nodes = set()
subgraph_edges = []

if mode == 'dfs':
    visited, stack = set(), [(n, 0) for n in reversed(start_nodes)]
    while stack:
        node, depth = stack.pop()
        if node in visited or depth > 6: continue
        visited.add(node); subgraph_nodes.add(node)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                stack.append((neighbor, depth + 1))
                subgraph_edges.append((node, neighbor))
else:
    frontier = set(start_nodes); subgraph_nodes = set(start_nodes)
    for _ in range(3):
        next_frontier = set()
        for n in frontier:
            for neighbor in G.neighbors(n):
                if neighbor not in subgraph_nodes:
                    next_frontier.add(neighbor); subgraph_edges.append((n, neighbor))
        subgraph_nodes.update(next_frontier); frontier = next_frontier

token_budget = BUDGET   # default 2000, or --budget N
char_budget = token_budget * 4
ranked_nodes = sorted(subgraph_nodes, key=lambda nid: sum(1 for t in terms if t in G.nodes[nid].get('label','').lower()), reverse=True)

lines = [f'Traversal: {mode.upper()} | Start: {[G.nodes[n].get(\"label\",n) for n in start_nodes]} | {len(subgraph_nodes)} nodes']
for nid in ranked_nodes:
    d = G.nodes[nid]
    lines.append(f'  NODE {d.get(\"label\",nid)} [src={d.get(\"source_file\",\"\")} loc={d.get(\"source_location\",\"\")}]')
for u, v in subgraph_edges:
    if u in subgraph_nodes and v in subgraph_nodes:
        d = G.edges[u, v]
        lines.append(f'  EDGE {G.nodes[u].get(\"label\",u)} --{d.get(\"relation\",\"\")} [{d.get(\"confidence\",\"\")}]--> {G.nodes[v].get(\"label\",v)}')

output = '\n'.join(lines)
if len(output) > char_budget:
    output = output[:char_budget] + f'\n... (truncated at ~{token_budget} tokens)'
print(output)
"
```

Answer using **only** what the graph contains. Quote `source_location` when citing. Do not hallucinate edges.

After answering, save the result back:
```bash
$(cat graphify-out/.graphify_python) -m graphify save-result --question "QUESTION" --answer "ANSWER" --type query --nodes NODE1 NODE2
```

---

## path

Find the shortest path between two named concepts.

```bash
$(cat graphify-out/.graphify_python) -c "
from pathlib import Path
if not Path('graphify-out/graph.json').exists():
    print('ERROR: No graph found. Run /graphify <path> first.')
    raise SystemExit(1)
"
```

```bash
$(cat graphify-out/.graphify_python) -c "
import json, sys
import networkx as nx
from networkx.readwrite import json_graph
from pathlib import Path

data = json.loads(Path('graphify-out/graph.json').read_text())
G = json_graph.node_link_graph(data, edges='links')

def find_node(term):
    term = term.lower()
    scored = sorted([(sum(1 for w in term.split() if w in G.nodes[n].get('label','').lower()), n) for n in G.nodes()], reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None

src = find_node('NODE_A')
tgt = find_node('NODE_B')

if not src or not tgt:
    print(f'Could not find nodes matching: {\"NODE_A\"!r} or {\"NODE_B\"!r}')
    sys.exit(0)

try:
    path = nx.shortest_path(G, src, tgt)
    print(f'Shortest path ({len(path)-1} hops):')
    for i, nid in enumerate(path):
        label = G.nodes[nid].get('label', nid)
        if i < len(path) - 1:
            edge = G.edges[nid, path[i+1]]
            print(f'  {label} --{edge.get(\"relation\",\"\")}→ [{edge.get(\"confidence\",\"\")}]')
        else:
            print(f'  {label}')
except nx.NetworkXNoPath:
    print(f'No path found between NODE_A and NODE_B')
"
```

After explaining the path in plain language, save back:
```bash
$(cat graphify-out/.graphify_python) -m graphify save-result --question "Path from NODE_A to NODE_B" --answer "ANSWER" --type path_query --nodes NODE_A NODE_B
```

---

## explain

Explain a single node and everything connected to it.

```bash
$(cat graphify-out/.graphify_python) -c "
from pathlib import Path
if not Path('graphify-out/graph.json').exists():
    print('ERROR: No graph found. Run /graphify <path> first.')
    raise SystemExit(1)
"
```

```bash
$(cat graphify-out/.graphify_python) -c "
import json, sys
import networkx as nx
from networkx.readwrite import json_graph
from pathlib import Path

data = json.loads(Path('graphify-out/graph.json').read_text())
G = json_graph.node_link_graph(data, edges='links')

term = 'NODE_NAME'
scored = sorted([(sum(1 for w in term.lower().split() if w in G.nodes[n].get('label','').lower()), n) for n in G.nodes()], reverse=True)
if not scored or scored[0][0] == 0:
    print(f'No node matching {term!r}'); sys.exit(0)

nid = scored[0][1]
data_n = G.nodes[nid]
print(f'NODE: {data_n.get(\"label\", nid)}')
print(f'  source: {data_n.get(\"source_file\",\"unknown\")}')
print(f'  type: {data_n.get(\"file_type\",\"unknown\")}')
print(f'  degree: {G.degree(nid)}')
print()
print('CONNECTIONS:')
for neighbor in G.neighbors(nid):
    edge = G.edges[nid, neighbor]
    print(f'  --{edge.get(\"relation\",\"\")}→ {G.nodes[neighbor].get(\"label\",neighbor)} [{edge.get(\"confidence\",\"\")}] ({G.nodes[neighbor].get(\"source_file\",\"\")})')
"
```

Write a 3-5 sentence explanation. Cite source locations. Then save back:
```bash
$(cat graphify-out/.graphify_python) -m graphify save-result --question "Explain NODE_NAME" --answer "ANSWER" --type explain --nodes NODE_NAME
```

---

## add

Fetch a URL and add it to the corpus, then update the graph.

```bash
$(cat graphify-out/.graphify_python) -c "
import sys
from graphify.ingest import ingest
from pathlib import Path

try:
    out = ingest('URL', Path('./raw'), author='AUTHOR', contributor='CONTRIBUTOR')
    print(f'Saved to {out}')
except (ValueError, RuntimeError) as e:
    print(f'error: {e}', file=sys.stderr)
    sys.exit(1)
"
```

On success, auto-run `--update` on `./raw` to merge into the existing graph.

Supported URL types (auto-detected): YouTube/video → yt-dlp transcript, Twitter/X → oEmbed markdown, arXiv → abstract markdown, PDF → downloaded, images → vision on next run, any webpage → html2text markdown.

---

## --watch

```bash
python3 -m graphify.watch INPUT_PATH --debounce 3
```

- **Code files (.py, .ts, etc.):** instant AST rebuild, no LLM needed.
- **Docs/papers/images:** writes `graphify-out/needs_update` flag; manual `/graphify --update` required.

Debounce (default 3s) waits until file activity stops before triggering. Press Ctrl+C to stop.

---

## git-hook

```bash
graphify hook install    # install post-commit hook
graphify hook uninstall  # remove
graphify hook status     # check
```

After every `git commit`: detects changed code files via `git diff HEAD~1`, re-runs AST extraction on those files, rebuilds `graph.json` and `GRAPH_REPORT.md`. Doc/image changes ignored — run `/graphify --update` manually for those. Appends to existing hooks, does not replace them.

---

## claude-md

```bash
graphify claude install    # write ## graphify section to local CLAUDE.md
graphify claude uninstall  # remove the section
```

Makes graphify always-on: Claude checks the graph before answering codebase questions and rebuilds it after code changes. No manual `/graphify` needed in future sessions.

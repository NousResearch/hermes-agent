# graphify — Full Pipeline Reference

Complete bash/Python implementation for Steps 0–9. Each step maps to the table in SKILL.md.

---

## Step 0 — Clone GitHub repo(s) (only if a GitHub URL was given)

**Single repo:**
```bash
LOCAL_PATH=$(graphify clone <github-url> [--branch <branch>])
# Use LOCAL_PATH as the target for all subsequent steps
```

**Multiple repos (cross-repo graph):**
```bash
graphify clone <url1>   # → ~/.graphify/repos/<owner1>/<repo1>
graphify clone <url2>   # → ~/.graphify/repos/<owner2>/<repo2>
# Run /graphify on each local path to produce their graph.json files, then merge:
graphify merge-graphs \
  ~/.graphify/repos/<owner1>/<repo1>/graphify-out/graph.json \
  ~/.graphify/repos/<owner2>/<repo2>/graphify-out/graph.json \
  --out graphify-out/cross-repo-graph.json
```

Graphify clones into `~/.graphify/repos/<owner>/<repo>` and reuses existing clones. Each node in the merged graph carries a `repo` attribute so you can filter by origin.

---

## Step 1 — Ensure graphify is installed

```bash
PYTHON=""
GRAPHIFY_BIN=$(which graphify 2>/dev/null)
# 1. uv tool installs
if [ -z "$PYTHON" ] && command -v uv >/dev/null 2>&1; then
    _UV_PY=$(uv tool run graphifyy python -c "import sys; print(sys.executable)" 2>/dev/null)
    if [ -n "$_UV_PY" ]; then PYTHON="$_UV_PY"; fi
fi
# 2. Read shebang from graphify binary
if [ -z "$PYTHON" ] && [ -n "$GRAPHIFY_BIN" ]; then
    _SHEBANG=$(head -1 "$GRAPHIFY_BIN" | tr -d '#!')
    case "$_SHEBANG" in
        *[!a-zA-Z0-9/_.-]*) ;;
        *) "$_SHEBANG" -c "import graphify" 2>/dev/null && PYTHON="$_SHEBANG" ;;
    esac
fi
# 3. Fall back to python3
if [ -z "$PYTHON" ]; then PYTHON="python3"; fi
"$PYTHON" -c "import graphify" 2>/dev/null || "$PYTHON" -m pip install graphifyy -q 2>/dev/null || "$PYTHON" -m pip install graphifyy -q --break-system-packages 2>&1 | tail -3
mkdir -p graphify-out
"$PYTHON" -c "import sys; open('graphify-out/.graphify_python', 'w').write(sys.executable)"
```

In every subsequent bash block, replace `python3` with `$(cat graphify-out/.graphify_python)`.

---

## Step 2 — Detect files

```bash
$(cat graphify-out/.graphify_python) -c "
import json
from graphify.detect import detect
from pathlib import Path
result = detect(Path('INPUT_PATH'))
print(json.dumps(result))
" > graphify-out/.graphify_detect.json
```

Replace `INPUT_PATH` with the actual path. Do NOT cat the JSON — read it silently and present a clean summary:

```
Corpus: X files · ~Y words
  code:     N files (.py .ts .go ...)
  docs:     N files (.md .txt ...)
  papers:   N files (.pdf ...)
  images:   N files
  video:    N files (.mp4 .mp3 ...)
```

Omit categories with 0 files. Then:
- `total_files == 0`: stop with "No supported files found in [path]."
- `skipped_sensitive` non-empty: mention file count skipped, not filenames.
- `total_words > 2,000,000` OR `total_files > 200`: show warning + top 5 subdirs by file count; ask which subfolder to run on. Wait for answer.
- Otherwise: proceed to Step 2.5 if video files detected, else Step 3.

---

## Step 2.5 — Transcribe video/audio (only if video files detected)

Skip entirely if `detect` returned zero `video` files.

**Step 1 — Write the Whisper prompt yourself.** Read the top god node labels from detect output and compose a short domain hint:
- Labels: `transformer, attention, encoder` → `"Machine learning research on transformer architectures. Use proper punctuation and paragraph breaks."`
- Labels: `kubernetes, deployment, pod` → `"DevOps discussion about Kubernetes deployments. Use proper punctuation and paragraph breaks."`

If corpus has *only* video files, use: `"Use proper punctuation and paragraph breaks."`

**Step 2 — Transcribe:**

```bash
GRAPHIFY_WHISPER_MODEL=base  # or --whisper-model value from user
$(cat graphify-out/.graphify_python) -c "
import json, os
from pathlib import Path
from graphify.transcribe import transcribe_all

detect = json.loads(Path('graphify-out/.graphify_detect.json').read_text())
video_files = detect.get('files', {}).get('video', [])
prompt = os.environ.get('GRAPHIFY_WHISPER_PROMPT', 'Use proper punctuation and paragraph breaks.')

transcript_paths = transcribe_all(video_files, initial_prompt=prompt)
print(json.dumps(transcript_paths))
" > graphify-out/.graphify_transcripts.json
```

Add transcripts to docs list before Step 3B subagents. Print: `Transcribed N video file(s) -> treating as docs`. On failure per file, warn and continue.

---

## Step 3 — Extract entities and relationships

Run Part A (AST) and Part B (semantic subagents) **in parallel** — dispatch all in the same message.

Note whether `--mode deep` was given; pass `DEEP_MODE=true` to every subagent if so.

### Part A — Structural extraction (code files)

```bash
$(cat graphify-out/.graphify_python) -c "
import sys, json
from graphify.extract import collect_files, extract
from pathlib import Path

code_files = []
detect = json.loads(Path('graphify-out/.graphify_detect.json').read_text())
for f in detect.get('files', {}).get('code', []):
    code_files.extend(collect_files(Path(f)) if Path(f).is_dir() else [Path(f)])

if code_files:
    result = extract(code_files, cache_root=Path('.'))
    Path('graphify-out/.graphify_ast.json').write_text(json.dumps(result, indent=2))
    print(f'AST: {len(result[\"nodes\"])} nodes, {len(result[\"edges\"])} edges')
else:
    Path('graphify-out/.graphify_ast.json').write_text(json.dumps({'nodes':[],'edges':[],'input_tokens':0,'output_tokens':0}))
    print('No code files - skipping AST extraction')
"
```

### Part B — Semantic extraction (parallel subagents)

**Fast path:** If zero docs/papers/images, skip Part B — go straight to Part C.

**MANDATORY: Use the Agent tool here. Reading files yourself is 5-10x slower.**

Before dispatching, print an estimate: `ceil(uncached_non_code_files / 22)` agents, ~45s per batch.

**Step B0 — Check extraction cache:**

```bash
$(cat graphify-out/.graphify_python) -c "
import json
from graphify.cache import check_semantic_cache
from pathlib import Path

detect = json.loads(Path('graphify-out/.graphify_detect.json').read_text())
all_files = [f for files in detect['files'].values() for f in files]
cached_nodes, cached_edges, cached_hyperedges, uncached = check_semantic_cache(all_files)
if cached_nodes or cached_edges or cached_hyperedges:
    Path('graphify-out/.graphify_cached.json').write_text(json.dumps({'nodes': cached_nodes, 'edges': cached_edges, 'hyperedges': cached_hyperedges}))
Path('graphify-out/.graphify_uncached.txt').write_text('\n'.join(uncached))
print(f'Cache: {len(all_files)-len(uncached)} files hit, {len(uncached)} files need extraction')
"
```

Only dispatch subagents for files in `.graphify_uncached.txt`. If all cached, skip to Part C.

**Step B1 — Split into chunks** of 20-25 files. Each image gets its own chunk. Group files from the same directory together.

**Step B2 — Dispatch ALL subagents in a single message** (not sequential calls). Always use `subagent_type="general-purpose"` — NOT `Explore` (read-only, silently drops results).

Each subagent prompt (substitute FILE_LIST, CHUNK_NUM, TOTAL_CHUNKS, DEEP_MODE):

```
You are a graphify extraction subagent. Read the files listed and extract a knowledge graph fragment.
Output ONLY valid JSON matching the schema below - no explanation, no markdown fences, no preamble.

Files (chunk CHUNK_NUM of TOTAL_CHUNKS):
FILE_LIST

Rules:
- EXTRACTED: relationship explicit in source (import, call, citation, "see §3.2")
- INFERRED: reasonable inference (shared data structure, implied dependency)
- AMBIGUOUS: uncertain - flag for review, do not omit

Code files: focus on semantic edges AST cannot find. Do not re-extract imports.
Doc/paper files: extract named concepts, entities, citations, rationale (WHY decisions were made).
Image files: use vision — UI screenshots, charts, tweets, diagrams, research figures.

DEEP_MODE (if --mode deep): be aggressive with INFERRED edges. Mark uncertain ones AMBIGUOUS.

Semantic similarity: add `semantically_similar_to` edges (INFERRED) when two concepts solve the same
problem without any structural link. Only when genuinely non-obvious. confidence_score 0.6-0.95.

Hyperedges: if 3+ nodes participate together in a shared concept/flow/pattern, add to `hyperedges`.
Max 3 per chunk.

Node ID format: lowercase `[a-z0-9_]` only. Format: `{stem}_{entity}`. Example:
`src/auth/session.py` + `ValidateToken` → `session_validatetoken`.
CRITICAL: never append chunk numbers or suffixes. IDs must be deterministic from label alone.

confidence_score REQUIRED on every edge:
- EXTRACTED: 1.0 always
- INFERRED: 0.6-0.9 (reason about each individually)
- AMBIGUOUS: 0.1-0.3

Output exactly this JSON:
{"nodes":[{"id":"session_validatetoken","label":"Human Readable Name","file_type":"code|document|paper|image","source_file":"relative/path","source_location":null,"source_url":null,"captured_at":null,"author":null,"contributor":null}],"edges":[{"source":"node_id","target":"node_id","relation":"calls|implements|references|cites|conceptually_related_to|shares_data_with|semantically_similar_to|rationale_for","confidence":"EXTRACTED|INFERRED|AMBIGUOUS","confidence_score":1.0,"source_file":"relative/path","source_location":null,"weight":1.0}],"hyperedges":[{"id":"snake_case_id","label":"Human Readable Label","nodes":["node_id1","node_id2","node_id3"],"relation":"participate_in|implement|form","confidence":"EXTRACTED|INFERRED","confidence_score":0.75,"source_file":"relative/path"}],"input_tokens":0,"output_tokens":0}
```

**Step B3 — Collect, cache, and merge:**

Check `graphify-out/.graphify_chunk_NN.json` exists for each subagent. If missing, the subagent was read-only — warn, do not silently skip. If >50% of chunks failed, stop and ask user to re-run with `general-purpose` agents.

Save new results to cache:
```bash
$(cat graphify-out/.graphify_python) -c "
import json
from graphify.cache import save_semantic_cache
from pathlib import Path
new = json.loads(Path('graphify-out/.graphify_semantic_new.json').read_text()) if Path('graphify-out/.graphify_semantic_new.json').exists() else {'nodes':[],'edges':[],'hyperedges':[]}
saved = save_semantic_cache(new.get('nodes', []), new.get('edges', []), new.get('hyperedges', []))
print(f'Cached {saved} files')
"
```

Merge cached + new into `.graphify_semantic.json`, deduplicating nodes by id. Then:
```bash
rm -f graphify-out/.graphify_cached.json graphify-out/.graphify_uncached.txt graphify-out/.graphify_semantic_new.json
```

### Part C — Merge AST + semantic

```bash
$(cat graphify-out/.graphify_python) -c "
import sys, json
from pathlib import Path

ast = json.loads(Path('graphify-out/.graphify_ast.json').read_text())
sem = json.loads(Path('graphify-out/.graphify_semantic.json').read_text())

seen = {n['id'] for n in ast['nodes']}
merged_nodes = list(ast['nodes'])
for n in sem['nodes']:
    if n['id'] not in seen:
        merged_nodes.append(n)
        seen.add(n['id'])

merged = {
    'nodes': merged_nodes,
    'edges': ast['edges'] + sem['edges'],
    'hyperedges': sem.get('hyperedges', []),
    'input_tokens': sem.get('input_tokens', 0),
    'output_tokens': sem.get('output_tokens', 0),
}
Path('graphify-out/.graphify_extract.json').write_text(json.dumps(merged, indent=2))
print(f'Merged: {len(merged_nodes)} nodes, {len(merged[\"edges\"])} edges')
"
```

---

## Step 4 — Build graph, cluster, analyze, generate outputs

Note whether `--directed` was given; if so, pass `directed=True` to `build_from_json()`.

```bash
mkdir -p graphify-out
$(cat graphify-out/.graphify_python) -c "
import sys, json
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json
from pathlib import Path

extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text())
detection  = json.loads(Path('graphify-out/.graphify_detect.json').read_text())

G = build_from_json(extraction)
communities = cluster(G)
cohesion = score_all(G, communities)
tokens = {'input': extraction.get('input_tokens', 0), 'output': extraction.get('output_tokens', 0)}
gods = god_nodes(G)
surprises = surprising_connections(G, communities)
labels = {cid: 'Community ' + str(cid) for cid in communities}
questions = suggest_questions(G, communities, labels)

report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, 'INPUT_PATH', suggested_questions=questions)
Path('graphify-out/GRAPH_REPORT.md').write_text(report)
to_json(G, communities, 'graphify-out/graph.json')

analysis = {
    'communities': {str(k): v for k, v in communities.items()},
    'cohesion': {str(k): v for k, v in cohesion.items()},
    'gods': gods,
    'surprises': surprises,
    'questions': questions,
}
Path('graphify-out/.graphify_analysis.json').write_text(json.dumps(analysis, indent=2))
if G.number_of_nodes() == 0:
    print('ERROR: Graph is empty.')
    raise SystemExit(1)
print(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(communities)} communities')
"
```

If `ERROR: Graph is empty` — stop and tell the user. Do not proceed.

---

## Step 5 — Label communities

Read `.graphify_analysis.json`. Write 2-5 word names per community (e.g. "Attention Mechanism", "Training Pipeline"). Then regenerate report:

```bash
$(cat graphify-out/.graphify_python) -c "
import sys, json
from graphify.build import build_from_json
from graphify.cluster import score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from pathlib import Path

extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text())
detection  = json.loads(Path('graphify-out/.graphify_detect.json').read_text())
analysis   = json.loads(Path('graphify-out/.graphify_analysis.json').read_text())

G = build_from_json(extraction)
communities = {int(k): v for k, v in analysis['communities'].items()}
cohesion = {int(k): v for k, v in analysis['cohesion'].items()}
tokens = {'input': extraction.get('input_tokens', 0), 'output': extraction.get('output_tokens', 0)}

labels = LABELS_DICT  # replace with {0: 'Attention Mechanism', 1: 'Training Pipeline', ...}
questions = suggest_questions(G, communities, labels)

report = generate(G, communities, cohesion, labels, analysis['gods'], analysis['surprises'], detection, tokens, 'INPUT_PATH', suggested_questions=questions)
Path('graphify-out/GRAPH_REPORT.md').write_text(report)
Path('graphify-out/.graphify_labels.json').write_text(json.dumps({str(k): v for k, v in labels.items()}))
print('Report updated with community labels')
"
```

---

## Step 6 — Visualize

**HTML always** (unless `--no-viz`). **Obsidian only if `--obsidian` was explicitly given.**

### Obsidian vault (only if --obsidian)

```bash
$(cat graphify-out/.graphify_python) -c "
import sys, json
from graphify.build import build_from_json
from graphify.export import to_obsidian, to_canvas
from pathlib import Path

extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text())
analysis   = json.loads(Path('graphify-out/.graphify_analysis.json').read_text())
labels_raw = json.loads(Path('graphify-out/.graphify_labels.json').read_text()) if Path('graphify-out/.graphify_labels.json').exists() else {}

G = build_from_json(extraction)
communities = {int(k): v for k, v in analysis['communities'].items()}
cohesion = {int(k): v for k, v in analysis['cohesion'].items()}
labels = {int(k): v for k, v in labels_raw.items()}
obsidian_dir = 'OBSIDIAN_DIR'  # --obsidian-dir value, or 'graphify-out/obsidian'

n = to_obsidian(G, communities, obsidian_dir, community_labels=labels or None, cohesion=cohesion)
to_canvas(G, communities, f'{obsidian_dir}/graph.canvas', community_labels=labels or None)
print(f'Obsidian vault: {n} notes + graph.canvas in {obsidian_dir}/')
"
```

### HTML graph (always unless --no-viz)

Graphs >5,000 nodes get an aggregated community view instead of node-level detail.

```bash
$(cat graphify-out/.graphify_python) -c "
import sys, json
from graphify.build import build_from_json
from graphify.export import to_html
from pathlib import Path

extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text())
analysis   = json.loads(Path('graphify-out/.graphify_analysis.json').read_text())
labels_raw = json.loads(Path('graphify-out/.graphify_labels.json').read_text()) if Path('graphify-out/.graphify_labels.json').exists() else {}

G = build_from_json(extraction)
communities = {int(k): v for k, v in analysis['communities'].items()}
labels = {int(k): v for k, v in labels_raw.items()}

NODE_LIMIT = 5000
if G.number_of_nodes() > NODE_LIMIT:
    # Build aggregated community-level meta-graph
    from collections import Counter
    import networkx as nx_meta
    node_to_community = {nid: cid for cid, members in communities.items() for nid in members}
    meta = nx_meta.Graph()
    for cid in communities:
        meta.add_node(str(cid), label=labels.get(cid, f'Community {cid}'))
    edge_counts = Counter()
    for u, v in G.edges():
        cu, cv = node_to_community.get(u), node_to_community.get(v)
        if cu is not None and cv is not None and cu != cv:
            edge_counts[(min(cu,cv), max(cu,cv))] += 1
    for (cu, cv), w in edge_counts.items():
        meta.add_edge(str(cu), str(cv), weight=w, relation=f'{w} cross-community edges', confidence='AGGREGATED')
    if meta.number_of_nodes() > 1:
        member_counts = {cid: len(members) for cid, members in communities.items()}
        to_html(meta, {cid: [str(cid)] for cid in communities}, 'graphify-out/graph.html', community_labels=labels or None, member_counts=member_counts)
        print(f'graph.html (aggregated: {meta.number_of_nodes()} community nodes)')
else:
    to_html(G, communities, 'graphify-out/graph.html', community_labels=labels or None)
    print('graph.html written')
"
```

---

## Step 6b — Wiki (only if --wiki)

```bash
$(cat graphify-out/.graphify_python) -c "
import json
from graphify.build import build_from_json
from graphify.wiki import to_wiki
from graphify.analyze import god_nodes
from pathlib import Path

extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text())
analysis   = json.loads(Path('graphify-out/.graphify_analysis.json').read_text())
labels_raw = json.loads(Path('graphify-out/.graphify_labels.json').read_text()) if Path('graphify-out/.graphify_labels.json').exists() else {}

G = build_from_json(extraction)
communities = {int(k): v for k, v in analysis['communities'].items()}
labels = {int(k): v for k, v in labels_raw.items()}

n = to_wiki(G, communities, 'graphify-out/wiki', community_labels=labels or None, cohesion={int(k): v for k,v in analysis['cohesion'].items()}, god_nodes_data=god_nodes(G))
print(f'Wiki: {n} articles in graphify-out/wiki/')
print('  graphify-out/wiki/index.md  ->  agent entry point')
"
```

---

## Step 7 — Neo4j export (only if --neo4j or --neo4j-push)

**`--neo4j` (file):**
```bash
$(cat graphify-out/.graphify_python) -c "
from graphify.build import build_from_json
from graphify.export import to_cypher
import json
from pathlib import Path
G = build_from_json(json.loads(Path('graphify-out/.graphify_extract.json').read_text()))
to_cypher(G, 'graphify-out/cypher.txt')
print('cypher.txt written - import with: cypher-shell < graphify-out/cypher.txt')
"
```

**`--neo4j-push <uri>` (direct push):** Ask for credentials if not provided. Replace `NEO4J_URI/USER/PASSWORD`. Uses MERGE — safe to re-run.

```bash
$(cat graphify-out/.graphify_python) -c "
import json
from graphify.build import build_from_json
from graphify.export import push_to_neo4j
from pathlib import Path
extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text())
analysis   = json.loads(Path('graphify-out/.graphify_analysis.json').read_text())
G = build_from_json(extraction)
communities = {int(k): v for k, v in analysis['communities'].items()}
result = push_to_neo4j(G, uri='NEO4J_URI', user='NEO4J_USER', password='NEO4J_PASSWORD', communities=communities)
print(f'Pushed: {result[\"nodes\"]} nodes, {result[\"edges\"]} edges')
"
```

---

## Step 7b — SVG export (only if --svg)

```bash
$(cat graphify-out/.graphify_python) -c "
import json
from graphify.build import build_from_json
from graphify.export import to_svg
from pathlib import Path
extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text())
analysis   = json.loads(Path('graphify-out/.graphify_analysis.json').read_text())
labels_raw = json.loads(Path('graphify-out/.graphify_labels.json').read_text()) if Path('graphify-out/.graphify_labels.json').exists() else {}
G = build_from_json(extraction)
communities = {int(k): v for k, v in analysis['communities'].items()}
to_svg(G, communities, 'graphify-out/graph.svg', community_labels={int(k): v for k,v in labels_raw.items()} or None)
print('graph.svg written')
"
```

---

## Step 7c — GraphML export (only if --graphml)

```bash
$(cat graphify-out/.graphify_python) -c "
import json
from graphify.build import build_from_json
from graphify.export import to_graphml
from pathlib import Path
extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text())
analysis   = json.loads(Path('graphify-out/.graphify_analysis.json').read_text())
G = build_from_json(extraction)
communities = {int(k): v for k, v in analysis['communities'].items()}
to_graphml(G, communities, 'graphify-out/graph.graphml')
print('graph.graphml written - open in Gephi or yEd')
"
```

---

## Step 7d — MCP server (only if --mcp)

```bash
$(cat graphify-out/.graphify_python) -m graphify.serve graphify-out/graph.json
```

Exposes tools: `query_graph`, `get_node`, `get_neighbors`, `get_community`, `god_nodes`, `graph_stats`, `shortest_path`.

Claude Desktop config:
```json
{
  "mcpServers": {
    "graphify": {
      "command": "python3",
      "args": ["-m", "graphify.serve", "/absolute/path/to/graphify-out/graph.json"]
    }
  }
}
```

---

## Step 8 — Token reduction benchmark (only if total_words > 5,000)

```bash
$(cat graphify-out/.graphify_python) -c "
import json
from graphify.benchmark import run_benchmark, print_benchmark
from pathlib import Path
detection = json.loads(Path('graphify-out/.graphify_detect.json').read_text())
result = run_benchmark('graphify-out/graph.json', corpus_words=detection['total_words'])
print_benchmark(result)
"
```

---

## Step 9 — Finalize

```bash
$(cat graphify-out/.graphify_python) -c "
import json
from pathlib import Path
from datetime import datetime, timezone
from graphify.detect import save_manifest

detect = json.loads(Path('graphify-out/.graphify_detect.json').read_text())
save_manifest(detect['files'])

extract = json.loads(Path('graphify-out/.graphify_extract.json').read_text())
input_tok = extract.get('input_tokens', 0)
output_tok = extract.get('output_tokens', 0)

cost_path = Path('graphify-out/cost.json')
cost = json.loads(cost_path.read_text()) if cost_path.exists() else {'runs': [], 'total_input_tokens': 0, 'total_output_tokens': 0}
cost['runs'].append({'date': datetime.now(timezone.utc).isoformat(), 'input_tokens': input_tok, 'output_tokens': output_tok, 'files': detect.get('total_files', 0)})
cost['total_input_tokens'] += input_tok
cost['total_output_tokens'] += output_tok
cost_path.write_text(json.dumps(cost, indent=2))

print(f'This run: {input_tok:,} input tokens, {output_tok:,} output tokens')
print(f'All time: {cost[\"total_input_tokens\"]:,} input, {cost[\"total_output_tokens\"]:,} output ({len(cost[\"runs\"])} runs)')
"
rm -f graphify-out/.graphify_detect.json graphify-out/.graphify_extract.json graphify-out/.graphify_ast.json graphify-out/.graphify_semantic.json graphify-out/.graphify_analysis.json graphify-out/.graphify_labels.json graphify-out/.graphify_chunk_*.json graphify-out/.graphify_transcripts.json
rm -f graphify-out/.needs_update 2>/dev/null || true
```

Tell the user (omit obsidian line unless `--obsidian` was given):

```
Graph complete. Outputs in PATH_TO_DIR/graphify-out/

  graph.html            - interactive graph, open in browser
  GRAPH_REPORT.md       - audit report
  graph.json            - raw graph data
  obsidian/             - Obsidian vault (only if --obsidian was given)
```

Replace PATH_TO_DIR with the actual absolute path processed.

Paste these sections from GRAPH_REPORT.md directly into chat:
- God Nodes
- Surprising Connections
- Suggested Questions

Then pick the single most interesting suggested question and ask:

> "The most interesting question this graph can answer: **[question]**. Want me to trace it?"

If yes, run `/graphify query "[question]"` and walk through the answer — which nodes connect, which community boundaries get crossed, what the path reveals. Offer a follow-up each time.

If you found graphify useful, consider supporting it: https://github.com/sponsors/safishamsi

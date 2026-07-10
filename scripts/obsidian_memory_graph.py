#!/usr/bin/env python3
"""Build a cyberpunk 3D directed graph of Obsidian vaults (wikilinks + #tags) with WebXR VR."""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]")
TAG_RE = re.compile(r"(?<!\w)#([a-zA-Z0-9_\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff/-]+)")

CYBER_COLORS = {
    "concepts": "#ff00ff",
    "sessions": "#00fff9",
    "raw": "#ff6b00",
    "memory": "#b026ff",
    "root": "#00d4ff",
    "box": "#39ff14",
    "tag": "#ffd700",
    "ghost": "#666666",
    "other": "#7fffd4",
}

VAULT_PALETTE = {
    "obsidianvault": "memory",
    "box": "box",
}


def vault_slug(path: Path) -> str:
    return path.name.lower().replace(" ", "-")


def resolve_vaults(explicit: list[str] | None) -> list[Path]:
    if explicit:
        out: list[Path] = []
        for raw in explicit:
            p = Path(raw).expanduser().resolve()
            if not p.is_dir():
                raise SystemExit(f"Vault not found: {p}")
            out.append(p)
        return out

    appdata = Path.home() / "AppData" / "Roaming" / "obsidian" / "obsidian.json"
    if appdata.is_file():
        data = json.loads(appdata.read_text(encoding="utf-8"))
        vaults = data.get("vaults") or {}
        # Include every registered vault (box + ObsidianVault), not only "open".
        all_paths = [Path(v["path"]).resolve() for v in vaults.values()]
        found = sorted({p for p in all_paths if p.is_dir()}, key=lambda p: p.name.lower())
        if found:
            return found

    fallbacks = [
        Path.home() / "Documents" / "ObsidianVault",
        Path.home() / "Documents" / "box",
    ]
    found = [p.resolve() for p in fallbacks if p.is_dir()]
    if found:
        return found
    raise SystemExit("No Obsidian vault found. Pass --vault PATH (repeatable).")


def note_key(path: Path, vault: Path) -> str:
    return path.relative_to(vault).with_suffix("").as_posix()


def node_id(vault_slug_name: str, key: str) -> str:
    return f"{vault_slug_name}::{key}"


def tag_node_id(vault_slug_name: str, tag: str) -> str:
    return f"{vault_slug_name}::#tag::{tag.lower()}"


def display_title(path: Path, vault: Path) -> str:
    rel = path.relative_to(vault)
    if len(rel.parts) > 1:
        return rel.with_suffix("").as_posix().split("/")[-1]
    return path.stem


def category_for(rel_posix: str, vault_slug_name: str) -> str:
    base = VAULT_PALETTE.get(vault_slug_name, "other")
    low = rel_posix.lower()
    if "hermes-memory-wiki/concepts" in low or low.startswith("concepts/"):
        return "concepts"
    if "hermes-sessions" in low:
        return "sessions"
    if "/raw/" in low or low.startswith("raw/"):
        return "raw"
    if "hermes-memory" in low:
        return "memory"
    if rel_posix in ("index", "Hermes-Memory-Wiki/index"):
        return "root"
    if base == "box":
        return "box"
    return base if base != "memory" else "other"


def build_index(vault: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in vault.rglob("*.md"):
        if ".obsidian" in path.parts:
            continue
        key = note_key(path, vault)
        index[key.lower()] = path
        index[path.stem.lower()] = path
        alias = key.split("/")[-1].lower()
        if alias not in index:
            index[alias] = path
    return index


def resolve_link(target: str, source: Path, vault: Path, index: dict[str, Path]) -> Path | None:
    t = target.strip().replace("\\", "/")
    if not t:
        return None
    candidates = [
        t.lower(),
        t.split("/")[-1].lower(),
        note_key((source.parent / t).with_suffix(".md"), vault).lower()
        if not t.endswith(".md")
        else note_key((source.parent / t), vault).lower(),
    ]
    for c in candidates:
        hit = index.get(c)
        if hit and hit.exists():
            return hit
    direct = vault / f"{t}.md" if not t.endswith(".md") else vault / t
    if direct.is_file():
        return direct
    return None


def excerpt(text: str, limit: int = 160) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith(">"):
            s = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", s)
            return (s[: limit - 1] + "…") if len(s) > limit else s
    return ""


def strip_for_tags(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = WIKILINK_RE.sub("", text)
    return text


def extract_tags(text: str) -> list[str]:
    cleaned = strip_for_tags(text)
    seen: set[str] = set()
    tags: list[str] = []
    for m in TAG_RE.finditer(cleaned):
        t = m.group(1).strip().rstrip("/")
        if not t or t.isdigit():
            continue
        low = t.lower()
        if low not in seen:
            seen.add(low)
            tags.append(t)
    return tags


def ensure_note_node(
    nodes: dict[str, dict],
    nid: str,
    label: str,
    group: str,
    snippet: str = "",
    is_tag: bool = False,
) -> None:
    color = CYBER_COLORS.get(group, CYBER_COLORS["other"])
    if nid not in nodes:
        nodes[nid] = {
            "id": nid,
            "name": label,
            "group": group,
            "val": 3 if is_tag else 6,
            "color": color,
            "snippet": snippet,
            "isTag": is_tag,
        }
    elif not is_tag:
        nodes[nid]["val"] = nodes[nid].get("val", 6) + 0.5


def add_link(
    edges: list[dict],
    edge_seen: set[tuple[str, str, str]],
    source: str,
    target: str,
    kind: str,
    color: str,
) -> None:
    if source == target:
        return
    sig = (source, target, kind)
    if sig in edge_seen:
        return
    edge_seen.add(sig)
    edges.append(
        {
            "source": source,
            "target": target,
            "kind": kind,
            "color": color,
        }
    )


def scan_vault(
    vault: Path,
    vslug: str,
    nodes: dict[str, dict],
    edges: list[dict],
    edge_seen: set[tuple[str, str, str]],
) -> None:
    index = build_index(vault)
    md_files = [p for p in vault.rglob("*.md") if ".obsidian" not in p.parts]

    for path in tqdm(md_files, desc=f"Scan {vault.name}", unit="note"):
        key = note_key(path, vault)
        nid = node_id(vslug, key)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        cat = category_for(key, vslug)
        ensure_note_node(
            nodes,
            nid,
            display_title(path, vault),
            cat,
            excerpt(text),
        )

        for match in WIKILINK_RE.finditer(text):
            target_raw = match.group(1).strip()
            dest = resolve_link(target_raw, path, vault, index)
            if dest is None:
                ghost_key = target_raw.replace("\\", "/")
                ghost_id = node_id(vslug, f"_unresolved/{ghost_key}")
                ensure_note_node(
                    nodes,
                    ghost_id,
                    ghost_key.split("/")[-1][:28],
                    "ghost",
                    f"Unresolved: {ghost_key}",
                )
                dest_nid = ghost_id
            else:
                dest_key = note_key(dest, vault)
                dest_nid = node_id(vslug, dest_key)
                try:
                    dtext = dest.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    dtext = ""
                ensure_note_node(
                    nodes,
                    dest_nid,
                    display_title(dest, vault),
                    category_for(dest_key, vslug),
                    excerpt(dtext),
                )

            add_link(edges, edge_seen, nid, dest_nid, "wikilink", "#00fff9aa")
            nodes[nid]["val"] = nodes[nid].get("val", 6) + 0.4
            nodes[dest_nid]["val"] = nodes[dest_nid].get("val", 6) + 0.4

        tag_ids: list[str] = []
        for tag in extract_tags(text):
            tid = tag_node_id(vslug, tag)
            ensure_note_node(nodes, tid, f"#{tag}", "tag", f"Tag in {key}", is_tag=True)
            add_link(edges, edge_seen, nid, tid, "tag", "#ffd700cc")
            nodes[tid]["val"] = nodes[tid].get("val", 3) + 0.6
            tag_ids.append(tid)

        for i in range(len(tag_ids)):
            for j in range(i + 1, len(tag_ids)):
                add_link(edges, edge_seen, tag_ids[i], tag_ids[j], "tag-cooc", "#cc66ff99")
                nodes[tag_ids[i]]["val"] = nodes[tag_ids[i]].get("val", 3) + 0.15
                nodes[tag_ids[j]]["val"] = nodes[tag_ids[j]].get("val", 3) + 0.15

    for node in nodes.values():
        if node.get("isTag"):
            node["val"] = max(2, min(14, int(node.get("val", 3))))
        else:
            node["val"] = max(4, min(28, int(node.get("val", 6))))


def scan_vaults(vaults: list[Path]) -> tuple[list[dict], list[dict], list[str]]:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    edge_seen: set[tuple[str, str, str]] = set()
    labels: list[str] = []

    for vault in vaults:
        vslug = vault_slug(vault)
        labels.append(f"{vault.name} ({vault})")
        scan_vault(vault, vslug, nodes, edges, edge_seen)

    return list(nodes.values()), edges, labels


def local_lan_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def tailscale_magic_dns() -> str | None:
    try:
        import subprocess

        proc = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            timeout=8,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        stdout = (proc.stdout or "").strip()
        if proc.returncode != 0 or not stdout:
            return None
        data = json.loads(stdout)
        dns = (data.get("Self") or {}).get("DNSName") or ""
        dns = str(dns).strip().rstrip(".")
        return dns or None
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError):
        return None


def vr_access_urls(port: int = 8765) -> list[str]:
    ip = local_lan_ip()
    urls = [
        f"http://127.0.0.1:{port}/obsidian-memory-graph.html",
        "http://127.0.0.1:9120/memory-graph/obsidian-memory-graph.html",
    ]
    if ip:
        urls.insert(0, f"http://{ip}:{port}/obsidian-memory-graph.html")
    ts = tailscale_magic_dns()
    if ts:
        urls.insert(0, f"https://{ts}/memory-graph/obsidian-memory-graph.html")
    return urls


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Obsidian Memory Graph 3D — Cyberpunk VR</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap');
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; background: #030308; color: #00fff9; font-family: 'JetBrains Mono', monospace; overflow: hidden; }
    #hud {
      position: fixed; top: 0; left: 0; right: 0; z-index: 20;
      padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center;
      background: linear-gradient(180deg, rgba(3,3,10,0.96) 0%, rgba(3,3,10,0.55) 100%);
      border-bottom: 1px solid #ff00ff55;
      pointer-events: auto;
    }
    #hud h1 { font-family: 'Orbitron', sans-serif; font-size: 0.9rem; color: #ff00ff; text-shadow: 0 0 10px #ff00ff; letter-spacing: 0.1em; }
    #stats { font-size: 0.68rem; color: #7fffd4; max-width: 42vw; line-height: 1.35; }
    #graph { position: fixed; inset: 0; z-index: 1; }
    #filter, button.cyber {
      background: #0a0a14; border: 1px solid #00fff955; color: #00fff9;
      padding: 5px 10px; font-family: inherit; font-size: 0.68rem; cursor: pointer;
    }
    button.cyber { border-color: #ff00ff; color: #ff66ff; font-family: 'Orbitron', sans-serif; letter-spacing: 0.06em; }
    button.cyber:hover { background: #ff00ff18; box-shadow: 0 0 10px #ff00ff44; }
    button.cyber.vr { border-color: #39ff14; color: #39ff14; }
    button.cyber.vr:hover { background: #39ff1418; box-shadow: 0 0 10px #39ff1444; }
    .legend { display: flex; flex-wrap: wrap; gap: 6px; font-size: 0.62rem; }
    .legend span { padding: 1px 6px; border: 1px solid; border-radius: 2px; }
    #vr-hint {
      position: fixed; bottom: 12px; left: 12px; right: 12px; z-index: 20;
      font-size: 0.62rem; color: #888; background: #0a0a14cc; padding: 8px 12px;
      border: 1px solid #333; pointer-events: none;
    }
    a { color: #00fff9; }
  </style>
</head>
<body>
  <div id="hud">
    <h1>◈ Memory Nexus 3D</h1>
    <div id="stats"></div>
    <select id="filter" title="Filter cluster">
      <option value="all">ALL</option>
      <option value="concepts">CONCEPTS</option>
      <option value="sessions">SESSIONS</option>
      <option value="memory">MEMORY</option>
      <option value="box">BOX</option>
      <option value="tag">TAGS</option>
      <option value="raw">RAW</option>
    </select>
    <select id="edge-filter" title="Edge type">
      <option value="all">ALL EDGES</option>
      <option value="wikilink">WIKILINK</option>
      <option value="tag">#TAG</option>
      <option value="tag-cooc">TAG CO-OCCUR</option>
    </select>
    <button class="cyber" id="fit">FIT</button>
    <button class="cyber" id="pause">PAUSE</button>
    <button class="cyber vr" id="vr-btn">ENTER VR</button>
    <div class="legend">
      <span style="border-color:#ff00ff;color:#ff66ff">concept</span>
      <span style="border-color:#00fff9;color:#7fffd4">session</span>
      <span style="border-color:#39ff14;color:#b8ffb8">box</span>
      <span style="border-color:#ffd700;color:#ffd700">#tag</span>
    </div>
  </div>
  <div id="graph"></div>
  <div id="vr-hint">VR (Quest/VIVE/HMD): open on LAN — <span id="quest-url">…</span> · WebXR needs HTTP(S), not file://</div>

  <script type="module">
    import ForceGraph3D from 'https://esm.sh/3d-force-graph@1.73.0';
    import { VRButton } from 'https://esm.sh/three@0.160.0/examples/jsm/webxr/VRButton.js';

    const payload = __GRAPH_JSON__;
    const fullData = {
      nodes: payload.nodes.map(n => ({ ...n })),
      links: payload.links.map(l => ({ ...l }))
    };

    const statsEl = document.getElementById('stats');
    const vaults = payload.meta.vaults.join(' · ');
    statsEl.textContent = `${vaults} | ${payload.nodes.length} nodes | ${payload.links.length} edges | ${payload.meta.generated_at}`;

    const questEl = document.getElementById('quest-url');
    const vrList = (payload.meta.vr_urls || []).join(' · ');
    questEl.textContent = vrList || location.href;

    let paused = false;
    const elem = document.getElementById('graph');

    const Graph = new ForceGraph3D(elem)
      .backgroundColor('#030308')
      .showNavInfo(false)
      .nodeLabel(n => `<div style="font-family:monospace;color:#00fff9"><b>${n.name}</b><br/><span style="color:#aaa">${n.snippet || ''}</span></div>`)
      .nodeColor(n => n.color)
      .nodeVal(n => n.val)
      .nodeOpacity(0.92)
      .linkColor(l => l.color)
      .linkWidth(l => l.kind === 'tag-cooc' ? 0.35 : (l.kind === 'tag' ? 0.6 : 1.2))
      .linkDirectionalArrowLength(l => l.kind === 'wikilink' ? 3.5 : 0)
      .linkDirectionalArrowRelPos(0.85)
      .linkDirectionalParticles(l => l.kind === 'wikilink' ? 2 : 0)
      .linkDirectionalParticleWidth(1.2)
      .linkDirectionalParticleSpeed(0.006)
      .warmupTicks(120)
      .cooldownTicks(200)
      .graphData(fullData);

    Graph.d3Force('charge').strength(-140);
    Graph.d3Force('link').distance(l => l.kind === 'tag-cooc' ? 28 : (l.kind === 'tag' ? 40 : 70));

    const renderer = Graph.renderer();
    renderer.xr.enabled = true;
    const vrBtn = VRButton.createButton(renderer);
    vrBtn.id = 'webxr-vr';
    vrBtn.style.position = 'fixed';
    vrBtn.style.bottom = '14px';
    vrBtn.style.right = '14px';
    vrBtn.style.zIndex = '30';
    document.body.appendChild(vrBtn);

    document.getElementById('vr-btn').onclick = () => vrBtn.click();

    document.getElementById('fit').onclick = () => {
      const dist = Graph.cameraPosition().z;
      Graph.cameraPosition({ z: dist }, null, 800);
    };

    document.getElementById('pause').onclick = (e) => {
      paused = !paused;
      Graph.d3Force('charge').strength(paused ? 0 : -140);
      e.target.textContent = paused ? 'RESUME' : 'PAUSE';
    };

    function applyFilters() {
      const g = document.getElementById('filter').value;
      const ek = document.getElementById('edge-filter').value;
      let nodes = fullData.nodes;
      let links = fullData.links;
      if (g !== 'all') {
        nodes = nodes.filter(n => n.group === g);
      }
      const ids = new Set(nodes.map(n => n.id));
      links = links.filter(l => {
        const sid = typeof l.source === 'object' ? l.source.id : l.source;
        const tid = typeof l.target === 'object' ? l.target.id : l.target;
        if (!ids.has(sid) || !ids.has(tid)) return false;
        return ek === 'all' || l.kind === ek;
      });
      Graph.graphData({ nodes, links });
    }

    document.getElementById('filter').onchange = applyFilters;
    document.getElementById('edge-filter').onchange = applyFilters;

    if (!navigator.xr) {
      document.getElementById('vr-btn').disabled = true;
      document.getElementById('vr-btn').title = 'WebXR not available in this browser';
    }
  </script>
</body>
</html>
"""


def render_html(vault_labels: list[str], nodes: list[dict], links: list[dict], out: Path) -> None:
    meta = {
        "vaults": vault_labels,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "node_count": len(nodes),
        "edge_count": len(links),
        "vr_urls": vr_access_urls(),
    }
    payload = {"meta": meta, "nodes": nodes, "links": links}
    html = HTML_TEMPLATE.replace("__GRAPH_JSON__", json.dumps(payload, ensure_ascii=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Obsidian multi-vault 3D graph (wikilinks + #tags, WebXR VR)"
    )
    parser.add_argument(
        "--vault",
        action="append",
        dest="vaults",
        help="Vault path (repeat for multiple; default: all open Obsidian vaults)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/obsidian-memory-graph.html"),
        help="Output HTML path",
    )
    args = parser.parse_args(argv)
    vaults = resolve_vaults(args.vaults)
    nodes, links, labels = scan_vaults(vaults)
    out = args.output.resolve()
    render_html(labels, nodes, links, out)
    print(f"Wrote {out}")
    print(f"Vaults: {len(vaults)}  Nodes: {len(nodes)}  Edges: {len(links)}")
    tag_edges = sum(1 for e in links if e.get("kind") == "tag")
    cooc_edges = sum(1 for e in links if e.get("kind") == "tag-cooc")
    wiki_edges = sum(1 for e in links if e.get("kind") == "wikilink")
    print(f"  wikilink: {wiki_edges}  #tag: {tag_edges}  tag-cooc: {cooc_edges}")
    for url in vr_access_urls():
        print(f"  Quest/VR: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

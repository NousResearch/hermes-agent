#!/usr/bin/env bash
# antseed-smart-delegate/models.sh — List live models from AntSeed network
# Usage: bash models.sh [--json] [--top N]
set -uo pipefail

ANTSEED_BIN="$(command -v antseed 2>/dev/null || echo /home/ubuntu/.hermes/node/bin/antseed)"
PROXY_URL="http://127.0.0.1:8377"
JSON_ONLY=false
TOP=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json) JSON_ONLY=true; shift ;;
    --top)  TOP="$2"; shift 2 ;;
    *)      shift ;;
  esac
done

# Write Python script to temp file to avoid bash escaping hell
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

cat > "$TMPDIR/models.py" << 'PYEOF'
import subprocess, json, sys, re, os
import urllib.request

antseed = os.environ.get("ANTSEED_BIN", "antseed")
proxy_url = os.environ.get("PROXY_URL", "http://127.0.0.1:8377")
top = int(os.environ.get("TOP", "30"))
json_only = os.environ.get("JSON_ONLY", "0") == "1"

# Category detection from tags (generic tag→group mapping, NOT hardcoded models)
TAG_CATEGORIES = [
    (["reasoning", "research"], "Reasoning / Research"),
    (["coding", "code"], "Coding"),
    (["vision", "multimodal"], "Vision / Multimodal"),
    (["fast"], "Fast / Chat"),
    (["cheap", "free", "anon"], "Cheap / Free"),
    (["privacy", "tee"], "Privacy / TEE"),
    (["web-search"], "Web Search"),
    (["translate", "math"], "Specialist"),
    (["premium", "agents", "tasks"], "Premium / Agents"),
]

def categorize(tags_str):
    tset = set(t.strip().lower() for t in tags_str.split(",") if t.strip())
    for tag_list, label in TAG_CATEGORIES:
        if any(t in tset for t in tag_list):
            return label
    return "General"

def get_peers():
    try:
        raw = subprocess.run([antseed, "network", "browse", "--top", str(top)],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        raw = ""
    peers = []
    for line in raw.split("\n"):
        m = re.search(r"([0-9a-fA-F]{40,})", line)
        if m:
            pid = m.group(1)
            parts = re.split(r"[│┃]", line)
            name = parts[2].strip() if len(parts) >= 3 else "unknown"
            peers.append((pid, name))
    return peers

def get_peer_services(pid, pname):
    try:
        raw = subprocess.run([antseed, "network", "peer", pid],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    services = []
    for line in raw.split("\n"):
        line = line.strip()
        if "protocols:" not in line and "tags:" not in line:
            continue
        if " in " not in line:
            continue
        tokens = line.split()
        if not tokens:
            continue
        model = tokens[0]

        in_m = re.search(r"\bin\s+\$?([\d.]+)", line)
        out_m = re.search(r"\bout\s+\$?([\d.]+)", line)
        pin = in_m.group(1) if in_m else "0"
        pout = out_m.group(1) if out_m else "0"

        proto_m = re.search(r"protocols?:\s*([\w-]+)", line)
        proto = proto_m.group(1) if proto_m else "openai-chat-completions"

        tag_m = re.search(r"tags?:\s*([\w,\-./ ]+?)(?:\s{2,}|$)", line)
        tags = tag_m.group(1).strip().replace(" ", "") if tag_m else ""

        services.append({
            "peer_id": pid, "peer_name": pname, "model": model,
            "price_in": pin, "price_out": pout,
            "protocol": proto, "tags": tags
        })
    return services

def get_proxy_models():
    try:
        req = urllib.request.Request(f"{proxy_url}/v1/models",
                                     headers={"Authorization": "Bearer antseed-p2p"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return [{"peer_id": "proxy", "peer_name": "Buyer Proxy",
                 "model": m["id"], "price_in": "0", "price_out": "0",
                 "protocol": "openai-chat-completions", "tags": ""}
                for m in data.get("data", [])]
    except Exception:
        return []

# Collect all models
all_models = []
for pid, pname in get_peers():
    all_models.extend(get_peer_services(pid, pname))

if not all_models:
    all_models = get_proxy_models()

if not all_models:
    print(json.dumps({"error": "No models found from network or proxy"}))
    sys.exit(2)

# Deduplicate (keep cheapest per model)
seen = {}
for svc in all_models:
    mid = svc["model"]
    if mid not in seen or float(svc.get("price_in", "999")) < float(seen[mid].get("price_in", "999")):
        seen[mid] = svc
unique = list(seen.values())

# Group by category
cats = {}
for svc in unique:
    cat = categorize(svc.get("tags", ""))
    cats.setdefault(cat, []).append(svc)

# Sort categories by count descending
sorted_cats = sorted(cats.items(), key=lambda x: -len(x[1]))

# Build output
output = {
    "total_models": len(unique),
    "total_peers": len(set(s["peer_id"] for s in all_models)),
    "categories": {}
}
for cat, models in sorted_cats:
    output["categories"][cat] = {
        "count": len(models),
        "models": [{"model": m["model"], "peer_name": m["peer_name"],
                     "price_in": m["price_in"], "price_out": m["price_out"],
                     "protocol": m["protocol"], "tags": m.get("tags", "")}
                    for m in sorted(models, key=lambda x: float(x.get("price_in", "999")))]
    }

json_out = json.dumps(output, indent=2)

if json_only:
    print(json_out)
else:
    print(json_out)
    print(f"\n📦 {len(unique)} models from {output['total_peers']} peers:", file=sys.stderr)
    for cat, info in output["categories"].items():
        peers_in_cat = set(m["peer_name"] for m in info["models"])
        print(f"  {cat} ({info['count']} models, {len(peers_in_cat)} peers)", file=sys.stderr)
        for m in info["models"][:6]:
            tag_str = f" [{m['tags']}]" if m["tags"] else ""
            print(f"    • {m['model']} — {m['peer_name']} (${m['price_in']}/${m['price_out']}){tag_str}", file=sys.stderr)
        if len(info["models"]) > 6:
            print(f"    ... +{len(info['models'])-6} more", file=sys.stderr)
    print("", file=sys.stderr)
PYEOF

ANTSEED_BIN="$ANTSEED_BIN" PROXY_URL="$PROXY_URL" TOP="$TOP" \
  JSON_ONLY=$( [[ "$JSON_ONLY" == true ]] && echo 1 || echo 0 ) \
  python3 "$TMPDIR/models.py"

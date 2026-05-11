#!/usr/bin/env bash
# antseed-smart-delegate/best-peer.sh — Find optimal AntSeed peer+model for a task type
# Usage: bash best-peer.sh <task_type> [--json] [--peer <peer-id>]
#   task_type: code | research | vision | chat | cheap | any
# Output: JSON to stdout, human-readable summary to stderr
set -uo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ANTSEED_BIN="$(command -v antseed 2>/dev/null || echo /home/ubuntu/.hermes/node/bin/antseed)"
PROXY_URL="http://127.0.0.1:8377"

TASK_TYPE="${1:-any}"
JSON_ONLY=false
TARGET_PEER=""

shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --json) JSON_ONLY=true; shift ;;
    --peer) TARGET_PEER="$2"; shift 2 ;;
    *)      shift ;;
  esac
done

# Write Python script to temp file to avoid bash escaping hell
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

cat > "$TMPDIR/best_peer.py" << 'PYEOF'
import subprocess, json, sys, re, os

antseed = os.environ.get("ANTSEED_BIN", "antseed")
proxy_url = os.environ.get("PROXY_URL", "http://127.0.0.1:8377")
task_type = os.environ.get("TASK_TYPE", "any")
target_peer = os.environ.get("TARGET_PEER", "")
json_only = os.environ.get("JSON_ONLY", "0") == "1"

# Which tags are desirable for each task type
TASK_TAGS = {
    "research": ["reasoning", "research"],
    "code":     ["coding", "code", "reasoning"],
    "vision":   ["vision", "multimodal"],
    "chat":     ["chat", "fast"],
    "cheap":    ["cheap", "free", "anon"],
    "any":      [],
}

desired = set(TASK_TAGS.get(task_type, []))

def get_peers():
    try:
        raw = subprocess.run([antseed, "network", "browse", "--top", "30"],
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
    if target_peer:
        peers = [(p, n) for p, n in peers if target_peer in p]
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
        pin = float(in_m.group(1)) if in_m else 999.0
        pout = float(out_m.group(1)) if out_m else 999.0

        proto_m = re.search(r"protocols?:\s*([\w-]+)", line)
        proto = proto_m.group(1) if proto_m else "openai-chat-completions"

        tag_m = re.search(r"tags?:\s*([\w,\-./ ]+?)(?:\s{2,}|$)", line)
        tags = set(t.strip().lower() for t in tag_m.group(1).split(",") if t.strip()) if tag_m else set()

        is_free = (pin == 0 and pout == 0) or "free" in tags

        services.append({
            "peer_id": pid, "peer_name": pname, "model": model,
            "price_in": pin, "price_out": pout, "protocol": proto,
            "tags": tags, "is_free": is_free
        })
    return services

def score(svc, desired):
    s = 0
    if desired:
        overlap = len(svc["tags"] & desired)
        s += overlap * 15
    else:
        s += min(len(svc["tags"]), 5) * 3
    if svc["is_free"]:
        s += 20
        if "cheap" in desired:
            s += 30
    if "chat-completions" in svc["protocol"]:
        s += 10
    s -= int(min(svc["price_in"], 20))
    return max(s, 0)

# Collect candidates
candidates = []
for pid, pname in get_peers():
        for svc in get_peer_services(pid, pname):
            svc["score"] = score(svc, desired)
            candidates.append(svc)

# Fallback to proxy if no peers
if not candidates:
    try:
        import urllib.request
        req = urllib.request.Request(f"{proxy_url}/v1/models",
                                     headers={"Authorization": "Bearer antseed-p2p"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        for m in data.get("data", []):
            candidates.append({
                "peer_id": "proxy", "peer_name": "Buyer Proxy",
                "model": m["id"], "price_in": 0, "price_out": 0,
                "protocol": "openai-chat-completions", "tags": set(),
                "is_free": False, "score": 0
            })
    except Exception:
        pass

if not candidates:
    print(json.dumps({"error": "No peers or models found", "task_type": task_type}))
    sys.exit(2)

# Sort by score desc, then price asc
candidates.sort(key=lambda x: (-x["score"], x["price_in"]))

best = candidates[0]
alts = candidates[1:6]
fallback = list(dict.fromkeys(c["peer_id"] for c in candidates))[:5]

def fmtags(tags):
    return ",".join(sorted(tags)) if isinstance(tags, set) else str(tags)

output = {
    "task_type": task_type,
    "total_candidates": len(candidates),
    "unique_peers": len(set(c["peer_id"] for c in candidates)),
    "recommended": {
        "peer_id": best["peer_id"],
        "peer_name": best["peer_name"],
        "model": best["model"],
        "price_in": f"${best['price_in']}/1M",
        "price_out": f"${best['price_out']}/1M",
        "protocol": best["protocol"],
        "tags": fmtags(best["tags"]),
        "free": best["is_free"],
        "score": best["score"]
    },
    "alternatives": [{
        "peer_id": a["peer_id"], "peer_name": a["peer_name"],
        "model": a["model"],
        "price_in": f"${a['price_in']}/1M",
        "tags": fmtags(a["tags"]),
        "free": a["is_free"]
    } for a in alts],
    "fallback_chain": fallback
}

json_out = json.dumps(output, indent=2)
print(json_out)

if not json_only:
    free_str = " ✨ FREE!" if best["is_free"] else ""
    print(f'\n🐝 Best for "{task_type}": {best["peer_name"]}', file=sys.stderr)
    print(f'   Model: {best["model"]} (${best["price_in"]}/${best["price_out"]} per 1M){free_str}', file=sys.stderr)
    print(f'   Protocol: {best["protocol"]}', file=sys.stderr)
    print(f'   Tags: {fmtags(best["tags"])}', file=sys.stderr)
    print(f'   Alternatives: {len(alts)} more | Fallback chain: {len(fallback)} peers', file=sys.stderr)
PYEOF

ANTSEED_BIN="$ANTSEED_BIN" PROXY_URL="$PROXY_URL" TASK_TYPE="$TASK_TYPE" \
  TARGET_PEER="$TARGET_PEER" JSON_ONLY=$( [[ "$JSON_ONLY" == true ]] && echo 1 || echo 0 ) \
  python3 "$TMPDIR/best_peer.py"

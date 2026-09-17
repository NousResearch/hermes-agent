"""Suggestion-arm harness: same 47 requests as skill_baseline.py, but each request is
preceded by the plugin's <skill_relevance> block (as pre_llm_call context would
inject it). Measures end-to-end improvement over the 20.0%/33.3% baseline.

Suggestion via the plugin's real suggest() (deepseek-v4-flash judge), cached to
/tmp/suggestion_arm_results.jsonl; agent turns cached to
/tmp/agent_arm_results.jsonl (resumable, independent cache key).
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "/home/duncan/.hermes/hermes-agent")

BASE_URL = "http://127.0.0.1:11434/v1"
MODEL = "glm-5.3:cloud"
API_KEY = "ollama"
MAX_TOKENS = 3000
TIMEOUT = 300
SUGGESTIONS = Path(__file__).with_name("suggestion_arm_suggestions.jsonl")
AGENT_TURNS = Path(__file__).with_name("suggestion_arm_results.jsonl")

sys.path.insert(0, str(Path(__file__).parent))
from skill_baseline import COVERED, NEGATIVES, TOOLS, request_hash, first_skill_view  # noqa: E402

TESTS = [(r, g) for r, g in COVERED] + [(r, None) for r in NEGATIVES]


class FakeCtx:
    plugin_id = "skill-suggest"

    def get_config(self, key, default=None):
        from hermes_cli.config import load_config_readonly
        entry = (load_config_readonly() or {}).get("plugins", {}).get("entries", {}).get("skill-suggest", {})
        return entry.get("settings", {}).get(key, default)

    @property
    def llm(self):
        from agent.plugin_llm import PluginLlm
        return PluginLlm(plugin_id="skill-suggest")


def suggest_for(request):
    """Plugin's real suggest(); returns block or None."""
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("skill_suggest", "/home/duncan/.hermes/plugins/skill-suggest/__init__.py")
    ss = ilu.module_from_spec(spec)
    spec.loader.exec_module(ss)
    ss._CACHE.clear()
    try:
        return ss.suggest(FakeCtx(), request)
    except Exception as e:  # noqa: BLE001
        print(f"  suggest() exc: {e}")
        return None


def agent_turn(system, user):
    payload = {"model": MODEL, "max_tokens": MAX_TOKENS, "tools": TOOLS,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    req = urllib.request.Request(BASE_URL + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        body = json.loads(resp.read().decode())
    return body, round(time.perf_counter() - t0, 1)


def main():
    from agent.prompt_builder import build_skills_system_prompt
    system = build_skills_system_prompt(
        available_tools={"skill_view", "skills_list", "skill_manage", "terminal", "read_file",
                         "write_file", "patch", "search_files", "web_search", "web_extract"})
    print(f"system prompt: {len(system):,} chars", flush=True)

    # Pass 1: suggestions (cached)
    s_done = {}
    if SUGGESTIONS.exists():
        for line in SUGGESTIONS.read_text().splitlines():
            rec = json.loads(line)
            s_done[rec["h"]] = rec["block"]
    print(f"suggestions cached: {len(s_done)}", flush=True)
    for i, (text, _g) in enumerate(TESTS):
        h = request_hash(text)
        if h in s_done:
            continue
        print(f"  [{i+1}/47] suggesting for: {text[:50]}", flush=True)
        block = suggest_for(text)
        s_done[h] = block
        with SUGGESTIONS.open("a") as f:
            f.write(json.dumps({"h": h, "request": text, "block": block}) + "\n")

    # Pass 2: agent turns with suggestion prepended (cached, separate arm key)
    a_done = {}
    if AGENT_TURNS.exists():
        for line in AGENT_TURNS.read_text().splitlines():
            rec = json.loads(line)
            a_done[rec["h"]] = rec
    print(f"agent turns cached: {len(a_done)}", flush=True)
    for i, (text, gold) in enumerate(TESTS):
        h = request_hash(text)
        if h in a_done:
            continue
        block = s_done[h]
        user = (block + "\n\n" + text) if block else text
        body, secs = agent_turn(system, user)
        loaded = first_skill_view(body)
        rec = {"h": h, "request": text, "gold": gold, "loaded": loaded, "secs": secs,
               "block": block}
        a_done[h] = rec
        with AGENT_TURNS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        tag = "OK" if (loaded == gold if gold else loaded is None) else "MISS"
        print(f"  [{tag}] {secs:>4}s  gold={gold!r:24} loaded={loaded!r:22} sug={bool(block)}  {text[:40]}", flush=True)

    # Score
    pos = [r for r in a_done.values() if r["gold"]]
    neg = [r for r in a_done.values() if not r["gold"]]
    wrong = [r for r in pos if r["loaded"] != r["gold"]]
    needless = [r for r in neg if r["loaded"] is not None]
    sug_pos = sum(1 for r in pos if r["block"])
    sug_neg = sum(1 for r in neg if r["block"])
    print(f"\n=== SUGGESTION ARM RESULTS ===")
    print(f"covered: {len(pos)}  uncovered: {len(neg)}")
    print(f"suggestions fired: {sug_pos}/{len(pos)} covered, {sug_neg}/{len(neg)} uncovered")
    print(f"wrong_load:    {len(wrong)}/{len(pos)} = {len(wrong)/len(pos):.1%}   (baseline 20.0%)")
    print(f"needless_load: {len(needless)}/{len(neg)} = {len(needless)/len(neg):.1%}   (baseline 33.3%)")
    print(f"\nwrong loads:")
    for r in wrong:
        print(f"  gold={r['gold']!r:26} got={r['loaded']!r} sug={r['block'] is not None}")
    print(f"needless loads:")
    for r in needless:
        print(f"  loaded={r['loaded']!r:20} sug={r['block'] is not None}  {r['request'][:50]}")


if __name__ == "__main__":
    main()
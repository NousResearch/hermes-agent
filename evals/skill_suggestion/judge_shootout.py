"""Judge shootout: run the plugin's suggest() across candidate judge models."""
import sys
import time

sys.path.insert(0, "/home/duncan/.hermes/hermes-agent")
import importlib.util as ilu

spec = ilu.spec_from_file_location("skill_suggest", "/home/duncan/.hermes/plugins/skill-suggest/__init__.py")
ss = ilu.module_from_spec(spec)
spec.loader.exec_module(ss)

import json
from hermes_cli.config import load_config_readonly


def make_ctx(judge_model):
    entry = (load_config_readonly() or {}).get("plugins", {}).get("entries", {}).get("skill-suggest", {})
    settings = dict(entry.get("settings", {}))
    settings["judge_model"] = judge_model
    settings["judge_provider"] = "gemini" if judge_model.startswith("gemini") else "ollama-launch"

    class Ctx:
        plugin_id = "skill-suggest"

        def get_config(self, key, default=None):
            return settings.get(key, default)

        @property
        def llm(self):
            from agent.plugin_llm import PluginLlm
            return PluginLlm(plugin_id="skill-suggest")

    return Ctx()


CASES = [
    ("Put together a pitch deck skeleton as a .pptx using our firm-template.pptx for branding.", "powerpoint|pptx-author"),
    ("Where is my iPad right now according to Find My?", "findmy"),
    ("Send an email from the terminal via himalaya and attach the report.", "himalaya"),
    ("Post this announcement to my Mastodon account.", None),
    ("Add these three cards to our Trello backlog.", None),
    ("Explain what a monad is in functional programming.", None),
]

MODELS = ["deepseek-v4-flash:cloud", "gemini-2.5-flash", "gemini-3.6-flash"]

for model in MODELS:
    print(f"\n=== judge: {model} ===")
    ctx = make_ctx(model)
    correct, total, t0 = 0, len(CASES), time.perf_counter()
    for msg, gold in CASES:
        ss._CACHE.clear()
        try:
            block = ss.suggest(ctx, msg)
        except Exception as e:  # noqa: BLE001
            block = None
            print(f"  EXC {type(e).__name__}: {e}")
        pick = "-"
        if block:
            import re
            m = re.search(r"Relevant to the current request: ([^.]+)\.", block)
            pick = m.group(1) if m else "?"
        if gold is None:
            ok = pick == "-"
        else:
            ok = pick in gold.split("|")
        correct += ok
        print(f"  [{'OK ' if ok else 'MISS'}] pick={pick:<12} gold={gold}  {msg[:48]}")
    dt = time.perf_counter() - t0
    print(f"  -> {correct}/{total} correct, {dt/total:.1f}s/case")
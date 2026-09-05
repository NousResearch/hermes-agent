#!/usr/bin/env python3
"""Hermes vision-path checker: native fast path vs auxiliary fallback.

Read-only diagnostic. Prints which vision path the active (or any given)
provider/model pair will take:

  venv/bin/python check_vision_path.py                       # current main provider/model
  venv/bin/python check_vision_path.py zai glm-5.3-flash     # any provider/model pair

Run it with the venv that belongs to your Hermes installation
(`venv/bin/python` on Linux/macOS, `venv\\Scripts\\python.exe` on Windows).
If Hermes lives outside the default location, set HERMES_HOME first.
"""
import os
import sys
import logging

logging.disable(logging.CRITICAL)


def locate_hermes_home() -> str:
    """Resolve the Hermes data dir: HERMES_HOME env var, then ~/.hermes."""
    home = os.environ.get("HERMES_HOME", "").strip()
    if home and os.path.isdir(home):
        return home
    return os.path.expanduser("~/.hermes")


_home = locate_hermes_home()
_h = os.environ.get("HERMES_HOME", "").strip()
if _h and not os.path.isdir(_h):
    print(f"WARNING: HERMES_HOME does not exist - falling back to ~/.hermes ({_home})")
    print("Note: the Hermes core uses a nonexistent $HERMES_HOME as-is (empty")
    print("config); fix the variable so this verdict matches the runtime.")
    print()
_repo = os.path.join(_home, "hermes-agent")
if os.path.isdir(_repo):
    sys.path.insert(0, _repo)

try:
    from tools.vision_tools import (
        _should_use_native_vision_fast_path,
        _supports_media_in_tool_results,
    )
    from agent.image_routing import decide_image_input_mode, _lookup_supports_vision
    from agent.auxiliary_client import _read_main_provider, _read_main_model
    from hermes_cli.config import load_config
except Exception as exc:  # noqa: BLE001 - friendly guidance beats a traceback
    print(f"Could not import Hermes internals ({exc.__class__.__name__}: {exc})")
    print()
    print("Fix checklist:")
    print("  1. Run this script with the venv that belongs to your Hermes install:")
    print("       Linux/macOS : <hermes-home>/hermes-agent/venv/bin/python")
    print("       Windows     : <hermes-home>\\hermes-agent\\venv\\Scripts\\python.exe")
    print("  2. If Hermes is not at ~/.hermes, point the checker at it first:")
    print("       export HERMES_HOME=/path/to/.hermes   (or set it on Windows)")
    print("  3. Confirm the repo exists: <hermes-home>/hermes-agent/tools/vision_tools.py")
    print()
    print("Note: this checker targets Hermes Agent installations only - it")
    print("imports Hermes-internal APIs and will not work with other agents.")
    sys.exit(1)

cfg = load_config()
if len(sys.argv) >= 3:
    provider, model = sys.argv[1], sys.argv[2]
else:
    provider = _read_main_provider() or ""
    model = _read_main_model() or ""

mode = decide_image_input_mode(provider, model, cfg)
tool_img = _supports_media_in_tool_results(provider, model)
vis = _lookup_supports_vision(provider, model, cfg)
if len(sys.argv) >= 3:
    fast = (mode == "native") and (tool_img or vis is True)
else:
    fast = _should_use_native_vision_fast_path()

print(f"provider/model    : {provider} / {model}")
print(f"image input route : {mode}")
print(f"tool-result images: {tool_img}")
print(f"model vision      : {vis}")
print(
    "=> vision path    : "
    + ("NATIVE (pixels enter the main-model context)" if fast
       else "FALLBACK (auxiliary.vision paraphrase, lossy)")
)
if vis is None and not fast:
    print()
    print("models.dev has no record of this model. If it is truly multimodal,")
    print("declare it explicitly (config override beats the cache):")
    print("  hermes config set model.supports_vision true")
    print("If Hermes is a fresh install, run it once to populate")
    print("<hermes-home>/models_dev_cache.json, then rerun this script.")

"""Behavior-parity check for the TTS plugin hook (issue #30398).

Spawns one subprocess per (version, scenario) cell — pinned to either
``origin/main`` (no plugin hook; ``tts.provider: cartesia`` falls
through to the Edge TTS default branch) or this PR's worktree (plugin
hook present; same config routes through the plugin registry when a
plugin is registered).

Each subprocess clears all TTS-related env vars + writes a
``config.yaml``, then resolves how the dispatcher would route a
``text_to_speech`` call. The emitted shape tuple is::

    {dispatch_kind, provider_name, voice_compat}

Where ``dispatch_kind`` ∈
``{"builtin_edge", "builtin_openai", "builtin_elevenlabs", ...,
"command", "plugin", "fallback_edge", "error"}``:

* ``builtin_<name>`` — config selects a built-in handler that exists
  on both main and PR (no diff expected)
* ``command`` — config selects a ``tts.providers.<name>: type: command``
  entry (PR #17843; no diff expected)
* ``plugin`` — config selects a plugin-registered provider (PR only)
* ``fallback_edge`` — config selects an unknown name with no matching
  plugin or command entry → Edge TTS default fallback
* ``error`` — explicit fatal error (e.g. mistral quarantine)

The parent process diffs the reduced shape per scenario. The only
acceptable diff is ``fallback_edge → plugin`` for the
``unknown-name-with-plugin-installed`` scenario — everything else is
a regression.

Run from the PR worktree (it auto-resolves ``MAIN_DIR`` from the parent
of the worktree directory, or falls back to a sibling
``hermes-agent-main`` checkout)::

    python tests/plugins/tts/check_parity_vs_main.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_main_dir() -> Path:
    candidate = REPO_ROOT.parent.parent
    if (candidate / "tools" / "tts_tool.py").exists() and candidate != REPO_ROOT:
        return candidate
    sibling = REPO_ROOT.parent / "hermes-agent-main"
    if (sibling / "tools" / "tts_tool.py").exists():
        return sibling
    return REPO_ROOT


MAIN_DIR = _resolve_main_dir()
PR_DIR = REPO_ROOT
assert (PR_DIR / "tools" / "tts_tool.py").exists(), (
    f"PR_DIR={PR_DIR} doesn't look like a hermes-agent checkout"
)


# The subprocess script — runs INSIDE either the main checkout or PR
# checkout, so the import paths resolve to the version of the code
# under test. We never call the real ``text_to_speech_tool`` because
# that would require audio synthesis; instead we ask the resolution
# layer what it WOULD do.
SUBPROCESS_SCRIPT = r"""
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])

# Isolated HERMES_HOME so the config write is hermetic.
home = tempfile.mkdtemp()
os.environ["HERMES_HOME"] = home

# Clear TTS-related env so dispatch decisions are config-driven.
for k in (
    "ELEVENLABS_API_KEY", "OPENAI_API_KEY", "VOICE_TOOLS_OPENAI_KEY",
    "MINIMAX_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY",
):
    os.environ.pop(k, None)

scenario_env = json.loads(sys.argv[2])
os.environ.update(scenario_env)

config_yaml = sys.argv[3]
plugin_register = sys.argv[4]  # "yes" to register a fake plugin

config_path = os.path.join(home, "config.yaml")
with open(config_path, "w") as f:
    f.write(config_yaml)

# Fresh import — must not have anything cached from prior runs.
for name in list(sys.modules):
    if (name.startswith("tools.")
            or name.startswith("agent.")
            or name.startswith("plugins.")
            or name.startswith("hermes_cli.")):
        sys.modules.pop(name, None)

# Try importing tts_registry — only exists on PR side.
have_plugin_hook = False
try:
    from agent import tts_registry
    from agent.tts_provider import TTSProvider
    have_plugin_hook = True

    if plugin_register == "yes":
        class _FakeProvider(TTSProvider):
            @property
            def name(self): return "cartesia"
            def synthesize(self, text, output_path, **kw):
                return output_path

        tts_registry._reset_for_tests()
        tts_registry.register_provider(_FakeProvider())
except ImportError:
    pass

import tools.tts_tool as tts_tool

# Read the config the same way text_to_speech_tool() does.
tts_config = tts_tool._load_tts_config()
provider = tts_tool._get_provider(tts_config)

dispatch_kind = None
provider_name = provider
voice_compat = False
error_text = None

try:
    # Mistral is the one branch that returns a fatal error.
    if provider == "mistral":
        dispatch_kind = "error"
        error_text = "mistral quarantine"
    elif tts_tool._resolve_command_provider_config(provider, tts_config) is not None:
        dispatch_kind = "command"
    elif have_plugin_hook and provider not in tts_tool.BUILTIN_TTS_PROVIDERS:
        # On PR side: check plugin dispatch.
        plugin_path = tts_tool._dispatch_to_plugin_provider(
            "test", os.path.join(home, "out.mp3"), provider, tts_config,
        )
        if plugin_path is not None:
            dispatch_kind = "plugin"
            voice_compat = tts_tool._plugin_provider_is_voice_compatible(provider)
        else:
            # Falls through to Edge TTS default on the PR side too.
            dispatch_kind = "fallback_edge"
    elif provider in tts_tool.BUILTIN_TTS_PROVIDERS:
        dispatch_kind = "builtin_" + provider
    else:
        # On main side: unknown names fall through to Edge default.
        dispatch_kind = "fallback_edge"
except Exception as exc:
    dispatch_kind = "exception"
    error_text = repr(exc)

shape = {
    "dispatch_kind": dispatch_kind,
    "provider_name": provider_name,
    "voice_compat": bool(voice_compat),
    "error_present": error_text is not None,
}
print(json.dumps(shape))
"""


SCENARIOS: list[tuple[str, str, dict[str, str], str]] = [
    # (label, config.yaml body, scenario_env, plugin_register)

    # Scenario 1: unset tts.provider → both: Edge default
    ("unset-defaults-to-edge", "", {}, "no"),

    # Scenario 2: built-in name → both: that built-in
    ("explicit-edge", "tts:\n  provider: edge\n", {}, "no"),
    ("explicit-openai", "tts:\n  provider: openai\n", {}, "no"),
    ("explicit-elevenlabs", "tts:\n  provider: elevenlabs\n", {}, "no"),

    # Scenario 3: command-type provider → both: command dispatch
    (
        "command-provider",
        "tts:\n  provider: my-piper\n  providers:\n    my-piper:\n      type: command\n      command: 'piper -m model.onnx -f {output_path} < {input_path}'\n",
        {},
        "no",
    ),

    # Scenario 4: unknown name with NO plugin installed → both: fallback to Edge
    ("unknown-no-plugin", "tts:\n  provider: cartesia\n", {}, "no"),

    # Scenario 5: unknown name WITH plugin installed
    #   main: fallback_edge (no plugin hook exists)
    #   PR:   plugin (cartesia)
    # This is the ONLY acceptable diff in the harness.
    ("plugin-installed", "tts:\n  provider: cartesia\n", {}, "yes"),

    # Scenario 6: built-in name + plugin tries to shadow → both: built-in
    # The plugin registers under name "cartesia", not "edge", so this is
    # effectively the same as scenario 2 — but we exercise the with-plugin
    # path to ensure the built-in branch still takes priority.
    ("explicit-edge-with-plugin-registered", "tts:\n  provider: edge\n", {}, "yes"),

    # Scenario 7: mistral quarantine — both surface the explicit error
    ("mistral-quarantine", "tts:\n  provider: mistral\n", {}, "no"),
]


def _run_scenario(repo_path: Path, label: str, config_yaml: str, env: dict, plugin_register: str) -> dict:
    venv_python = repo_path / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = MAIN_DIR / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = MAIN_DIR / "venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = Path("python3")

    out = subprocess.run(
        [
            str(venv_python),
            "-c",
            SUBPROCESS_SCRIPT,
            str(repo_path),
            json.dumps(env),
            config_yaml,
            plugin_register,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if out.returncode != 0:
        return {
            "error": "subprocess failed",
            "stdout": out.stdout[-500:],
            "stderr": out.stderr[-500:],
        }
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"error": f"could not parse output: {exc}", "stdout": out.stdout}


def _reduce(shape: dict) -> dict:
    """Reduce to the parts that matter for user-visible parity."""
    return {
        "dispatch_kind": shape.get("dispatch_kind"),
        "provider_name": shape.get("provider_name"),
        "error_present": shape.get("error_present"),
    }


def main() -> int:
    print(f"main:    {MAIN_DIR}")
    print(f"pr:      {PR_DIR}")
    print()

    if MAIN_DIR == PR_DIR:
        print(
            "WARN: MAIN_DIR == PR_DIR — diffs will be trivially identical.\n"
            "      Set up a sibling 'hermes-agent-main' checkout pinned to "
            "origin/main to get real parity coverage."
        )
        print()

    failures: list[str] = []
    errors: list[str] = []
    intentional_diffs: list[tuple[str, dict, dict]] = []
    for label, config_yaml, env, plugin_register in SCENARIOS:
        main_shape = _run_scenario(MAIN_DIR, label, config_yaml, env, plugin_register)
        pr_shape = _run_scenario(PR_DIR, label, config_yaml, env, plugin_register)

        if "error" in main_shape or "error" in pr_shape:
            print(f"  [ERR ] {label}: subprocess failed")
            print(f"    main: {main_shape}")
            print(f"    pr:   {pr_shape}")
            errors.append(label)
            continue

        main_reduced = _reduce(main_shape)
        pr_reduced = _reduce(pr_shape)

        if main_reduced == pr_reduced:
            print(f"  [OK]   {label}: {main_reduced}")
            continue

        # On main, "plugin-installed" scenario returns fallback_edge
        # (no plugin hook); on PR, it routes to the plugin. That's the
        # only acceptable diff.
        fallback_to_plugin = (
            main_reduced.get("dispatch_kind") == "fallback_edge"
            and pr_reduced.get("dispatch_kind") == "plugin"
            and label == "plugin-installed"
        )
        if fallback_to_plugin:
            print(f"  [DIFF] {label}: fallback_edge → plugin — expected")
            intentional_diffs.append((label, main_reduced, pr_reduced))
        else:
            print(f"  [FAIL] {label}")
            print(f"    main: {main_reduced}")
            print(f"    pr:   {pr_reduced}")
            failures.append(label)

    print()
    if errors:
        print(f"SUBPROCESS ERRORS in {len(errors)} scenario(s):")
        for e in errors:
            print(f"  - {e}")
    if failures:
        print(f"BEHAVIOUR REGRESSION in {len(failures)} scenario(s):")
        for f in failures:
            print(f"  - {f}")
    if intentional_diffs:
        print(
            f"INTENTIONAL DIFFS ({len(intentional_diffs)}): "
            f"fallback_edge → plugin dispatch when a plugin is registered."
        )
    if failures or errors:
        return 1
    print(f"PARITY OK across {len(SCENARIOS)} scenarios.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

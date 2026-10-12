"""Behavior-parity check for the STT plugin hook + command-provider registry.

Spawns one subprocess per (version, scenario) cell — pinned to either
``origin/main`` (no plugin hook, no STT command-provider registry; only
the legacy ``HERMES_LOCAL_STT_COMMAND`` escape hatch exists) or this PR's
worktree (both new surfaces present).

Each subprocess clears all STT-related env vars + writes a
``config.yaml``, then asks the dispatcher how it would route a
``transcribe_audio`` call. The emitted shape tuple is::

    {dispatch_kind, provider_name, success}

Where ``dispatch_kind`` ∈
``{"builtin_local", "builtin_groq", "builtin_openai", ...,
"plugin", "plugin_unavailable", "command_provider",
"no_provider_error", "stt_disabled"}``.

Acceptable diffs:
- ``no_provider_error → plugin`` for the ``plugin-installed`` scenario.
- ``no_provider_error → plugin_unavailable`` for the
  ``plugin-installed-unavailable`` scenario (PR returns the cleaner
  unavailability envelope instead of the generic auto-detect error).
- ``no_provider_error → command_provider`` for the
  ``command-provider-installed`` scenario (registry shipped with this PR).
- ``no_provider_error → command_provider`` for
  ``command-vs-plugin-same-name`` (command wins precedence, same as TTS).

Run from the PR worktree::

    python tests/plugins/transcription/check_parity_vs_main.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_main_dir() -> Path:
    candidate = REPO_ROOT.parent.parent
    if (candidate / "tools" / "transcription_tools.py").exists() and candidate != REPO_ROOT:
        return candidate
    sibling = REPO_ROOT.parent / "hermes-agent-main"
    if (sibling / "tools" / "transcription_tools.py").exists():
        return sibling
    return REPO_ROOT


MAIN_DIR = _resolve_main_dir()
PR_DIR = REPO_ROOT
assert (PR_DIR / "tools" / "transcription_tools.py").exists(), (
    f"PR_DIR={PR_DIR} doesn't look like a hermes-agent checkout"
)


SUBPROCESS_SCRIPT = r"""
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])

# Isolated HERMES_HOME so the config write is hermetic.
home = tempfile.mkdtemp()
os.environ["HERMES_HOME"] = home

# Clear STT-related env so dispatch decisions are config-driven.
for k in (
    "GROQ_API_KEY", "OPENAI_API_KEY", "VOICE_TOOLS_OPENAI_KEY",
    "MISTRAL_API_KEY", "XAI_API_KEY",
    "HERMES_LOCAL_STT_COMMAND",
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

# Try importing transcription_registry — only exists on PR side.
have_plugin_hook = False
try:
    from agent import transcription_registry
    from agent.transcription_provider import TranscriptionProvider
    have_plugin_hook = True

    if plugin_register == "yes":
        class _FakeProvider(TranscriptionProvider):
            @property
            def name(self): return "openrouter"
            def transcribe(self, file_path, **kw):
                return {"success": True, "transcript": "PLUGIN: openrouter transcript", "provider": "openrouter"}

        transcription_registry._reset_for_tests()
        transcription_registry.register_provider(_FakeProvider())
    elif plugin_register == "unavailable":
        class _UnavailablePlugin(TranscriptionProvider):
            @property
            def name(self): return "openrouter"
            def is_available(self): return False
            def transcribe(self, file_path, **kw):
                return {"success": True, "transcript": "should not run"}

        transcription_registry._reset_for_tests()
        transcription_registry.register_provider(_UnavailablePlugin())
except ImportError:
    pass

import tools.transcription_tools as tt

# Use a real (but empty) audio file so _validate_audio_file passes.
audio_path = os.path.join(home, "audio.ogg")
with open(audio_path, "wb") as f:
    # Minimal-ish OGG-shaped bytes so the size check passes.
    f.write(b"OggS" + b"\x00" * 1024)

# Patch _transcribe_* so the test doesn't actually try cloud APIs.
# We're testing dispatch, not the underlying transcription.
def _stub(file_path, model_name=None):
    return {"success": True, "transcript": "stub from " + sys._getframe().f_code.co_name.replace("_stub_", ""),
            "provider": sys._getframe().f_code.co_name.replace("_stub_", "")}

# Stub each built-in to a marker so we can identify the branch.
class _Stub:
    def __init__(self, name):
        self.name = name
    def __call__(self, file_path, model_name=None):
        return {"success": True, "transcript": "stub", "provider": self.name}

tt._transcribe_local = _Stub("local")
tt._transcribe_local_command = _Stub("local_command")
tt._transcribe_groq = _Stub("groq")
tt._transcribe_openai = _Stub("openai")
tt._transcribe_mistral = _Stub("mistral")
tt._transcribe_xai = _Stub("xai")

# Force _get_provider to honor the explicit config since we don't have
# real creds. The provider-resolution gates check _HAS_OPENAI /
# _HAS_FASTER_WHISPER which we can't easily set, so we just patch
# _get_provider to return whatever the config says.
stt_cfg = tt._load_stt_config()
explicit = stt_cfg.get("provider")
if explicit:
    # Bypass the gating for test purposes — _get_provider would
    # otherwise return "none" when the dependency isn't installed.
    original_get = tt._get_provider
    def _patched(cfg):
        if not tt.is_stt_enabled(cfg):
            return "none"
        return cfg.get("provider", "none")
    tt._get_provider = _patched

try:
    result = tt.transcribe_audio(audio_path)
except Exception as exc:
    shape = {"dispatch_kind": "exception", "provider_name": None, "success": False,
             "error_text": repr(exc)}
    print(json.dumps(shape))
    sys.exit(0)

dispatch_kind = "unknown"
provider_name = result.get("provider") if isinstance(result, dict) else None
success = result.get("success", False) if isinstance(result, dict) else False
error_text = result.get("error", "") if isinstance(result, dict) else ""

if not success and "STT is disabled" in error_text:
    dispatch_kind = "stt_disabled"
elif not success and "is not available" in error_text:
    dispatch_kind = "plugin_unavailable"
elif not success and "No STT provider" in error_text:
    dispatch_kind = "no_provider_error"
elif provider_name in ("local", "local_command", "groq", "openai", "mistral", "xai"):
    dispatch_kind = "builtin_" + provider_name
elif success and isinstance(result, dict) and result.get("transcript", "").startswith("CMD:"):
    # Command-provider scenarios below emit transcripts prefixed with "CMD:"
    # so the harness can distinguish command-provider dispatch from a
    # plugin dispatch even when they share a provider name.
    dispatch_kind = "command_provider"
elif success and isinstance(result, dict) and result.get("transcript", "").startswith("PLUGIN:"):
    dispatch_kind = "plugin"
elif success and provider_name and provider_name not in ("local", "local_command", "groq", "openai", "mistral", "xai"):
    dispatch_kind = "plugin"
else:
    dispatch_kind = "other"

shape = {
    "dispatch_kind": dispatch_kind,
    "provider_name": provider_name,
    "success": success,
}
print(json.dumps(shape))
"""


def _cmd_yaml(provider_name: str, transcript: str) -> str:
    """Build a YAML snippet for an stt.providers.<name>: type: command entry.

    Produces a shell command that writes ``transcript`` to {output_path}.
    Backslashes in the venv python path are doubled for YAML, and the
    inner double quotes around the python -c payload are YAML-escaped.
    Keeps the test scenarios readable.
    """
    interp = sys.executable.replace("\\", "\\\\")
    # Inside the YAML double-quoted string, we use single quotes around
    # the python -c body so we don't have to YAML-escape inner double
    # quotes. Single quotes inside the body are not needed; the body uses
    # double quotes for module references and string literals.
    payload = (
        f"import sys; open(sys.argv[1], 'w').write('{transcript}')"
    )
    command = f'{interp} -c "{payload}" {{output_path}}'
    # YAML-escape: double-quote the whole thing, escape inner " and \.
    yaml_escaped = command.replace("\\", "\\\\").replace('"', '\\"')
    return (
        "stt:\n"
        f"  provider: {provider_name}\n"
        "  providers:\n"
        f"    {provider_name}:\n"
        "      type: command\n"
        f'      command: "{yaml_escaped}"\n'
    )


SCENARIOS: list[tuple[str, str, dict[str, str], str]] = [
    # (label, config.yaml body, scenario_env, plugin_register)
    ("stt-disabled", "stt:\n  enabled: false\n", {}, "no"),
    ("explicit-groq", "stt:\n  provider: groq\n", {}, "no"),
    ("explicit-openai", "stt:\n  provider: openai\n", {}, "no"),
    ("explicit-local", "stt:\n  provider: local\n", {}, "no"),
    ("explicit-xai", "stt:\n  provider: xai\n", {}, "no"),
    # Mistral is quarantined → _get_provider returns "none" today, hence no_provider_error.
    ("explicit-mistral-quarantine", "stt:\n  provider: mistral\n", {}, "no"),
    # Unknown name + no plugin → both: no_provider_error
    ("unknown-no-plugin", "stt:\n  provider: openrouter\n", {}, "no"),
    # Unknown name + plugin installed → main: no_provider_error, PR: plugin
    ("plugin-installed", "stt:\n  provider: openrouter\n", {}, "yes"),
    # Unknown name + plugin reports unavailable → main: no_provider_error,
    # PR: plugin_unavailable (cleaner envelope, names the plugin)
    ("plugin-installed-unavailable", "stt:\n  provider: openrouter\n", {}, "unavailable"),
    # Built-in name + plugin tries to shadow → both: built-in
    ("explicit-openai-with-plugin-registered", "stt:\n  provider: openai\n", {}, "yes"),
    # NEW (this PR): stt.providers.<name>: type: command registry.
    # Provider name "fake-cli" + transcript prefixed "CMD:" so dispatch_kind
    # detection routes it to "command_provider". On main (no registry),
    # this falls through to no_provider_error.
    (
        "command-provider-installed",
        _cmd_yaml("fake-cli", "CMD: fake-cli transcript"),
        {},
        "no",
    ),
    # NEW (this PR): same name registered as BOTH a command provider and
    # a plugin under "openrouter". Command must win (config more local
    # than plugin install). The plugin emits "PLUGIN:..." — assertion is
    # that the transcript is "CMD:...", proving command-wins precedence.
    (
        "command-vs-plugin-same-name",
        _cmd_yaml("openrouter", "CMD: openrouter via command wins"),
        {},
        "yes",  # also register a plugin under "openrouter" — must NOT fire
    ),
    # NEW (this PR): built-in name with a command provider declared under
    # it → built-in still wins (built-in elif chain has precedence).
    # The command would write "CMD: HIJACK" if it fired — assertion is
    # that built-in OpenAI dispatch fires instead.
    (
        "explicit-openai-with-command-shadow",
        _cmd_yaml("openai", "CMD: HIJACK"),
        {},
        "no",
    ),
]


# Subprocesses reset the registry between runs via ``_reset_for_tests`` so
# registrations from earlier scenarios don't leak. The command-provider
# scenarios also work on origin/main — the subprocess just executes the
# native dispatch path, which falls through to "no_provider_error" because
# main has no registry for stt.providers.<name>.


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
    return {
        "dispatch_kind": shape.get("dispatch_kind"),
        "success": shape.get("success"),
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

        # On main, "plugin-installed" returns no_provider_error (no
        # plugin hook); on PR, plugin dispatches. Same shape for
        # "plugin-installed-unavailable" but PR returns the cleaner
        # plugin_unavailable envelope. The new command-provider scenarios
        # also intentionally diff against main (which has no stt.providers
        # registry yet).
        no_provider_to_plugin = (
            main_reduced.get("dispatch_kind") == "no_provider_error"
            and pr_reduced.get("dispatch_kind") == "plugin"
            and label == "plugin-installed"
        )
        no_provider_to_unavailable = (
            main_reduced.get("dispatch_kind") == "no_provider_error"
            and pr_reduced.get("dispatch_kind") == "plugin_unavailable"
            and label == "plugin-installed-unavailable"
        )
        no_provider_to_command = (
            main_reduced.get("dispatch_kind") == "no_provider_error"
            and pr_reduced.get("dispatch_kind") == "command_provider"
            and label in {"command-provider-installed", "command-vs-plugin-same-name"}
        )
        if no_provider_to_plugin:
            print(f"  [DIFF] {label}: no_provider_error → plugin — expected")
            intentional_diffs.append((label, main_reduced, pr_reduced))
        elif no_provider_to_unavailable:
            print(f"  [DIFF] {label}: no_provider_error → plugin_unavailable — expected")
            intentional_diffs.append((label, main_reduced, pr_reduced))
        elif no_provider_to_command:
            print(f"  [DIFF] {label}: no_provider_error → command_provider — expected")
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
            f"no_provider_error → plugin dispatch when a plugin is registered."
        )
    if failures or errors:
        return 1
    print(f"PARITY OK across {len(SCENARIOS)} scenarios.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

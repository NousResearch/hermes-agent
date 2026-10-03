"""
Loader for G0DM0D3 scripts. Handles the exec-scoping issues.

Usage in execute_code:
    exec(open(os.path.expanduser(
        os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "skills/security/godmode/scripts/load_godmode.py")
    )).read())
    
    # Now all functions are available:
    # - auto_jailbreak(), undo_jailbreak()
    # - race_models(), race_godmode_classic()
    # - generate_variants(), obfuscate_query(), detect_triggers()
    # - score_response(), is_refusal(), count_hedges()
    # - escalate_encoding()
"""

import os, sys
from pathlib import Path

_gm_scripts_dir = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "skills" / "security" / "godmode" / "scripts"

_gm_old_argv = sys.argv
sys.argv = ["_godmode_loader"]

# Where the scripts actually live: the documented install path first, then the directory this loader
# itself sits in (its category has moved before), then any skills/*/godmode/scripts beside it. A stale
# guess used to make this loader bind NOTHING and report nothing, so every caller silently got an empty
# namespace instead of an error.
_gm_candidates = [_gm_scripts_dir]
_gm_here = globals().get("__file__")
if _gm_here:
    _gm_candidates.append(Path(_gm_here).resolve().parent)
_gm_skills_root = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "skills"
_gm_candidates += [p for p in sorted(_gm_skills_root.glob("*/godmode/scripts")) if p not in _gm_candidates]

def _gm_load(path):
    ns = dict(globals())
    ns["__name__"] = "_godmode_module"
    ns["__file__"] = str(path)
    exec(compile(open(path).read(), str(path), 'exec'), ns)
    return ns

_gm_loaded = 0
for _gm_candidate in _gm_candidates:
    for _gm_script in ["parseltongue.py", "godmode_race.py", "auto_jailbreak.py"]:
        _gm_path = _gm_candidate / _gm_script
        if not _gm_path.exists():
            continue
        _gm_ns = _gm_load(_gm_path)
        _gm_loaded += 1
        for _gm_k, _gm_v in _gm_ns.items():
            if not _gm_k.startswith('_gm_') and (callable(_gm_v) or _gm_k.isupper()):
                globals()[_gm_k] = _gm_v
    if _gm_loaded:
        _gm_scripts_dir = _gm_candidate
        break

if not _gm_loaded:
    raise RuntimeError(
        "GODMODE loader: found no scripts to load in any of: "
        + ", ".join(str(c) for c in _gm_candidates)
        + ". The skill is installed somewhere this loader does not guess — reinstall it "
          "(`hermes skills install godmode`) or point HERMES_HOME at the right profile."
    )

sys.argv = _gm_old_argv

# Cleanup loader vars
for _gm_cleanup in ['_gm_scripts_dir', '_gm_old_argv', '_gm_load', '_gm_ns', '_gm_k',
                     '_gm_v', '_gm_script', '_gm_path', '_gm_cleanup', '_gm_candidate',
                     '_gm_candidates', '_gm_here', '_gm_skills_root', '_gm_loaded']:
    globals().pop(_gm_cleanup, None)

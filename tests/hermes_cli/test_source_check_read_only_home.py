"""The source check is a read-only probe: it must not seed SOUL.md or the home skeleton (#131026).

Importing ``hermes_cli.config`` runs ``_inject_profile_env_vars()`` at module import, which walks
provider discovery into ``load_config()`` and ``ensure_hermes_home()``. The desktop app runs the
source check on every launch, so a remote-only laptop kept getting a default SOUL.md and an empty
skills/ in a home whose identity deliberately lives elsewhere.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_initialize_home_inside_read_only_home_creates_nothing_but_the_home(tmp_path, monkeypatch):
    from hermes_cli.config_home import initialize_home, read_only_home

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    ensured: set = set()

    with read_only_home():
        initialize_home(home, ("skills", "memories"), ensured)
    assert home.is_dir()
    assert sorted(p.name for p in home.iterdir()) == []
    assert ensured == set(), "a read-only pass must not memoize the home as initialized"

    initialize_home(home, ("skills", "memories"), ensured)
    assert (home / "SOUL.md").is_file()
    assert (home / "skills").is_dir()
    assert ensured


def test_check_for_updates_does_not_seed_the_home(tmp_path):
    """Fresh interpreter: the import-time seeding only fires on first import of hermes_cli.config."""
    repo_root = Path(__file__).resolve().parents[2]
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("updates:\n  check: false\n", encoding="utf-8")
    code = (
        "import json, os; from pathlib import Path\n"
        "from hermes_cli.source_check import check_for_updates\n"
        "home = Path(os.environ['HERMES_HOME'])\n"
        "status = check_for_updates(install_root=Path(os.environ['REPO_ROOT']), home=home,\n"
        "                           cache_path=home / 'source-checks' / 'probe.json', passive=True)\n"
        "print(json.dumps({'status': status, 'entries': sorted(p.name for p in home.iterdir())}))\n"
    )
    env = {**os.environ, "HERMES_HOME": str(home), "REPO_ROOT": str(repo_root),
           "PYTHONPATH": str(repo_root)}
    env.pop("HERMES_MANAGED", None)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         timeout=120, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr[-2000:]
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    assert payload["status"].get("reason") == "disabled"
    skeleton = {"SOUL.md", "skills", "memories", "cron", "sessions", "logs", "hooks", "pairing",
                "image_cache", "audio_cache"}
    assert not (skeleton & set(payload["entries"])), payload["entries"]
    # The config loader's known-good snapshot (backups/config/config.yaml.good.*) is its own
    # behavior and may appear; it is not home initialization.
    assert set(payload["entries"]) <= {"config.yaml", "backups"}, payload["entries"]

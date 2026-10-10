"""Under a named profile the updater (root store) and pm (profile store) still see each other's receipts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]

_PROBE = textwrap.dedent("""
    import json, sys, time
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    from hermes_cli import update_receipt as receipts
    from pm import receipt as pm_receipt

    old = Path(sys.argv[2]) / 'logs' / 'update_receipts'
    old.mkdir(parents=True)
    (old / 'update_20260101_000000_1_old.json').write_text(
        json.dumps({'update_id': 'old', 'action_id': 'a' * 32, 'outcome': 'success'}), encoding='utf-8')
    receipts.begin_update_receipt()
    receipts.finalize_update_receipt('success')
    update_id = receipts.read_latest_receipt()['update_id']
    pm_sees_update = (pm_receipt.latest() or {}).get('update_id')
    time.sleep(0.05)
    pm_receipt.begin('sync')
    pm_receipt.finalize('ok')
    Path('result.json').write_text(json.dumps({
        'update_id': update_id, 'pm_sees_update': pm_sees_update,
        'updater_sees_pm': (receipts.read_latest_receipt() or {}).get('kind'),
        'old_archive': (receipts.read_receipt_for_action('a' * 32) or {}).get('update_id'),
        'root_archives': [p.name for p in receipts._receipt_dir().glob('update_*.json')],
    }))
""")


def test_named_profile_sees_update_and_pm_receipts_in_both_directions(tmp_path):
    home = tmp_path / "account"
    profile = home / ".hermes" / "profiles" / "foo"
    profile.mkdir(parents=True)
    script = tmp_path / "probe.py"
    script.write_text(_PROBE, encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("HERMES_", "PYTEST_"))}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(profile),
               HERMES_RUNTIME_DIR=str(tmp_path / "tools"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(ROOT), *filter(None, sys.path)]))
    done = subprocess.run([sys.executable, str(script), str(ROOT), str(profile)], env=env, cwd=tmp_path,
                          text=True, capture_output=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    # The updater still writes the root store the Desktop and hand-off scripts read.
    assert any(result["update_id"] in name for name in result["root_archives"])
    assert result["pm_sees_update"] == result["update_id"]  # `hermes -p foo pm status`
    assert result["updater_sees_pm"] == "sync"  # read_latest_receipt() consumers
    assert result["old_archive"] == "old"  # receipts written to the profile store before the move

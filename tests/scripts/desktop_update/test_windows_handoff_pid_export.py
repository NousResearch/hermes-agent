"""A Windows Desktop hand-off's update adopts the hand-off's own claim after launch preparation.

``hermes update`` adopts an existing claim only when a live partner is its own pid, the
``HERMES_UPDATE_HANDOFF_PID`` value, or an ancestor (``update_lock.UpdateLock._is_partner``).
``windows.ps1`` hands marker line 1 to a custodian before any update work starts; the custodian
is a sibling of the update, never an ancestor. The update child is named on the delegate line,
but when its tree's dependencies are stale (a hand merge, an interrupted pull) or a tail is owed,
``hermes_bootstrap`` -> ``venv_sync.prepare_launch`` takes and releases a pre-flight claim in that
very process first, and releasing as the delegate drops the delegate line while the owner lives
(A7 rule 5). The relaunched ``hermes update`` then saw only the custodian and refused its own
claim with exit 2 ("Another Hermes update is already running (... process <custodian>)").
``posix.sh`` exports the custodian since 013e94b38a (#134268); windows.ps1 must too.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import sysconfig

import pytest

from tests.installation_launcher_fixture import publish_fixture_launcher
from tests.scripts.desktop_update.windows_handoff_support import _custodian, _finish, _script

pytestmark = pytest.mark.platforms("windows")

REPOSITORY = Path(__file__).resolve().parents[3]

# Generation 1 is the process windows.ps1 starts and names as the delegate. It does what
# prepare_launch does under a live hand-off claim -- take and release a pre-flight claim -- with
# this repository's real UpdateLock, then relaunches generation 2 the way hermes_bootstrap does on
# Windows (subprocess.call), which claims the update as `hermes update` would.
GENERATIONS = """
import json, os, subprocess, sys
from pathlib import Path


def _real_update_lock():
    # The stand-in checkout shadows the repository's packages: forget them, use the real ones.
    fake = str(Path(__file__).resolve().parents[1])
    for name, module in list(sys.modules.items()):
        if (getattr(module, '__file__', None) or '').startswith(fake):
            del sys.modules[name]
    sys.path[:0] = [os.environ['GEN_REPO'], os.environ['GEN_SITE']]
    from hermes_cli import update_lock
    return update_lock


PROBE = (
    "import json, os, sys; from pathlib import Path\\n"
    "sys.path[:0] = [os.environ['GEN_REPO'], os.environ['GEN_SITE']]\\n"
    "from hermes_cli.update_lock import UpdateLock\\n"
    "lock = UpdateLock()\\n"
    "ok = lock.acquire()\\n"
    "Path(os.environ['GEN_OUT']).write_text(json.dumps({'adopted': ok, "
    "'holder': lock.holder.pid if lock.holder else None, "
    "'told': os.environ.get('HERMES_UPDATE_HANDOFF_PID')}), encoding='utf-8')\\n"
    "ok and lock.release()\\n"
    "sys.exit(0 if ok else 2)\\n")


def main():
    if '--version' in sys.argv:
        print('Install directory: ' + str(Path(__file__).resolve().parents[1])); return 0
    if '--help' in sys.argv:
        print('--keep-stash'); return 0
    if sys.argv[1:2] == ['update']:
        update_lock = _real_update_lock()
        preflight = update_lock.UpdateLock(install_root=Path(__file__).resolve().parents[1], checkout_first=False)
        if not preflight.acquire():
            return 2
        preflight.release()
        return subprocess.call([sys.executable, '-I', '-c', PROBE])
    return 0


if __name__ == '__main__':
    sys.exit(main())
"""


def test_update_relaunched_after_launch_preparation_adopts_the_handoffs_claim(tmp_path):
    install = tmp_path / "checkout"
    # A real interpreter, as PM's store python is: a venv python.exe is a redirector that would
    # put a launcher process, not generation 1, on the delegate line.
    publish_fixture_launcher(install, GENERATIONS, interpreter=Path(sys._base_executable))
    home = tmp_path / "home"
    home.mkdir()
    out = tmp_path / "gen2.json"

    script = _script(home, install=install, GEN_OUT=str(out), GEN_REPO=str(REPOSITORY),
                     GEN_SITE=sysconfig.get_paths()["purelib"])
    code, output = _finish(script)

    custodian = _custodian(home, script.pid)
    assert custodian, f"the hand-off never named a custodian: {output}"
    verdict = json.loads(out.read_text(encoding="utf-8-sig"))
    assert verdict["adopted"], (f"the relaunched update refused its own hand-off's claim "
                                f"(holder {verdict['holder']}, custodian {custodian})")
    assert verdict["told"] == custodian, "the update must be told the marker's owner, not the hand-off script"
    assert code == 0, output

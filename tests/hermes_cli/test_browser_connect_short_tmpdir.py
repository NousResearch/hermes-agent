"""The Chrome that Connect Chrome launches gets a temp root short enough for its singleton socket.

Hermes points ``TMPDIR`` at ``<HERMES_HOME>/cache/scratch`` for itself and its children
(``hermes_constants.export_scratch_tmp_env``, run at boot). Chrome binds its singleton socket at
``$TMPDIR/com.google.Chrome.XXXXXX/SingletonSocket``, and an AF_UNIX path holds at most 107 bytes,
so under a home longer than about 52 characters the Chrome that ``launch_chrome_debug`` spawned died
at start ("Socket path too long").
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

from hermes_cli.browser_connect import launch_chrome_debug
from hermes_constants import SCRATCH_DIR_MARKER_ENV, SCRATCH_TMP_ENV_VARS, apply_scratch_tmp_env

_SUN_PATH_BYTES = 107
# Google Chrome binds com.google.Chrome.XXXXXX/SingletonSocket; Chromium's name is the longer one.
_CHROME_SOCKET = os.path.join("org.chromium.Chromium.XXXXXX", "SingletonSocket")


def test_chrome_gets_a_temp_root_its_singleton_socket_fits_under_a_long_home(tmp_path, monkeypatch):
    home = tmp_path / ("long-hermes-home-" * 4)  # with the scratch dir, well past 107 bytes
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in (*SCRATCH_TMP_ENV_VARS, SCRATCH_DIR_MARKER_ENV):
        monkeypatch.delenv(key, raising=False)
    boot_env = {"HERMES_HOME": str(home)}
    assert apply_scratch_tmp_env(boot_env)  # what the boot hook exports for this home
    for key, value in boot_env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(tempfile, "tempdir", None)
    scratch = os.environ["TMPDIR"]
    assert len(os.fsencode(os.path.join(scratch, _CHROME_SOCKET))) > _SUN_PATH_BYTES

    spawned = []

    class _Chrome:
        pid, returncode = 4321, 1

        def poll(self):
            return 1

    def popen(argv, **kwargs):
        spawned.append(kwargs)
        return _Chrome()

    with patch("hermes_cli.browser_connect.get_chrome_debug_candidates", return_value=["/usr/bin/google-chrome"]), \
         patch("hermes_cli.browser_connect.discover_local_cdp_url", return_value=None), \
         patch("subprocess.Popen", side_effect=popen):
        launch_chrome_debug(9333, "Linux")

    assert len(spawned) == 1
    chrome_env = spawned[0].get("env") or os.environ  # no env: Chrome inherits Hermes' own
    chrome_tmp = chrome_env.get("TMPDIR") or "/tmp"  # no-tmp: ok — Chrome's own default root
    assert len(os.fsencode(os.path.join(chrome_tmp, _CHROME_SOCKET))) <= _SUN_PATH_BYTES
    # Only Chrome moves: Hermes and its other children keep the scratch dir.
    assert os.environ["TMPDIR"] == scratch
    assert chrome_env.get("HERMES_HOME") == str(home)

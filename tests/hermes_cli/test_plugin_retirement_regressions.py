"""Plugin retirement preserves nonblocking probes."""

import threading
from pathlib import Path

from hermes_cli import plugins
from hermes_cli.plugins_lifecycle import reserve_plugin_manager_for_home


def test_nowait_probes_do_not_wait_for_profile_teardown():
    home = Path(plugins.get_hermes_home())
    plugins.get_plugin_manager()
    reserved = threading.Event()
    release = threading.Event()

    def teardown():
        with reserve_plugin_manager_for_home(home):
            reserved.set()
            release.wait(5)

    worker = threading.Thread(target=teardown)
    worker.start()
    try:
        assert reserved.wait(5)
        done = threading.Event()
        results = []

        def probe():
            results.extend((plugins.hook_unused_nowait('on_session_end'),
                            plugins.plugin_command_absent_nowait('missing')))
            done.set()

        probe_thread = threading.Thread(target=probe)
        probe_thread.start()
        try:
            assert done.wait(2)
            assert results == [False, False]
        finally:
            release.set()
            probe_thread.join(5)
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()

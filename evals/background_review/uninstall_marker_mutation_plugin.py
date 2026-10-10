"""pytest plugin: mutate ``_hermes_path_markers`` to prove the scoping test is not vacuous.

Card ``t_0cf0aa7e``. The third test in
``tests/hermes_cli/test_uninstall_windows_registry_isolation.py`` claims the deleter is
prefix-scoped to Hermes-owned entries. A green run proves nothing unless the test fails when
that property is broken, so this plugin installs two plausible "simplifications" of
``_hermes_path_markers`` and the same test file runs against them:

  M1  the markers lose the sub-suffix anchor (``[<root>]`` instead of ``[<root>\\sub, ...]``)
      -- the sibling ``<parent>\\hermes-setup\\bin`` lookalike starts disappearing;
  M2  the markers stop being derived from the argument (a fixed, unrelated root)
      -- nothing is removed at all, so the Hermes-owned entries survive.

Expected: exactly one test fails (``test_strip_is_prefix_scoped_to_hermes_owned_entries``);
the two guard tests stay green, because the guard is what they cover.

    MUTATION=M1 PYTEST_PLUGINS=uninstall_marker_mutation_plugin \\
        PYTHONPATH=evals/background_review:. python -m pytest \\
        tests/hermes_cli/test_uninstall_windows_registry_isolation.py

Pass it through ``PYTEST_PLUGINS``, not ``-p``: ``hermes_cli.main._apply_profile_override``
parses ``sys.argv`` and reads a bare ``-p <name>`` as the *profile* flag, which aborts the run
with ``Profile '<name>' does not exist`` before a single test starts.
"""
from __future__ import annotations

import os
from pathlib import Path

MUT = os.environ.get("MUTATION", "M1")


def pytest_configure(config):
    import hermes_cli.uninstall as uninstall

    if MUT == "M1":
        uninstall._hermes_path_markers = lambda hermes_home, *, include_managed_bin=False: [
            str(hermes_home).rstrip("\\/")
        ]
    else:
        uninstall._hermes_path_markers = (
            lambda hermes_home, *, include_managed_bin=False: [
                str(Path("C:/unrelated/hermes")) + "\\" + sub
                for sub in (("hermes-agent", "git", "node", "venv")
                            + (("bin",) if include_managed_bin else ()))
            ]
        )
    print(f"\n[mutate] _hermes_path_markers replaced (MUTATION={MUT})")

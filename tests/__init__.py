"""Test package.

Arm the runner-independent write guard for runs that bypass pytest:
``conftest.py`` sandboxes ``HERMES_HOME`` and trips on real-home I/O, but only
pytest loads a conftest — ``python -m unittest ...`` and direct file runs never
do, and those are the runs that overwrote the live config on 2026-10-10.

Under pytest the conftest guard owns this, including its ``allow_real_home_io``
opt-out, so we stay out of the way there.
"""

import sys as _sys

if "pytest" not in _sys.modules:
    try:
        from tests.safe_home import arm as _arm
    except ImportError:  # direct-file run: the repo root is not on sys.path
        pass
    else:
        _arm()

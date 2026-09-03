"""The gateway systemd unit must raise the file-descriptor ceiling.

Regression test for a real incident: the kenbot gateway wedged at 994 of 1024
open fds (475 state.db + 455 state.db-wal + 34 plugin db handles) and stayed
wedged ~52 minutes. Tools returned "[Errno 24] Too many open files" and the
cron loader could not read its own jobs.json.

SQLite read connections are pooled and recede when idle, so the fd count is a
high-water mark that tracks concurrent work rather than a monotonic leak. The
default 1024 soft limit is simply too low a ceiling for that high-water mark
under sustained load, while systemd permits up to 1048576.

Both unit templates (system-scope and user-scope) must carry the raised limit;
the incident happened on the user-scope unit.
"""
import getpass
import re

import pytest

from hermes_cli.gateway import generate_systemd_unit


def _service_section(unit: str) -> str:
    m = re.search(r"^\[Service\]\s*$(.*?)(?=^\[|\Z)", unit, re.M | re.S)
    assert m, "unit has no [Service] section"
    return m.group(1)


@pytest.mark.parametrize("system", [False, True])
def test_gateway_unit_raises_nofile_limit(system):
    unit = generate_systemd_unit(system=system, run_as_user=getpass.getuser())
    service = _service_section(unit)

    m = re.search(r"^LimitNOFILE=(\d+)\s*$", service, re.M)
    assert m, (
        f"{'system' if system else 'user'}-scope unit does not set LimitNOFILE; "
        "it would inherit the 1024 soft limit and re-wedge under sustained load"
    )

    limit = int(m.group(1))
    assert limit >= 65536, (
        f"LimitNOFILE={limit} is too low: the observed high-water mark reached "
        "994 fds against a 1024 ceiling, so the new ceiling needs real headroom"
    )
    assert limit <= 1048576, (
        f"LimitNOFILE={limit} exceeds the systemd maximum (1048576)"
    )


@pytest.mark.parametrize("system", [False, True])
def test_nofile_limit_declared_once(system):
    """A duplicated directive means the last one silently wins."""
    unit = generate_systemd_unit(system=system, run_as_user=getpass.getuser())
    assert unit.count("LimitNOFILE=") == 1

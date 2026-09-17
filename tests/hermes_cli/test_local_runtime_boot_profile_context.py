"""Profile-scope regression for the on-demand local-runtime boot thread."""

import threading

import hermes_cli.config as config_module
import hermes_cli.local_runtime.bootstrap as bootstrap
import hermes_cli.local_runtime.endpoint as endpoint
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


def test_on_demand_boot_loads_config_in_routed_profile(tmp_path, monkeypatch):
    """The boot worker must load config under the profile that requested it."""
    launch_home = tmp_path / "launch"
    served_home = tmp_path / "served"
    launch_home.mkdir()
    served_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    observed = {}
    done = threading.Event()

    def fake_load_config():
        return {"observed_home": str(get_hermes_home())}

    def fake_ensure_local_runtime(config):
        observed["home"] = config["observed_home"]
        done.set()

    monkeypatch.setattr(config_module, "load_config", fake_load_config)
    monkeypatch.setattr(bootstrap, "ensure_local_runtime", fake_ensure_local_runtime)

    token = set_hermes_home_override(served_home)
    try:
        endpoint._kick_managed_boot(None)
        assert done.wait(2), "on-demand boot worker did not run"
    finally:
        reset_hermes_home_override(token)

    assert observed["home"] == str(served_home)

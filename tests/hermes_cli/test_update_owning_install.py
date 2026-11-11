"""``hermes update`` from a PM environment's workspace goes to the checkout that owns it (#122627)."""
import sys

from hermes_cli.update_owning_install import owning_install_root
from pm.environments import install_key


def test_pm_generation_venv_is_owned_by_the_checkout_its_install_key_names(tmp_path, monkeypatch):
    checkout = tmp_path / "hermes-agent"
    (checkout / "hermes_cli").mkdir(parents=True)
    (checkout / "hermes_cli" / "main.py").write_text("", encoding="utf-8")
    state = tmp_path / "installs" / install_key(checkout)
    generation = state / "environments" / "0123abcd"
    (generation / "venv").mkdir(parents=True)
    workspace = generation / "workspace"
    (workspace / "hermes_cli").mkdir(parents=True)
    (workspace / "hermes_cli" / "main.py").write_text("", encoding="utf-8")
    (state / "inputs").mkdir()
    record = state / "inputs" / ".project-root"
    record.write_text(str(checkout), encoding="utf-8")
    monkeypatch.setattr(sys, "prefix", str(generation / "venv"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "python"))
    monkeypatch.delenv("PYTHONPATH", raising=False)

    assert owning_install_root(workspace) == checkout.resolve()

    record.write_text(str(tmp_path / "elsewhere"), encoding="utf-8")  # not the path the key hashes
    assert owning_install_root(workspace) is None

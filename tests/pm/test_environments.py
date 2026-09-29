import json
from pathlib import Path


def test_recorded_venv_accepts_case_variant_of_install_root(tmp_path, monkeypatch):
    import pm.environments as environments

    generations = tmp_path / "install" / "environments"
    recorded = generations / "generation" / "venv"
    recorded.mkdir(parents=True)
    (recorded / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")

    facts = tmp_path / "facts.json"
    facts.write_text(
        json.dumps({"packages": {"venv": {"environment": str(recorded)}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(environments, "runtime_facts_path", lambda project_root: facts)
    monkeypatch.setattr(environments, "install_state_dir", lambda project_root: tmp_path / "install")

    real_relative_check = Path.is_relative_to
    monkeypatch.setattr(
        Path,
        "is_relative_to",
        lambda self, other: False if self == recorded else real_relative_check(self, other),
    )

    assert environments._recorded_venv(tmp_path / "project") == recorded
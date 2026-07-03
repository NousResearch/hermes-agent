import stat
from pathlib import Path

from orchard.config import Settings
from orchard.models import Employee
from orchard.provisioner import deprovision, scaffold_home


def _settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.paths.root = tmp_path / "data"
    s.paths.runtime = tmp_path / "run"
    s.paths.registry_db = tmp_path / "data" / "orchard.db"
    s.llm.base_url = "http://llm.internal:8000/v1"
    s.llm.provider = "deepseek"
    s.llm.model = "deepseek-v4-flash"
    return s


def test_scaffold_creates_locked_home(tmp_path: Path):
    s = _settings(tmp_path)
    emp = Employee("alice", "Alice", "mm-1", 0.0)
    home = scaffold_home(s, emp)

    assert (home / "config.yaml").exists()
    assert (home / ".env").exists()
    for sub in ("skills", "sessions", "workspace", "home"):
        assert (home / sub).is_dir()

    # home dir is 0700
    assert stat.S_IMODE((home).stat().st_mode) == 0o700
    # secret is 0600
    assert stat.S_IMODE((home / ".env").stat().st_mode) == 0o600

    # config points at the internal endpoint + disables thinking
    cfg = (home / "config.yaml").read_text()
    assert "deepseek:deepseek-v4-flash" in cfg
    assert "reasoning_effort: none" in cfg
    env = (home / ".env").read_text()
    assert "DEEPSEEK_BASE_URL=http://llm.internal:8000/v1" in env


def test_scaffold_is_idempotent(tmp_path: Path):
    s = _settings(tmp_path)
    emp = Employee("bob", "Bob", "mm-2", 0.0)
    scaffold_home(s, emp)
    scaffold_home(s, emp)  # must not raise
    assert deprovision(s, "bob") is True
    assert deprovision(s, "bob") is False


def test_tenant_homes_are_separate(tmp_path: Path):
    s = _settings(tmp_path)
    a = scaffold_home(s, Employee("alice", "A", "mm-1", 0.0))
    b = scaffold_home(s, Employee("bob", "B", "mm-2", 0.0))
    assert a != b
    assert a.parent == b.parent  # both under employees/, distinct dirs

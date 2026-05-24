from pathlib import Path
import tomllib


def test_lark_oapi_does_not_bypass_exclude_newer_policy():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    uv_options = pyproject.get("tool", {}).get("uv", {})
    package_overrides = uv_options.get("exclude-newer-package", {})
    lock_text = Path("uv.lock").read_text(encoding="utf-8")

    assert "lark-oapi" not in package_overrides
    assert "[options.exclude-newer-package]" not in lock_text

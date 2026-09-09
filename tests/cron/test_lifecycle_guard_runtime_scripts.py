"""Runtime source text must not become a recursive shell dependency graph."""

import pytest

from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script as guard


@pytest.mark.parametrize("shebang", ["#!/usr/bin/env bun", "#!/usr/bin/env -S node", "#!/usr/bin/python3"])
def test_runtime_entry_does_not_execute_source_references(tmp_path, shebang):
    dependency = tmp_path / "dependency"
    dependency.write_text("hermes gateway restart\n")
    entry = tmp_path / "entry.ts"
    entry.write_text(f"{shebang}\n// documentation example\n{dependency}\n")
    link = tmp_path / "cli"
    link.symlink_to(entry)
    assert not guard(f"{link} --help", cwd=str(tmp_path))
    # An explicit shell ignores the shebang: retain recursive protection.
    assert guard(f"bash {link} --help", cwd=str(tmp_path))
    assert guard(f"source {link}", cwd=str(tmp_path))
    assert guard(f"{link} --help; hermes gateway restart", cwd=str(tmp_path))
    entry.write_text(f"{shebang}\nhermes gateway restart\n")
    assert guard(f"{link} --help", cwd=str(tmp_path))


def test_runtime_entry_remains_bounded(tmp_path, monkeypatch):
    import cron.lifecycle_guard as module

    entry = tmp_path / "cli"
    entry.write_text("#!/usr/bin/env bun\n" + "x" * 2048)
    monkeypatch.setattr(module, "_MAX_LIFECYCLE_SCAN_BYTES", 1024)
    assert guard(str(entry), cwd=str(tmp_path))

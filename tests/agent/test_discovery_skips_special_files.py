"""A FIFO or device link named SKILL.md / plugin.yaml must not hang discovery (Copilot CLI 1.0.111 parity)."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")

_PROBE = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, sys.argv[1])
    from agent.prompt_builder import build_skills_system_prompt
    from agent.skill_commands import scan_skill_commands
    from hermes_cli.plugins_discovery import scan_directory
    from pathlib import Path
    prompt = build_skills_system_prompt()
    commands = scan_skill_commands()
    names = sorted(m.name for m in scan_directory(Path(sys.argv[2]), "user"))
    print("fine skill" in prompt, "/ok" in commands, names)
    """
)


@pytest.mark.parametrize("kind", ["fifo", "dev_zero"])
def test_special_file_manifests_are_skipped_and_siblings_still_load(tmp_path, kind):
    home = tmp_path / "home"
    for d in ("skills/bad", "skills/ok", "plugins/bad", "plugins/good"):
        (home / d).mkdir(parents=True)
    (home / "skills/ok/SKILL.md").write_text("---\nname: ok\ndescription: fine skill\n---\nbody\n", encoding="utf-8")
    (home / "plugins/good/plugin.yaml").write_text("name: good\n", encoding="utf-8")
    (home / "plugins/good/__init__.py").write_text("def register(ctx): pass\n", encoding="utf-8")
    (home / "plugins/bad/__init__.py").write_text("def register(ctx): pass\n", encoding="utf-8")
    for bad in (home / "skills/bad/SKILL.md", home / "plugins/bad/plugin.yaml"):
        if kind == "fifo":
            os.mkfifo(bad)
        else:
            bad.symlink_to("/dev/zero")
    env = {**os.environ, "HERMES_HOME": str(home)}
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(REPO), str(home / "plugins")],
        env=env, capture_output=True, text=True, encoding="utf-8", timeout=60, stdin=subprocess.DEVNULL, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "True True ['good']"

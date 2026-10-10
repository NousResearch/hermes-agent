"""The repository shebang checker scans tracked scripts and embedded payloads."""
from pathlib import Path
import subprocess
import sys


CHECKER = Path(__file__).resolve().parents[2] / "scripts" / "check_bash_shebangs.py"


def test_checker_reports_and_clears_fixed_bash_paths(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    fixed = "#!" + "/bin/bash"
    files = {
        "activate": fixed + "\necho ready\n",
        "example.md": "```bash\n" + fixed + "\n```\n",
        "generator.py": 'script = "' + fixed + '\\necho ready\\n"\n',
        "other.sh": "#!/bin/sh\n",
        "portable.sh": "#!/usr/bin/env bash\n",
        "usr.sh": "#! /usr/bin/" + "bash\n",
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "--", *files], cwd=tmp_path, check=True)
    (tmp_path / "untracked.sh").write_text(fixed, encoding="utf-8")
    command = [sys.executable, str(CHECKER), "--root", str(tmp_path)]
    red = subprocess.run(command, capture_output=True, text=True, check=False)
    assert red.returncode == 1
    assert red.stdout.splitlines() == [
        "activate:1: use #!/usr/bin/env bash",
        "example.md:2: use #!/usr/bin/env bash",
        "generator.py:1: use #!/usr/bin/env bash",
        "usr.sh:1: use #!/usr/bin/env bash",
        "Bash shebang check: 4 violation(s)",
    ]
    for name, text in files.items():
        (tmp_path / name).write_text(
            text.replace(fixed, "#!/usr/bin/env bash").replace("#! /usr/bin/" + "bash", "#!/usr/bin/env bash"),
            encoding="utf-8",
        )
    green = subprocess.run(command, capture_output=True, text=True, check=False)
    assert green.returncode == 0, green.stdout + green.stderr
    assert green.stdout.strip() == "Bash shebang check: 0 violation(s)"

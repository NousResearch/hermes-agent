import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _write_sourceable_install_script(tmp_path: Path) -> Path:
    script = INSTALL_SH.read_text(encoding="utf-8")
    script = re.sub(r"\nmain\s*$", "\n", script)
    script_path = tmp_path / "install-under-test.sh"
    script_path.write_text(script, encoding="utf-8")
    return script_path


def test_prompt_yes_no_consumes_entire_line(tmp_path: Path):
    script_path = _write_sourceable_install_script(tmp_path)
    input_path = tmp_path / "prompt-input.txt"
    input_path.write_text("no\nNEXT\n", encoding="utf-8")

    command = f"""
source "{script_path}"
exec 3< "{input_path}"
if prompt_yes_no "Import from OpenClaw during setup?" true <&3; then
  echo "ANSWER:YES"
else
  echo "ANSWER:NO"
fi
IFS= read -r leftover <&3 || true
echo "LEFTOVER:$leftover"
"""
    result = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "ANSWER:NO" in result.stdout
    assert "LEFTOVER:NEXT" in result.stdout

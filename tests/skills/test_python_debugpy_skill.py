"""Security guidance contract for the bundled python-debugpy skill."""

from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "skills/software-development/python-debugpy/SKILL.md"
GENERATED_DOCS = tuple((REPO / "website").glob("**/software-development-python-debugpy.md"))
UNSAFE_DOWNGRADE_COMMANDS = (
    re.compile(r"(?:sudo\s+)?sysctl[^\n]*ptrace_scope\s*=\s*0", re.IGNORECASE),
    re.compile(r"echo\s+0[^\n]*ptrace_scope", re.IGNORECASE),
    re.compile(r"(?:>|tee\s+)[^\n]*/proc/sys/kernel/yama/ptrace_scope", re.IGNORECASE),
)


def test_debugpy_guidance_never_publishes_a_ptrace_scope_zero_command():
    documents = (SKILL, *GENERATED_DOCS)
    assert len(documents) >= 3

    for document in documents:
        content = document.read_text(encoding="utf-8")
        for pattern in UNSAFE_DOWNGRADE_COMMANDS:
            assert not pattern.search(content), (
                f"{document.relative_to(REPO)} publishes an unsafe ptrace downgrade command"
            )

"""Executable acceptance test generator & red-first verifier (05-final-plan.md §4 step 4, §7).

Compiles SpecDocument stories and roles into executable pytest suites under
``.samagent/acceptance/``:
- ``test_acceptance.py``: story-by-story acceptance checks against the app's ASGI/WSGI or HTTP surface
- ``test_security_probes.py``: role-matrix authz, IDOR, SQLi, and secret-exposure probes (H6)
And verifies that the generated tests are RED (fail for the right reason) before workers run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

from samagent.spec.models import SpecDocument


@dataclass
class RedFirstCheckResult:
    is_red_for_right_reason: bool
    exit_code: int
    failed_count: int
    syntax_error: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_acceptance_suite(project_dir: Path, spec: SpecDocument) -> List[Path]:
    """Generate executable acceptance and security probe test files in .samagent/acceptance/."""
    acc_dir = Path(project_dir) / ".samagent" / "acceptance"
    acc_dir.mkdir(parents=True, exist_ok=True)

    has_roles = any(r.lower() not in ("visitor", "anonymous", "public", "guest") for r in spec.roles)

    story_assertions: List[str] = []
    for s in spec.stories:
        fn_name = f"test_story_{s.id.lower()}_{s.as_role.lower()}"
        story_assertions.append(
            f'''def {fn_name}(app_client):
    """Story {s.id} (as {s.as_role}): {s.can}
    Acceptance: {s.accept}
    """
    resp = app_client.request_story({s.id!r}, method={s.method!r}, route={s.route!r}, role={s.as_role!r})
    assert resp["implemented"] is True, f"Story {s.id} endpoint {s.method} {s.route} not implemented yet: {{resp}}"
    assert resp["passed"] is True, f"Story {s.id} acceptance failed: {{resp}}"
'''
        )

    acceptance_code = f'''"""Auto-generated executable acceptance suite for: {spec.goal}
Generated from .samagent/spec.yaml. Do NOT weaken or delete these tests.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _AppHarness:
    """Loads the target application (app/main.py) and executes story probes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.app_module = self._load_app()

    def _load_app(self):
        main_py = self.root / "app" / "main.py"
        if not main_py.exists():
            return None
        spec = importlib.util.spec_from_file_location("target_app_main", main_py)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def request_story(self, story_id: str, *, method: str, route: str, role: str) -> dict:
        if self.app_module is None or not hasattr(self.app_module, "handle_request"):
            return {{"implemented": False, "passed": False, "reason": "app/main.py handle_request missing"}}
        return self.app_module.handle_request(story_id=story_id, method=method, route=route, role=role)


@pytest.fixture()
def app_client():
    return _AppHarness(PROJECT_ROOT)


''' + "\n\n".join(story_assertions)

    security_code = f'''"""Auto-generated L3 Security & Role-Matrix Probes for: {spec.goal}
Tests OWASP Top-10 vibe-coding failure modes (missing auth, IDOR, SQLi, leaked secrets, invalid input).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HAS_ROLES = {has_roles!r}

_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9]{{20,}}|AIza[0-9A-Za-z-_]{{35}}|-----BEGIN (?:RSA |EC )?PRIVATE KEY-----)"
)
_RAW_SQL_FSTRING_RE = re.compile(
    r"""(?:execute|cursor\\.execute)\\s*\\(\\s*f['"](?:SELECT|INSERT|UPDATE|DELETE)\\b""",
    re.IGNORECASE,
)


def _load_app():
    main_py = PROJECT_ROOT / "app" / "main.py"
    if not main_py.exists():
        return None
    spec = importlib.util.spec_from_file_location("target_app_sec", main_py)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_hardcoded_secrets_in_source():
    """L3.1: No hardcoded cloud API keys or private keys in repository files."""
    app_dir = PROJECT_ROOT / "app"
    assert app_dir.exists(), "app/ directory must exist"
    for p in app_dir.rglob("*"):
        if p.is_file() and p.suffix in (".py", ".js", ".ts", ".html", ".json", ".env"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            assert not _SECRET_RE.search(text), f"Hardcoded secret pattern found in {{p}}"


def test_no_raw_fstring_sql_injection():
    """L3.2: Queries must use parameterized placeholders, never f-string SQL."""
    app_dir = PROJECT_ROOT / "app"
    assert app_dir.exists(), "app/ directory must exist"
    for p in app_dir.rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        assert not _RAW_SQL_FSTRING_RE.search(text), f"Raw f-string SQL injection sink found in {{p}}"


def test_role_matrix_and_idor_isolation():
    """L3.3: Anonymous callers cannot mutate protected resources; Member B cannot read Member A's private record."""
    mod = _load_app()
    assert mod is not None and hasattr(mod, "run_security_probes"), "app/main.py must expose run_security_probes()"
    report = mod.run_security_probes(has_roles=HAS_ROLES)
    assert report.get("unauth_rejected") is True, f"Anonymous access to protected route was not rejected: {{report}}"
    assert report.get("idor_blocked") is True, f"IDOR probe failed (Member B accessed Member A record): {{report}}"
    assert report.get("input_validated") is True, f"Invalid/malformed payload was not rejected: {{report}}"
'''

    acc_file = acc_dir / "test_acceptance.py"
    sec_file = acc_dir / "test_security_probes.py"
    acc_file.write_text(acceptance_code, encoding="utf-8")
    sec_file.write_text(security_code, encoding="utf-8")
    return [acc_file, sec_file]


def verify_red_first(project_dir: Path) -> RedFirstCheckResult:
    """Run the generated acceptance tests before implementation and verify they fail cleanly (red-first)."""
    acc_dir = Path(project_dir) / ".samagent" / "acceptance"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(acc_dir), "-q", "--tb=short"],
        cwd=str(project_dir),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    syntax_err = "SyntaxError" in out or "IndentationError" in out
    failed_count = 0
    for token in out.split():
        if token.isdigit():
            pass
    import re

    m = re.search(r"(\d+)\s+failed", out)
    if m:
        failed_count = int(m.group(1))
    is_red = proc.returncode != 0 and failed_count > 0 and not syntax_err
    return RedFirstCheckResult(
        is_red_for_right_reason=is_red,
        exit_code=proc.returncode,
        failed_count=failed_count,
        syntax_error=syntax_err,
        summary=out.strip().splitlines()[-1] if out.strip() else "no output",
    )

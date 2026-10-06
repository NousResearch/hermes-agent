"""The dashboard's production typecheck must not compile Vitest suites.

Regression for #134168: ``web/tsconfig.app.json`` included every ``*.test.ts(x)``
under ``src`` (and the bundled ``../apps/shared/src``), so a test-only type error
failed ``scripts/build/web.mjs``'s ``tsc -b``, ``_resolve_dashboard_web_dist``
exited 1, and the LaunchAgent crash-looped with nothing listening on port 9119.
"""

import json
import re
from pathlib import Path

_WEB_TSCONFIG = Path(__file__).resolve().parents[2] / "web" / "tsconfig.app.json"

_EXPECTED_EXCLUDES = {
    "src/**/*.test.ts",
    "src/**/*.test.tsx",
    "../apps/shared/src/**/*.test.ts",
    "../apps/shared/src/**/*.test.tsx",
}


def _load_tsconfig() -> dict:
    # tsconfig comments in this file always occupy whole lines (the "paths" keys
    # contain "/*" inside strings, so a global strip would eat the aliases too).
    raw = _WEB_TSCONFIG.read_text(encoding="utf-8")
    stripped = re.sub(
        r"^[ \t]*/\*.*?\*/[ \t]*$", "", raw, flags=re.MULTILINE | re.DOTALL
    )
    return json.loads(stripped)


def test_app_project_excludes_all_vitest_suites():
    cfg = _load_tsconfig()
    assert set(cfg.get("exclude", [])) >= _EXPECTED_EXCLUDES


def test_test_files_still_exist_to_exclude():
    """The guard is meaningful only while real suites sit inside the include roots."""
    web_root = _WEB_TSCONFIG.parent
    app_tests = list(web_root.glob("src/**/*.test.ts*"))
    shared_tests = list((web_root / "../apps/shared/src").resolve().glob("*.test.ts"))
    assert app_tests and shared_tests

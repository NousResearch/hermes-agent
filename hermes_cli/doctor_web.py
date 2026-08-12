"""Read-only, advisory dashboard bundle provenance checks."""

import json
import os
from pathlib import Path

from hermes_cli.doctor_report import (
    Finding, _section, check_info, check_ok, check_warn, doctor_check,
)


_WEB_BUILD_PROVENANCE_FILE = "build-provenance.json"


def _active_web_dist() -> Path:
    """Return the dashboard bundle this process would serve."""
    from hermes_cli.doctor import PROJECT_ROOT

    return (Path(os.environ["HERMES_WEB_DIST"]) if "HERMES_WEB_DIST" in os.environ
            else PROJECT_ROOT / "hermes_cli" / "web_dist")


def _load_web_bundle_provenance(dist_root: Path) -> tuple[dict | None, str | None]:
    """Read and minimally validate Vite's fail-open provenance record."""
    provenance_path = dist_root / _WEB_BUILD_PROVENANCE_FILE
    try:
        data = json.loads(provenance_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "record is missing"
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"record is unreadable: {exc}"

    required = {"schemaVersion", "commitSha", "branch", "dirty", "builtAt", "invocation"}
    if not isinstance(data, dict) or not required.issubset(data):
        return None, "record has an invalid schema"
    if data["schemaVersion"] != 1:
        return None, f"unsupported schema version {data['schemaVersion']!r}"
    if data["dirty"] is not None and not isinstance(data["dirty"], bool):
        return None, "record has an invalid dirty flag"
    for key in ("commitSha", "branch"):
        if data[key] is not None and not isinstance(data[key], str):
            return None, f"record has an invalid {key}"
    for key in ("builtAt", "invocation"):
        if not isinstance(data[key], str) or not data[key].strip():
            return None, f"record has an invalid {key}"
    return data, None


def _current_source_head() -> str | None:
    """Resolve the source/package revision doctor is running against."""
    from hermes_cli.doctor import PROJECT_ROOT
    try:
        from hermes_cli._subprocess_compat import bounded_git_probe

        current = bounded_git_probe(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], timeout=2
        )
    except Exception:
        current = ""
    if current:
        return current

    embedded = os.environ.get("HERMES_REVISION", "").strip()
    if embedded:
        return embedded
    try:
        from hermes_cli.build_info import get_build_sha

        return get_build_sha(short=0)
    except Exception:
        return None


def _short_revision(revision: str | None) -> str:
    return revision[:12] if revision else "unknown"


@doctor_check("Web bundle provenance unavailable: {e}")
def _check_web_bundle_provenance(should_fix: bool, f: Finding) -> None:
    """Report bundle source identity and drift without modifying either tree."""
    _section("Web Dashboard Bundle")
    dist_root = _active_web_dist()
    if not (dist_root / "index.html").is_file():
        check_info(f"Web dashboard bundle not present at {dist_root}")
        return

    provenance, error = _load_web_bundle_provenance(dist_root)
    if provenance is None:
        check_warn(
            "Web bundle provenance unavailable",
            f"({error}; {dist_root / _WEB_BUILD_PROVENANCE_FILE})",
        )
        return

    deployed = provenance["commitSha"] or None
    branch = provenance["branch"] or "unknown branch"
    dirty = provenance["dirty"]
    identity = f"{_short_revision(deployed)} ({branch})"
    if dirty is True:
        check_warn("Web bundle was built from a dirty tree", f"({identity})")
    elif dirty is False:
        check_ok(f"Web bundle source: {identity}, clean")
    else:
        check_warn("Web bundle dirty state is unknown", f"({identity})")

    check_info(f"Built at: {provenance['builtAt']}")
    check_info(f"Build invocation: {provenance['invocation']}")

    current = _current_source_head()
    if not deployed:
        check_warn("Web bundle source commit is unknown")
    elif not current:
        check_info("Current source HEAD is unavailable; drift cannot be compared")
    elif deployed.lower() == current.lower():
        check_ok(
            "Web bundle matches current source HEAD",
            f"({_short_revision(current)})",
        )
    else:
        check_warn(
            "Web bundle differs from current source HEAD",
            f"(deployed {_short_revision(deployed)}; HEAD {_short_revision(current)})",
        )



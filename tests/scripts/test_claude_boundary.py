"""Pytest wrapper for scripts/check_claude_boundary.py.

Runs the boundary checker as part of the suite so a regression is caught even
when the dedicated CI job is not what a contributor is looking at.
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_claude_boundary.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("_claude_boundary", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves annotations via
    # sys.modules[cls.__module__], which fails for an unregistered module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_boundary_check_passes_on_the_current_tree():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Claude boundary check failed:\n{result.stdout}\n{result.stderr}"
    )


def test_every_allowlisted_path_has_a_justification():
    """An allow-list entry without a reason is an untracked exemption."""
    checker = _load_checker()
    listed = {p for rule in checker.RULES for p in rule.allowlist}
    missing = sorted(p for p in listed if not checker.ALLOWLIST_REASONS.get(p))
    assert not missing, f"allow-listed with no justification: {missing}"


def test_legacy_path_exemptions_are_tagged_for_removal():
    """Exemptions covering the pre-SDK path must name the PR that deletes them.

    Without the tag the allow-list silently becomes permanent; with it, the
    check tightens on its own as the legacy path is removed.
    """
    checker = _load_checker()
    legacy_prefixes = ("agent/anthropic_adapter.py", "agent/account_usage.py")
    for prefix in legacy_prefixes:
        reason = checker.ALLOWLIST_REASONS[prefix]
        assert "TODO(legacy-retirement)" in reason, f"{prefix} exemption is not tagged for removal"


def test_rules_cover_each_documented_category():
    """All four documented rule categories must actually be enforced."""
    checker = _load_checker()
    names = {rule.name for rule in checker.RULES}
    for expected in (
        "claude-credential-file",           # 1. credential access
        "anthropic-endpoint-with-claude-token",  # 2. direct OAuth inference
        "claude-code-user-agent",           # 3. identity spoofing
        "claude-oauth-token-env",           # 4. token env reads
    ):
        assert expected in names, f"boundary rule '{expected}' is missing"


def test_legitimate_anthropic_api_usage_is_not_flagged():
    """The API-key provider's own endpoint calls must stay allowed."""
    checker = _load_checker()
    rule = next(
        r for r in checker.RULES if r.name == "anthropic-endpoint-with-claude-token"
    )
    benign = 'req = urllib.request.Request("https://api.anthropic.com/v1/models")'
    assert rule.pattern.search(benign)
    assert rule.require_also is not None
    assert not rule.require_also.search(benign), (
        "a plain Anthropic API call must not trip the direct-OAuth rule"
    )


def test_spoofed_identity_patterns_are_detected():
    """The spoofing rules must match the shapes they exist to forbid."""
    checker = _load_checker()
    by_name = {r.name: r for r in checker.RULES}
    samples = {
        "claude-code-user-agent": '"user-agent": f"claude-code/{ver} (external, cli)"',
        "claude-code-cli-app-header": '"x-app": "cli",',
        "claude-code-beta-flag": '"anthropic-beta": "claude-code-20250219"',
        "claude-code-system-prompt": 'P = "You are Claude Code, Anthropic\'s official CLI."',
        "claude-code-identity-rewrite": 'text = text.replace("Hermes Agent", "Claude Code")',
    }
    for name, sample in samples.items():
        assert by_name[name].pattern.search(sample), f"{name} failed to match its own shape"


def test_inline_suppression_marker_is_honored():
    checker = _load_checker()
    assert checker.SUPPRESS_MARKER.search("x = 1  # claude-boundary: ok — why")
    assert not checker.SUPPRESS_MARKER.search("x = 1  # ok")


def test_docstring_documents_the_allowlist_policy():
    """The script is the audit trail; it must say so."""
    checker = _load_checker()
    assert re.search(r"ALLOW-LIST POLICY", checker.__doc__ or "")

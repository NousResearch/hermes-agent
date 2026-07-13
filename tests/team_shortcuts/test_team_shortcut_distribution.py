from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEAM = ROOT / "team-shortcuts"


def _table_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("| `"))


def _skill_rows(path: Path) -> int:
    active = False
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "## Shortcut Map":
            active = True
            continue
        if active and line.startswith("## "):
            break
        if active and line.startswith("| `"):
            count += 1
    return count


def test_distribution_counts_are_consistent_without_hardcoded_old_totals():
    payload = TEAM / "payload"
    registry = payload / "ai-context/prompt-shortcut-registry.md"
    skill = payload / "skills/prompt-shortcuts/SKILL.md"
    index = payload / "skills/prompt-shortcuts/Prompt Shortcuts.md"
    refs = payload / "skills/prompt-shortcuts/references"

    registry_count = _table_rows(registry)
    assert registry_count == _skill_rows(skill)
    assert registry_count == _table_rows(index)
    assert len(list(refs.glob("*.md"))) >= registry_count


def test_distribution_has_traceable_version_and_required_runtime_tools():
    version = (TEAM / "VERSION").read_text(encoding="utf-8").strip()
    installer = (TEAM / "install-shortcuts.sh").read_text(encoding="utf-8")
    checker = (TEAM / "check-shortcuts.sh").read_text(encoding="utf-8")

    assert version == "2026.07.13-4"
    assert "INSTALLED_VERSION" in installer
    assert "ไม่พบตัวตรวจสุขภาพ Hook" in installer
    assert "registry_vs_skill" in checker
    assert '"29"' not in checker
    assert '"33"' not in checker


def test_github_installer_sets_up_shortcuts_relay_and_shell_path():
    installer = (TEAM / "install-from-github.sh").read_text(encoding="utf-8")

    assert 'RELAY_SRC="$ARCHIVE_ROOT/scripts/ai-relay"' in installer
    assert 'bash "$RELAY_DIR/scripts/ai-relay/install-local.sh"' in installer
    assert 'ensure_local_bin_path "$HOME/.zshrc"' in installer
    assert 'ensure_local_bin_path "$HOME/.bashrc"' in installer
    assert "codex login" in installer
    assert "relay-status --probe" in installer


def test_team_installer_includes_real_stop_hooks_for_fresh_notebooks():
    installer = (TEAM / "install-shortcuts.sh").read_text(encoding="utf-8")
    hook_installer = (TEAM / "install-team-hooks.py").read_text(encoding="utf-8")
    hook_dir = TEAM / "hooks"

    assert 'python3 "$TEAM_HOOK_INSTALLER"' in installer
    assert '".claude" / "settings.json"' in hook_installer
    assert '".codex" / "hooks.json"' in hook_installer
    assert (hook_dir / "validate-thai-language.py").is_file()
    assert (hook_dir / "enforce-codex-review.py").is_file()
    assert (hook_dir / "enforce-prompt-evidence.py").is_file()
    assert (hook_dir / "team-stop-gates.py").is_file()

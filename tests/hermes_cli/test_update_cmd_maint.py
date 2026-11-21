"""#143470 follow-through: the all-profile post-update skill reporter must surface deferred
relocations instead of reporting a profile whose copy is stranded behind a category symlink
as "up to date"."""

import types
from unittest.mock import patch

from hermes_cli.update_cmd_maint import _profile_skill_sync_status, _sync_profiles_after_update


class TestProfileSkillSyncStatus:
    def test_relocation_warnings_prevent_up_to_date(self):
        r = {"copied": [], "updated": [], "relocation_warnings": ["demo: stranded behind a category symlink"]}
        status = _profile_skill_sync_status(r)
        assert "up to date" not in status, "a deferred relocation is not a clean sync"
        assert "1 relocation(s) deferred" in status

    def test_counts_combine_with_copy_activity(self):
        r = {"copied": ["a"], "relocation_warnings": ["w1", "w2"]}
        status = _profile_skill_sync_status(r)
        assert "+1 new" in status
        assert "2 relocation(s) deferred" in status

    def test_clean_result_still_reads_up_to_date(self):
        assert _profile_skill_sync_status({"copied": [], "updated": []}) == "up to date"

    def test_opt_out_and_failure_paths_unchanged(self):
        assert _profile_skill_sync_status({"skipped_opt_out": True}) == "opted out (--no-skills)"
        assert _profile_skill_sync_status(None) == "sync failed"


class TestSyncProfilesAfterUpdate:
    def test_profile_report_prints_deferred_relocation_warnings(self, capsys):
        warning = "demo: upstream moved this skill, but the copy is only reachable through a category symlink."
        result = {"copied": [], "updated": [], "relocation_warnings": [warning]}
        profile = types.SimpleNamespace(name="work", path="/tmp/hermes-profiles/work")
        with patch("hermes_cli.profiles.list_profiles", return_value=[profile]), \
                patch("hermes_cli.profiles.seed_profile_skills", return_value=result), \
                patch("hermes_cli.profiles.backfill_profile_envs", return_value=[]):
            _sync_profiles_after_update()
        out = capsys.readouterr().out
        assert "→ Syncing bundled skills to all profiles..." in out, "patched profile loop did not run"
        assert "work: ⚠ 1 relocation(s) deferred" in out, "status line hid the deferred relocation"
        assert warning in out, "the profile line must be followed by the warning itself"
        assert "up to date" not in out

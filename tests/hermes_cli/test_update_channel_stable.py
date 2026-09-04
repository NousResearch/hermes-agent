"""Tests for the stable update channel (tag-tracking) in hermes_cli/update_cmd.py."""

from unittest.mock import patch

from hermes_cli.update_cmd import (
    _latest_release_tag_from_ls_remote,
    _parse_release_tag,
    _stable_channel_active,
)


class TestParseReleaseTag:
    def test_final_releases_parse(self):
        assert _parse_release_tag("v0.17.0") == (0, 17, 0)
        assert _parse_release_tag("v10.2.33") == (10, 2, 33)
        assert _parse_release_tag(" v1.2.3 ") == (1, 2, 3)

    def test_prereleases_and_garbage_rejected(self):
        for tag in ("v1.2.3-rc1", "v1.2.3-beta.1", "v1.2", "1.2.3", "release-1", "vv1.2.3", ""):
            assert _parse_release_tag(tag) is None, tag

    def test_canary_tags_never_reach_the_stable_channel(self):
        """The inverse direction of the canary channel: canary tags are
        version-suffixed (v0.28.0-canary.20260818), and the stable
        selector's no-suffix rule is what keeps a canary prerelease from
        ever winning a stable-channel update. If this shape ever parses,
        stable users get canaries."""
        assert _parse_release_tag("v0.28.0-canary.20260818") is None
        assert _parse_release_tag("v1.0.0-canary.1") is None

    def test_calver_tags_rejected(self):
        """Historical CalVer tags (v2026.7.20) must not win a numeric sort.

        The major component is capped at three digits, the same rule as
        _SEMVER_TAG_RE in scripts/write_install_stamp.py and
        latestReleaseFromLsRemote in apps/desktop. A four-digit year would
        rank above every SemVer release forever.
        """
        assert _parse_release_tag("v2026.7.20") is None
        assert _parse_release_tag("v1000.0.0") is None
        assert _parse_release_tag("v999.0.0") == (999, 0, 0)

    def test_numeric_ordering_not_lexicographic(self):
        """v0.10.0 must sort above v0.9.0 — the whole point of tuple parsing."""
        newer, older = _parse_release_tag("v0.10.0"), _parse_release_tag("v0.9.0")
        assert newer is not None and older is not None
        assert newer > older


class TestLatestReleaseTagFromLsRemote:
    def test_picks_newest_final_release(self):
        output = (
            "aaa1\trefs/tags/v0.9.0\n"
            "bbb2\trefs/tags/v0.10.0\n"
            "ccc3\trefs/tags/v0.10.1-rc1\n"
            "ddd4\trefs/tags/some-other-tag\n"
        )
        tag, sha = _latest_release_tag_from_ls_remote(output)
        assert tag == "v0.10.0"
        assert sha == "bbb2"

    def test_peeled_sha_wins_for_annotated_tags(self):
        output = (
            "tagobj\trefs/tags/v1.0.0\n"
            "commitsha\trefs/tags/v1.0.0^{}\n"
        )
        tag, sha = _latest_release_tag_from_ls_remote(output)
        assert tag == "v1.0.0"
        assert sha == "commitsha"

    def test_no_release_tags(self):
        assert _latest_release_tag_from_ls_remote("aaa\trefs/tags/nightly\n") == (None, None)
        assert _latest_release_tag_from_ls_remote("") == (None, None)

    def test_malformed_lines_ignored(self):
        output = "garbage line no tab\naaa\trefs/heads/main\nbbb\trefs/tags/v2.0.0\n"
        assert _latest_release_tag_from_ls_remote(output) == ("v2.0.0", "bbb")


class _Args:
    def __init__(self, branch=None, channel=None):
        self.branch = branch
        self.channel = channel


class TestStableChannelActive:
    def test_explicit_branch_always_wins(self):
        """--branch means main-style behavior regardless of channel config."""
        assert _stable_channel_active(_Args(branch="bb/gui")) is False

    def test_transient_channel_flag_wins(self):
        """--channel is the per-invocation override (--set-channel persists);
        no config read happens when it is present."""
        assert _stable_channel_active(_Args(channel="stable")) is True
        assert _stable_channel_active(_Args(channel="main")) is False
        # canary on a source tree normalizes to main, never stable.
        assert _stable_channel_active(_Args(channel="canary")) is False

    def test_per_install_record_activates(self, tmp_path, monkeypatch):
        from hermes_cli.update_channel import install_id

        root = tmp_path / "install"
        root.mkdir()
        (root / "install-stamp.json").write_text(
            '{"schemaVersion": 2, "updateMechanism": "self"}'
        )
        config = {
            "update": {"installs": {install_id(root): {"path": str(root), "channel": "stable"}}}
        }
        import hermes_cli.update_cmd as update_cmd

        monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", root)
        with patch("hermes_cli.config.load_config", return_value=config):
            assert _stable_channel_active(_Args()) is True

    def test_no_record_stays_main(self, tmp_path, monkeypatch):
        root = tmp_path / "install"
        root.mkdir()
        (root / "install-stamp.json").write_text(
            '{"schemaVersion": 2, "updateMechanism": "self"}'
        )
        import hermes_cli.update_cmd as update_cmd

        monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", root)
        with patch("hermes_cli.config.load_config", return_value={"update": {"installs": {}}}):
            assert _stable_channel_active(_Args()) is False

    def test_config_failure_defaults_to_main(self):
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            assert _stable_channel_active(_Args()) is False

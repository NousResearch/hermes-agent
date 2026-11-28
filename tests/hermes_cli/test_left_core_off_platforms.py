"""A platform that left core is off for every other row whose tools lived in its own platform's bundle only:
derived from the rows, so the order two platform moves merge in cannot leave one off the other's list."""

from hermes_cli import left_core_migration as lcm


def _row(platform, **kw):
    return lcm.LeftCoreFeature(plugin=platform, label=platform, in_use=lambda *a, **k: False, unchanged="",
                               platform=platform, toolsets=(f"{platform}_tools",), **kw)


def test_a_new_moved_platform_is_off_for_every_own_platform_row():
    rows = lcm._derive_off_platforms((
        _row("standin_a", own_platform_only=True), _row("standin_b", own_platform_only=True),
        _row("standin_c", off_platforms=("webhook",))))
    a, b, c = rows
    assert "standin_b" in a.off_platforms and "standin_c" in a.off_platforms and "standin_a" not in a.off_platforms
    assert "standin_a" in b.off_platforms and "standin_b" not in b.off_platforms
    assert set(lcm._core_platforms_except("standin_a")) <= set(a.off_platforms)
    assert c.off_platforms == ("webhook",)  # a hand list (a toolset core had on elsewhere) is left alone


def test_shipped_rows_list_every_other_moved_platform():
    moved = {f.platform for f in lcm.LEFT_CORE if f.platform}
    for row in (f for f in lcm.LEFT_CORE if f.own_platform_only):
        assert moved - {row.platform} <= set(row.off_platforms), row.plugin
        assert row.platform not in row.off_platforms

"""Generated workspaces must not solve for Android.

Regression guard for the update-time plugin eviction (2026-10-23): uv's universal
resolution solves every platform named by ``tool.uv.environments``, Android
included, and a plugin-carried native transitive (torch behind the nerve plugin's
optional ``laya`` extra) has no Python 3.14 Android wheels — one unsolvable
platform fails the whole workspace lock, and ``hermes update`` then evicts the
plugin. The generated workspace installs on one host, so its solve is narrowed to
non-Android platforms on a non-Android host (``pm.workspace._generate_pyproject``).
"""

from pm.workspace import _exclude_android_from_solve


def test_existing_environments_lose_android_only():
    uv_policy = {"environments": ["python_version >= '3.14'"]}
    _exclude_android_from_solve(uv_policy)
    assert uv_policy["environments"] == [
        "python_version >= '3.14' and sys_platform != 'android'"
    ]


def test_multiple_environment_entries_all_narrowed():
    uv_policy = {"environments": ["python_version >= '3.14'", "python_version < '3.14'"]}
    _exclude_android_from_solve(uv_policy)
    assert all(entry.endswith(" and sys_platform != 'android'")
               for entry in uv_policy["environments"])
    assert len(uv_policy["environments"]) == 2


def test_missing_environments_gets_platform_only_entry():
    uv_policy: dict = {}
    _exclude_android_from_solve(uv_policy)
    assert uv_policy["environments"] == ["sys_platform != 'android'"]


def test_unrelated_uv_policy_keys_untouched():
    uv_policy = {"default-groups": [], "environments": ["python_version >= '3.14'"]}
    _exclude_android_from_solve(uv_policy)
    assert uv_policy["default-groups"] == []
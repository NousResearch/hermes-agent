"""_merge_profile_tree folds projects across profiles by normalized path."""


def _make_project(path=None, pid="p_abc", is_auto=False, label="Proj"):
    """Minimal project dict matching the shape _merge_profile_tree expects."""
    return {
        "id": pid,
        "path": path,
        "label": label,
        "isAuto": is_auto,
        "repos": [],
        "previewSessions": [],
        "sessionCount": 0,
        "totalTokens": 0,
        "totalCostUsd": 0,
        "lastActive": 0,
    }


class TestMergeProfileTree:
    """Unit tests for the merge-key normalization in _merge_profile_tree."""

    def test_same_path_different_casing_produces_one_entry(self):
        """Windows case-insensitive paths must fold into one group on any host.

        The canonical folder key is shape-derived (``_path_key``): a drive-letter
        path case-folds even when the backend runs on POSIX, so this needs no
        monkeypatching — it exercises the real normalization.
        """
        from hermes_cli.web_routers.profiles import _merge_profile_tree

        merged = {}
        proj_a = _make_project(path=r"C:\Users\me\Project", pid="p_a")
        proj_b = _make_project(path=r"C:\Users\me\project", pid="p_b")

        _merge_profile_tree(merged, [proj_a], "alpha", preview_limit=5)
        _merge_profile_tree(merged, [proj_b], "beta", preview_limit=5)

        assert len(merged) == 1
        # First writer wins the identity slot.
        assert next(iter(merged.values()))["id"] == "p_a"

    def test_windows_forward_slash_spelling_folds(self):
        """``C:/Users/me/Project`` and ``C:\\Users\\me\\Project`` are one folder."""
        from hermes_cli.web_routers.profiles import _merge_profile_tree

        merged = {}
        proj_a = _make_project(path=r"C:\Users\me\Project", pid="p_a")
        proj_b = _make_project(path="C:/Users/me/Project", pid="p_b")

        _merge_profile_tree(merged, [proj_a], "alpha", preview_limit=5)
        _merge_profile_tree(merged, [proj_b], "beta", preview_limit=5)

        assert len(merged) == 1

    def test_accented_folder_nfd_nfc_spellings_fold(self):
        """The same accented folder may arrive NFC (typed) or NFD (macOS pickers) —
        one group (#65014 class)."""
        from hermes_cli.web_routers.profiles import _merge_profile_tree

        merged = {}
        proj_a = _make_project(path="/home/me/café", pid="p_a")
        proj_b = _make_project(path="/home/me/café", pid="p_b")

        _merge_profile_tree(merged, [proj_a], "alpha", preview_limit=5)
        _merge_profile_tree(merged, [proj_b], "beta", preview_limit=5)

        assert len(merged) == 1

    def test_trailing_slash_does_not_duplicate(self):
        """A trailing separator doesn't create a second folder identity."""
        from hermes_cli.web_routers.profiles import _merge_profile_tree

        merged = {}
        proj_a = _make_project(path="/home/me/proj", pid="p_a")
        proj_b = _make_project(path="/home/me/proj/", pid="p_b")

        _merge_profile_tree(merged, [proj_a], "alpha", preview_limit=5)
        _merge_profile_tree(merged, [proj_b], "beta", preview_limit=5)

        assert len(merged) == 1

    def test_different_paths_stay_separate(self):
        """Genuinely different folders must remain distinct groups."""
        from hermes_cli.web_routers.profiles import _merge_profile_tree

        merged = {}
        proj_a = _make_project(path="/home/me/alpha", pid="p_a")
        proj_b = _make_project(path="/home/me/beta", pid="p_b")

        _merge_profile_tree(merged, [proj_a], "alpha", preview_limit=5)
        _merge_profile_tree(merged, [proj_b], "beta", preview_limit=5)

        assert len(merged) == 2

    def test_id_fallback_when_path_is_none(self):
        """Projects without a path key on the project id instead."""
        from hermes_cli.web_routers.profiles import _merge_profile_tree

        merged = {}
        proj_a = _make_project(path=None, pid="p_same")
        proj_b = _make_project(path=None, pid="p_same")

        _merge_profile_tree(merged, [proj_a], "alpha", preview_limit=5)
        _merge_profile_tree(merged, [proj_b], "beta", preview_limit=5)

        assert len(merged) == 1

    def test_declared_project_wins_over_auto(self):
        """When a declared project meets an auto entry, declared wins identity."""
        from hermes_cli.web_routers.profiles import _merge_profile_tree

        merged = {}
        auto = _make_project(path="/home/me/proj", pid="p_auto", is_auto=True, label="auto")
        declared = _make_project(path="/home/me/proj", pid="p_decl", is_auto=False, label="My Proj")

        _merge_profile_tree(merged, [auto], "alpha", preview_limit=5)
        _merge_profile_tree(merged, [declared], "beta", preview_limit=5)

        assert len(merged) == 1
        assert next(iter(merged.values()))["label"] == "My Proj"

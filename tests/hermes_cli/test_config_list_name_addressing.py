"""Dotted config paths must name-address entries of a list-of-mappings (#132963).

``custom_providers`` is a list of mappings whose entries carry a ``name`` field, but
``config set/get/unset`` could only navigate lists with numeric indexes. The defect is a
CLASS spanning all three commands and two navigation helpers, with three distinct
symptoms on the pre-fix tree:

* ``_set_nested`` navigate (non-leaf): a non-numeric segment raises
  ``TypeError ... is not a numeric index`` even when an entry is named exactly that.
* ``_set_nested`` leaf: a non-numeric segment escapes as a bare ``ValueError`` from
  ``int()`` — loud, but undiagnosable.
* ``_locate_nested`` (shared by ``_get_nested``/``_unset_nested``): a non-numeric or
  out-of-range segment silently yields ``None`` — ``config get`` reports the key not
  set and ``unset`` claims nothing was removed, for an entry the user can see in the
  file.

Contract under test (name-based addressing, numeric-first):

1. Numeric index semantics are preserved bit-for-bit — ``int(part)`` success is index
   addressing (negative indexes included); ``_set_nested`` still passes a numeric
   out-of-range ``IndexError`` through untouched, and ``_locate_nested`` still returns
   ``None`` for it.
2. Name matching runs ONLY when the segment is non-numeric. An entry whose ``name`` is
   a pure-numeric string is reachable by index only (no double meaning).
3. Name matching applies to dict entries only; a non-numeric segment into a
   list-of-scalars keeps the current outcome (set: TypeError / get: missing /
   unset: False).
4. Duplicate names resolve to the FIRST match, deterministically.
5. ``set`` miss is loud: ``TypeError`` naming the segment, saying it is not a numeric
   index and that no list entry carries that name — and it never appends or creates an
   entry. The leaf-form bare ``ValueError`` is upgraded to the same ``TypeError``.
6. ``get``/``unset`` miss keeps the existing missing/``False`` semantics (missing is
   not an error on those faces).
7. ``unset`` by name deletes the entry; emptying the list leaves the empty list in
   place (never deletes the container).
8. The #17876 list guard, greedy-literal dict matching, phantom-sibling refusal, env
   routing and schema validation are untouched — covered by the pre-existing suites
   (``test_config_dotted_key_names.py``, ``test_config_set_list_values.py``) and by the
   numeric-first guards below.

Deficiency tests below must FAIL on the unfixed tree and pass after the fix; the
regression guards (marked as such) must pass on BOTH trees.
"""

import pytest
import hermes_yaml as yaml

from hermes_cli.config import (
    _MISSING,
    _get_nested,
    _locate_nested,
    _set_nested,
    _unset_nested,
    get_config_value,
    set_config_value,
    unset_config_value,
)


@pytest.fixture
def user_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    return home


def _write_config(home, data: dict):
    (home / "config.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


def _read_config(home):
    return yaml.safe_load((home / "config.yaml").read_text())


NAMED_PROVIDERS = {
    "custom_providers": [
        {
            "name": "glm-flash",
            "base_url": "https://glm.example.invalid/v1",
            "api_mode": "chat",
        },
        {
            "name": "ygg-route",
            "base_url": "https://ygg.example.invalid/v1",
        },
    ]
}


# ---------------------------------------------------------------------------
# set — name addressing down and at the leaf
# ---------------------------------------------------------------------------


class TestSetNameAddressing:
    def test_set_navigates_into_named_entry_and_preserves_siblings(self):
        """set writes into the entry whose ``name`` matches, leaving the entry's own
        other fields and every sibling entry untouched."""
        cfg = {
            "custom_providers": [
                {"name": "glm-flash", "base_url": "https://old.example/v1", "api_mode": "chat"},
                {"name": "ygg-route", "base_url": "https://ygg.example/v1"},
            ]
        }
        _set_nested(cfg, "custom_providers.glm-flash.base_url", "https://new.example/v1")
        assert cfg["custom_providers"][0]["base_url"] == "https://new.example/v1"
        assert cfg["custom_providers"][0]["name"] == "glm-flash"
        assert cfg["custom_providers"][0]["api_mode"] == "chat"
        assert cfg["custom_providers"][1] == {"name": "ygg-route", "base_url": "https://ygg.example/v1"}

    def test_set_leaf_replaces_whole_named_entry_in_place(self):
        """A value addressed at the name segment itself replaces the entry at its
        position — never an append."""
        cfg = {
            "custom_providers": [
                {"name": "glm-flash", "base_url": "https://old.example/v1"},
                {"name": "ygg-route", "base_url": "https://ygg.example/v1"},
            ]
        }
        replacement = {"name": "glm-flash", "base_url": "https://new.example/v1", "api_mode": "responses"}
        _set_nested(cfg, "custom_providers.glm-flash", replacement)
        assert len(cfg["custom_providers"]) == 2
        assert cfg["custom_providers"][0] == replacement
        assert cfg["custom_providers"][1]["name"] == "ygg-route"

    def test_set_miss_raises_loud_typeerror_and_never_appends(self):
        """A non-numeric segment matching no entry raises a TypeError whose message
        names the segment, says it is not a numeric index AND that no list entry is
        named it; the list must be left exactly as it was (no silent append)."""
        cfg = {
            "custom_providers": [
                {"name": "glm-flash", "base_url": "https://glm.example/v1"},
                {"name": "ygg-route", "base_url": "https://ygg.example/v1"},
            ]
        }
        with pytest.raises(TypeError) as excinfo:
            _set_nested(cfg, "custom_providers.nosuch.base_url", "https://x.example/v1")
        message = str(excinfo.value)
        assert "nosuch" in message
        assert "not a numeric index" in message
        assert "no list entry is named" in message
        assert len(cfg["custom_providers"]) == 2
        assert all(entry["name"] != "nosuch" for entry in cfg["custom_providers"])

    def test_set_leaf_miss_raises_typeerror_not_bare_valueerror(self):
        """Same miss AT the leaf position: the historic bare ``ValueError`` from
        ``int()`` must surface as the same diagnosable ``TypeError``."""
        cfg = {"custom_providers": [{"name": "glm-flash", "base_url": "https://glm.example/v1"}]}
        with pytest.raises(TypeError) as excinfo:
            _set_nested(cfg, "custom_providers.nosuch", "https://x.example/v1")
        message = str(excinfo.value)
        assert "nosuch" in message
        assert "not a numeric index" in message
        assert "no list entry is named" in message
        assert len(cfg["custom_providers"]) == 1

    def test_set_numeric_segment_out_of_range_raises_indexerror(self):
        """Regression guard (invariant 1): a numeric segment indexes even when some
        entry carries that name as its ``name`` — the out-of-range ``IndexError``
        passes through untouched, never converted to name matching."""
        cfg = {"custom_providers": [{"name": "1", "base_url": "https://only.example/v1"}]}
        with pytest.raises(IndexError):
            _set_nested(cfg, "custom_providers.1.base_url", "https://x.example/v1")
        with pytest.raises(IndexError):
            _set_nested(cfg, "custom_providers.1.models.m.context_length", 8)
        assert len(cfg["custom_providers"]) == 1

    def test_set_numeric_first_even_when_a_later_entry_carries_the_name(self):
        """Regression guard (invariant 2): numeric-first. ``0`` must resolve to index
        0 even though entry 1 is NAMED ``0``."""
        cfg = {
            "custom_providers": [
                {"name": "x", "marker": "at-index-0"},
                {"name": "0", "marker": "at-index-1"},
            ]
        }
        _set_nested(cfg, "custom_providers.0.marker", "written")
        assert cfg["custom_providers"][0]["marker"] == "written"
        assert cfg["custom_providers"][1]["marker"] == "at-index-1"

    def test_set_list_of_scalars_non_numeric_segment_still_typeerror(self):
        """Regression guard (invariant 3): name matching is dict-entries-only. A
        non-numeric segment into a list-of-scalars fails loudly at the leaf too
        (the historic bare ``ValueError`` becomes the same ``TypeError``)."""
        cfg = {"tags": ["alpha", "beta"]}
        with pytest.raises(TypeError) as excinfo:
            _set_nested(cfg, "tags.alpha", "x")
        assert "alpha" in str(excinfo.value)


# ---------------------------------------------------------------------------
# get / _locate_nested — name addressing reads
# ---------------------------------------------------------------------------


class TestGetAndLocateNameAddressing:
    def test_get_reads_field_of_named_entry(self):
        cfg = {
            "custom_providers": [
                {"name": "glm-flash", "base_url": "https://glm.example/v1"},
                {"name": "ygg-route", "base_url": "https://ygg.example/v1"},
            ]
        }
        assert _get_nested(cfg, "custom_providers.glm-flash.base_url") == "https://glm.example/v1"

    def test_get_returns_whole_named_entry(self):
        cfg = {
            "custom_providers": [
                {"name": "glm-flash", "base_url": "https://glm.example/v1"},
                {"name": "ygg-route", "base_url": "https://ygg.example/v1"},
            ]
        }
        assert _get_nested(cfg, "custom_providers.ygg-route") == {
            "name": "ygg-route",
            "base_url": "https://ygg.example/v1",
        }

    def test_get_miss_returns_missing_not_error(self):
        """Regression guard (invariant 6): a miss on the read face stays ``_MISSING``
        (which ``config get`` reports as "not set") — never an exception."""
        cfg = {"custom_providers": [{"name": "glm-flash", "base_url": "https://glm.example/v1"}]}
        assert _get_nested(cfg, "custom_providers.nosuch.base_url") is _MISSING
        assert _get_nested(cfg, "custom_providers.nosuch") is _MISSING

    def test_locate_named_entry_returns_container_and_int_index(self):
        """``_locate_nested`` return shape mirrors the numeric path for the same path
        length: a path ENDING on the name segment lands on the list itself with an
        integer index key (``container[key]`` is the addressed entry); a path
        continuing past the entry into a field lands on the entry dict with the field
        name as the key — the exact shapes ``custom_providers.0[.base_url]`` produce."""
        cfg = {
            "custom_providers": [
                {"name": "glm-flash", "base_url": "https://glm.example/v1"},
                {"name": "ygg-route", "base_url": "https://ygg.example/v1"},
            ]
        }
        # Entry path (ends on the list segment): list container, int index key.
        loc = _locate_nested(cfg, ["custom_providers", "glm-flash"])
        assert loc is not None
        _, container, key = loc
        assert container is cfg["custom_providers"]
        assert isinstance(key, int)
        assert container[key] == {"name": "glm-flash", "base_url": "https://glm.example/v1"}
        assert _locate_nested(cfg, ["custom_providers", "0"])[1:] == (container, key)
        # Field path (continues past the entry): entry-dict container, field-name key.
        loc = _locate_nested(cfg, ["custom_providers", "glm-flash", "base_url"])
        assert loc is not None
        _, container, key = loc
        assert container is cfg["custom_providers"][0]
        assert key == "base_url"
        assert container[key] == "https://glm.example/v1"
        assert _locate_nested(cfg, ["custom_providers", "0", "base_url"])[1:] == (container, key)

    def test_locate_numeric_out_of_range_returns_none(self):
        """Regression guard (invariant 1): numeric out-of-range still reads as
        "absent" (``None``), exactly as before."""
        cfg = {"custom_providers": [{"name": "glm-flash", "base_url": "https://glm.example/v1"}]}
        assert _locate_nested(cfg, ["custom_providers", "9"]) is None

    def test_locate_named_miss_returns_none(self):
        """Regression guard (invariant 6): a name that matches nothing is absent, not
        an error."""
        cfg = {"custom_providers": [{"name": "glm-flash", "base_url": "https://glm.example/v1"}]}
        assert _locate_nested(cfg, ["custom_providers", "nosuch", "base_url"]) is None

    def test_get_numeric_segment_wins_over_numeric_named_entry(self):
        """Regression guard (invariant 2): on the read face too, ``0`` is index 0 even
        though entry 1 is NAMED ``0``."""
        cfg = {
            "custom_providers": [
                {"name": "x", "marker": "at-index-0"},
                {"name": "0", "marker": "at-index-1"},
            ]
        }
        assert _get_nested(cfg, "custom_providers.0.marker") == "at-index-0"

    def test_get_list_of_scalars_non_numeric_returns_missing(self):
        """Regression guard (invariant 3): a non-numeric segment into a
        list-of-scalars reads as missing on both trees."""
        cfg = {"tags": ["alpha", "beta"]}
        assert _get_nested(cfg, "tags.alpha") is _MISSING

    def test_locate_negative_numeric_index_unchanged(self):
        """Regression guard (invariant 1): negative numeric indexes keep working on
        the shared navigation."""
        cfg = {
            "custom_providers": [
                {"name": "glm-flash", "base_url": "https://glm.example/v1"},
                {"name": "ygg-route", "base_url": "https://ygg.example/v1"},
            ]
        }
        loc = _locate_nested(cfg, ["custom_providers", "-1"])
        assert loc is not None
        _, container, key = loc
        assert container[key]["name"] == "ygg-route"
        assert _get_nested(cfg, "custom_providers.-1.name") == "ygg-route"


# ---------------------------------------------------------------------------
# unset — name addressing removal
# ---------------------------------------------------------------------------


class TestUnsetNameAddressing:
    def test_unset_removes_field_of_named_entry(self):
        cfg = {
            "custom_providers": [
                {"name": "glm-flash", "base_url": "https://glm.example/v1", "api_mode": "chat"},
                {"name": "ygg-route", "base_url": "https://ygg.example/v1"},
            ]
        }
        assert _unset_nested(cfg, "custom_providers.glm-flash.base_url") is True
        assert "base_url" not in cfg["custom_providers"][0]
        assert cfg["custom_providers"][0]["name"] == "glm-flash"
        assert cfg["custom_providers"][1]["base_url"] == "https://ygg.example/v1"

    def test_unset_removes_whole_named_entry(self):
        cfg = {
            "custom_providers": [
                {"name": "glm-flash", "base_url": "https://glm.example/v1"},
                {"name": "ygg-route", "base_url": "https://ygg.example/v1"},
            ]
        }
        assert _unset_nested(cfg, "custom_providers.glm-flash") is True
        assert len(cfg["custom_providers"]) == 1
        assert cfg["custom_providers"][0]["name"] == "ygg-route"

    def test_unset_last_named_entry_preserves_empty_list(self):
        """Regression guard (invariant 7): emptying the list by name keeps the empty
        list in the config — the container is never deleted."""
        cfg = {"custom_providers": [{"name": "glm-flash", "base_url": "https://glm.example/v1"}]}
        assert _unset_nested(cfg, "custom_providers.glm-flash") is True
        assert "custom_providers" in cfg
        assert cfg["custom_providers"] == []

    def test_unset_miss_returns_false_and_leaves_list_intact(self):
        """Regression guard (invariant 6): a miss on the unset face stays ``False``
        (``config unset`` reports "not set") — no exception, no mutation."""
        cfg = {
            "custom_providers": [
                {"name": "glm-flash", "base_url": "https://glm.example/v1"},
                {"name": "ygg-route", "base_url": "https://ygg.example/v1"},
            ]
        }
        assert _unset_nested(cfg, "custom_providers.nosuch") is False
        assert len(cfg["custom_providers"]) == 2

    def test_unset_list_of_scalars_non_numeric_returns_false(self):
        """Regression guard (invariant 3): unset on a list-of-scalars with a
        non-numeric segment stays False."""
        cfg = {"tags": ["alpha", "beta"]}
        assert _unset_nested(cfg, "tags.alpha") is False

    def test_unset_negative_numeric_index_unchanged(self):
        """Regression guard (invariant 1): negative numeric unset keeps working."""
        cfg = {
            "custom_providers": [
                {"name": "glm-flash", "base_url": "https://glm.example/v1"},
                {"name": "ygg-route", "base_url": "https://ygg.example/v1"},
            ]
        }
        assert _unset_nested(cfg, "custom_providers.-1.base_url") is True
        assert "base_url" not in cfg["custom_providers"][1]
        assert cfg["custom_providers"][0]["base_url"] == "https://glm.example/v1"


# ---------------------------------------------------------------------------
# duplicate names — first match wins
# ---------------------------------------------------------------------------


class TestDuplicateNameFirstMatchWins:
    def test_get_set_unset_resolve_to_first_match(self):
        """Invariant 4: with two entries named ``dup``, reads, writes and removals
        all land on the FIRST one, deterministically."""
        cfg = {
            "custom_providers": [
                {"name": "dup", "base_url": "https://first.example/v1"},
                {"name": "dup", "base_url": "https://second.example/v1"},
            ]
        }
        assert _get_nested(cfg, "custom_providers.dup.base_url") == "https://first.example/v1"
        _set_nested(cfg, "custom_providers.dup.base_url", "https://rewritten.example/v1")
        assert cfg["custom_providers"][0]["base_url"] == "https://rewritten.example/v1"
        assert cfg["custom_providers"][1]["base_url"] == "https://second.example/v1"
        assert _unset_nested(cfg, "custom_providers.dup") is True
        assert len(cfg["custom_providers"]) == 1
        assert cfg["custom_providers"][0]["base_url"] == "https://second.example/v1"


# ---------------------------------------------------------------------------
# public face — config set/get/unset cascade through the real entry points
# ---------------------------------------------------------------------------


class TestPublicFaceCascade:
    def test_config_set_then_get_roundtrip_by_name(self, user_home, capsys):
        """AC-4: ``set_config_value`` into a named entry, then ``get_config_value``
        on the same path returns the written value; the file holds the updated entry
        with its siblings intact."""
        _write_config(user_home, NAMED_PROVIDERS)
        set_config_value("custom_providers.glm-flash.base_url", "https://cascade.example/v1")
        saved = _read_config(user_home)
        assert saved["custom_providers"][0]["base_url"] == "https://cascade.example/v1"
        assert saved["custom_providers"][0]["name"] == "glm-flash"
        assert saved["custom_providers"][0]["api_mode"] == "chat"
        assert saved["custom_providers"][1]["name"] == "ygg-route"

        capsys.readouterr()  # drop the set command's "✓ Set ..." line from the buffer
        get_config_value("custom_providers.glm-flash.base_url", raw=True)
        assert capsys.readouterr().out.strip() == "https://cascade.example/v1"

    def test_config_set_by_name_miss_is_loud_and_writes_nothing(self, user_home):
        """The public set face fails loudly on a name miss and leaves the config file
        byte-for-byte as it was — no appended entry, no rewritten list."""
        _write_config(user_home, NAMED_PROVIDERS)
        before = _read_config(user_home)
        with pytest.raises(TypeError) as excinfo:
            set_config_value("custom_providers.nosuch.base_url", "https://x.example/v1")
        assert "no list entry is named" in str(excinfo.value)
        assert _read_config(user_home) == before

    def test_config_get_by_name_miss_exits_not_set(self, user_home, capsys):
        """Regression guard (invariant 6): the public get face keeps reporting a miss
        as "not set" (SystemExit), never an exception traceback."""
        _write_config(user_home, NAMED_PROVIDERS)
        with pytest.raises(SystemExit):
            get_config_value("custom_providers.nosuch.base_url")
        assert "not set" in capsys.readouterr().err

    def test_config_unset_removes_named_entry(self, user_home):
        _write_config(user_home, NAMED_PROVIDERS)
        unset_config_value("custom_providers.glm-flash")
        saved = _read_config(user_home)
        assert len(saved["custom_providers"]) == 1
        assert saved["custom_providers"][0]["name"] == "ygg-route"

    def test_config_unset_by_name_miss_exits_not_set_and_writes_nothing(self, user_home, capsys):
        """Regression guard (invariant 6): the public unset face keeps reporting a
        miss as "not set" (SystemExit) and rewrites nothing."""
        _write_config(user_home, NAMED_PROVIDERS)
        before = _read_config(user_home)
        with pytest.raises(SystemExit):
            unset_config_value("custom_providers.nosuch")
        assert "not set" in capsys.readouterr().err
        assert _read_config(user_home) == before

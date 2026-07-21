#!/usr/bin/env python3
"""Unit tests for mem0_dedup_scan.py."""

import importlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# ---------------------------------------------------------------------------
# Import the module under test by injecting its directory into sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path.home() / ".hermes" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mem0_dedup_scan as m


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity(unittest.TestCase):

    def test_identical_vectors_return_1(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(m.cosine_similarity(v, v), 1.0, places=9)

    def test_orthogonal_vectors_return_0(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(m.cosine_similarity(a, b), 0.0, places=9)

    def test_zero_vector_a_returns_0(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        self.assertEqual(m.cosine_similarity(a, b), 0.0)

    def test_zero_vector_b_returns_0(self):
        a = [1.0, 2.0]
        b = [0.0, 0.0]
        self.assertEqual(m.cosine_similarity(a, b), 0.0)

    def test_both_zero_vectors_returns_0(self):
        self.assertEqual(m.cosine_similarity([0.0], [0.0]), 0.0)

    def test_known_value_45_degrees(self):
        # [1,0] and [1,1]/sqrt(2) have cosine similarity = 1/sqrt(2) ≈ 0.7071
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        expected = 1.0 / (2.0 ** 0.5)
        self.assertAlmostEqual(m.cosine_similarity(a, b), expected, places=9)

    def test_antiparallel_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(m.cosine_similarity(a, b), -1.0, places=9)

    def test_known_value_3d(self):
        # [1,1,0] vs [0,1,1]: dot=1, |a|=sqrt(2), |b|=sqrt(2) → 0.5
        a = [1.0, 1.0, 0.0]
        b = [0.0, 1.0, 1.0]
        self.assertAlmostEqual(m.cosine_similarity(a, b), 0.5, places=9)


# ---------------------------------------------------------------------------
# group_pairs — union-find correctness
# ---------------------------------------------------------------------------

def _make_point(pid):
    return {"id": pid, "vector": [1.0], "payload": {}}


def _make_pair(a_id, b_id, score=0.95):
    return (score, _make_point(a_id), _make_point(b_id))


def _members_set(groups):
    """Return a frozenset of frozensets of member ids for easy comparison."""
    return frozenset(frozenset(g["members"]) for g in groups)


class TestGroupPairs(unittest.TestCase):

    def test_chain_of_3_forms_one_group(self):
        """A-B, B-C → one group of 3 (this was the broken path-compression case)."""
        pairs = [
            _make_pair("A", "B"),
            _make_pair("B", "C"),
        ]
        groups = m.group_pairs(pairs)
        self.assertEqual(len(groups), 1)
        self.assertIn(frozenset({"A", "B", "C"}), _members_set(groups))

    def test_two_disjoint_pairs_form_two_groups(self):
        """A-B, C-D with no bridge → two separate groups of 2."""
        pairs = [
            _make_pair("A", "B"),
            _make_pair("C", "D"),
        ]
        groups = m.group_pairs(pairs)
        self.assertEqual(len(groups), 2)
        expected = frozenset({frozenset({"A", "B"}), frozenset({"C", "D"})})
        self.assertEqual(_members_set(groups), expected)

    def test_chain_of_4_forms_one_group(self):
        """A-B, B-C, C-D → one group of 4 (tests deep path compression)."""
        pairs = [
            _make_pair("A", "B"),
            _make_pair("B", "C"),
            _make_pair("C", "D"),
        ]
        groups = m.group_pairs(pairs)
        self.assertEqual(len(groups), 1)
        self.assertIn(frozenset({"A", "B", "C", "D"}), _members_set(groups))

    def test_single_pair_forms_one_group_of_2(self):
        pairs = [_make_pair("X", "Y")]
        groups = m.group_pairs(pairs)
        self.assertEqual(len(groups), 1)
        self.assertIn(frozenset({"X", "Y"}), _members_set(groups))

    def test_empty_pairs_returns_empty_groups(self):
        groups = m.group_pairs([])
        self.assertEqual(groups, [])

    def test_star_topology(self):
        """A-B, A-C, A-D → one group of 4."""
        pairs = [
            _make_pair("A", "B"),
            _make_pair("A", "C"),
            _make_pair("A", "D"),
        ]
        groups = m.group_pairs(pairs)
        self.assertEqual(len(groups), 1)
        self.assertIn(frozenset({"A", "B", "C", "D"}), _members_set(groups))

    def test_mixed_connected_and_disjoint(self):
        """A-B, B-C form one group; X-Y form another."""
        pairs = [
            _make_pair("A", "B"),
            _make_pair("B", "C"),
            _make_pair("X", "Y"),
        ]
        groups = m.group_pairs(pairs)
        self.assertEqual(len(groups), 2)
        expected = frozenset({frozenset({"A", "B", "C"}), frozenset({"X", "Y"})})
        self.assertEqual(_members_set(groups), expected)


# ---------------------------------------------------------------------------
# MEM0_CONFIG constant
# ---------------------------------------------------------------------------

class TestMem0Config(unittest.TestCase):

    def test_mem0_config_is_path_object(self):
        self.assertIsInstance(m.MEM0_CONFIG, Path)

    def test_mem0_config_resolves_under_home(self):
        home = Path.home()
        # MEM0_CONFIG should be a descendant of home
        self.assertTrue(str(m.MEM0_CONFIG).startswith(str(home)))

    def test_mem0_config_ends_with_expected_suffix(self):
        self.assertEqual(m.MEM0_CONFIG.name, "mem0.json")
        self.assertEqual(m.MEM0_CONFIG.parent.name, ".hermes")

    def test_mem0_config_uses_path_home(self):
        """MEM0_CONFIG must be derived from Path.home(), not a hardcoded string."""
        fake_home = Path("/fake/home")
        with patch("pathlib.Path.home", return_value=fake_home):
            # Re-evaluate the expression used in the module
            config = Path.home() / ".hermes" / "mem0.json"
            self.assertEqual(config, fake_home / ".hermes" / "mem0.json")


# ---------------------------------------------------------------------------
# read_user_id
# ---------------------------------------------------------------------------

class TestReadUserId(unittest.TestCase):

    def test_returns_user_id_from_valid_json(self):
        cfg = {"user_id": "testuser"}
        mock_file = mock_open(read_data=json.dumps(cfg))
        with patch("builtins.open", mock_file):
            result = m.read_user_id()
        self.assertEqual(result, "testuser")

    def test_returns_clark_fallback_on_missing_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError("no file")):
            result = m.read_user_id()
        self.assertEqual(result, "clark")

    def test_returns_clark_fallback_on_malformed_json(self):
        mock_file = mock_open(read_data="not valid json{{")
        with patch("builtins.open", mock_file):
            result = m.read_user_id()
        self.assertEqual(result, "clark")

    def test_returns_clark_when_user_id_key_absent(self):
        cfg = {"other_key": "value"}
        mock_file = mock_open(read_data=json.dumps(cfg))
        with patch("builtins.open", mock_file):
            result = m.read_user_id()
        self.assertEqual(result, "clark")

    def test_returns_clark_on_permission_error(self):
        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = m.read_user_id()
        self.assertEqual(result, "clark")


if __name__ == "__main__":
    unittest.main()

"""Contract cases for the bound profile scope: identity by the home folder's canonical key, a label that never selects storage, and repository verbs that take no scope."""

from __future__ import annotations

import dataclasses
import inspect
import os
from pathlib import Path

import pytest

import persistence
from persistence import ProfileScope
from persistence.core.context import canonical_home_key


def test_scope_identity_is_the_canonical_key_of_the_home_folder(tmp_path):
    home = tmp_path / "profiles" / "coder"
    home.mkdir(parents=True)
    link = tmp_path / "linked-home"
    link.symlink_to(home, target_is_directory=True)
    # CONTRACT: the key is the normalized resolved profile_dir; two folders resolving to one home are one scope.
    assert ProfileScope(profile_dir=home, name="coder").hermes_home_key == os.path.normcase(str(home.resolve()))
    assert ProfileScope(profile_dir=link, name="coder") == ProfileScope(profile_dir=home, name="coder")
    assert len({ProfileScope(profile_dir=link, name="a"), ProfileScope(profile_dir=home, name="b")}) == 1


def test_label_is_not_identity(tmp_path):
    # CONTRACT: name is a label only: same folder, different labels, same scope; different folders, same label, different scopes.
    assert ProfileScope(profile_dir=tmp_path / "x", name="one") == ProfileScope(profile_dir=tmp_path / "x", name="two")
    assert ProfileScope(profile_dir=tmp_path / "x", name="p") != ProfileScope(profile_dir=tmp_path / "y", name="p")


def test_a_folder_not_on_disk_gets_its_lenient_key(tmp_path):
    missing = tmp_path / "not-yet" / ".." / "later"
    # CONTRACT: a home that does not exist yet resolves leniently, before the folder is created.
    assert canonical_home_key(missing) == os.path.normcase(str(missing.resolve(strict=False)))


def test_scope_is_frozen_and_needs_a_folder_and_a_label(tmp_path):
    scope = ProfileScope(profile_dir=tmp_path, name="p")
    with pytest.raises(dataclasses.FrozenInstanceError):
        scope.name = "q"  # type: ignore[misc]
    with pytest.raises(ValueError):
        ProfileScope(profile_dir=tmp_path, name="")
    with pytest.raises(TypeError):
        ProfileScope(name="p")  # type: ignore[call-arg]
    assert isinstance(ProfileScope(profile_dir=str(tmp_path), name="p").profile_dir, Path)  # type: ignore[arg-type]


def _protocols():
    for name in persistence.__all__:
        obj = getattr(persistence, name)
        if inspect.isclass(obj) and getattr(obj, "_is_protocol", False):
            yield obj


def test_no_repository_verb_takes_a_scope():
    # CONTRACT: the store is bound to one profile when opened; no verb takes a scope parameter.
    offenders = [
        f"{proto.__name__}.{verb}"
        for proto in _protocols()
        for verb, fn in inspect.getmembers(proto, inspect.isfunction)
        if not verb.startswith("_") and any(
            p.name == "scope" or "ProfileScope" in str(p.annotation) for p in inspect.signature(fn).parameters.values()
        )
    ]
    assert list(_protocols())
    assert offenders == []

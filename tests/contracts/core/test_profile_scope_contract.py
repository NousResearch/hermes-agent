"""Contract cases for the bound profile scope: identity by the canonical home folder it resolves once when built, a label that never selects storage, and repository verbs that take no scope."""

from __future__ import annotations

import dataclasses
import inspect
import os
import typing
from pathlib import Path

import pytest

from persistence import ProfileScope
from persistence.authority.runtime import SessionRuntimeRepository
from persistence.conversation.messages import MessageRepository
from persistence.conversation.sessions import SessionRepository
from persistence.core.backend import PersistenceStore
from persistence.core.context import canonical_home_key
from persistence.gateway.routing import RoutingRepository
from persistence.housekeeping.maintenance import SessionMaintenance
from persistence.workers.executions import WorkerExecutionRepository


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


def test_a_scope_keeps_the_folder_it_resolved_when_its_symlink_is_repointed(tmp_path):
    home_a = tmp_path / "profiles" / "a"
    home_b = tmp_path / "profiles" / "b"
    home_a.mkdir(parents=True)
    home_b.mkdir(parents=True)
    link = tmp_path / "active-home"
    link.symlink_to(home_a, target_is_directory=True)
    scope = ProfileScope(profile_dir=link, name="active")
    link.unlink()
    link.symlink_to(home_b, target_is_directory=True)
    # CONTRACT: a scope names the folder it resolved when built; repointing a symlink on its path moves neither its directory nor its key.
    assert scope.profile_dir == home_a.resolve()
    assert scope.hermes_home_key == os.path.normcase(str(home_a.resolve()))
    # CONTRACT: a scope built through the link after the repoint names the new target.
    rebuilt = ProfileScope(profile_dir=link, name="active")
    assert rebuilt.profile_dir == home_b.resolve()
    assert rebuilt.hermes_home_key == os.path.normcase(str(home_b.resolve()))
    assert rebuilt != scope


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


def _repository_protocols():
    """The protocol of every repository attribute ``PersistenceStore`` declares."""
    return [hint for hint in typing.get_type_hints(PersistenceStore).values() if getattr(hint, "_is_protocol", False)]


def test_no_repository_verb_takes_a_scope():
    protocols = _repository_protocols()
    assert {MessageRepository, SessionRepository, SessionRuntimeRepository, RoutingRepository, WorkerExecutionRepository, SessionMaintenance} <= set(protocols)
    # CONTRACT: the store is bound to one profile when opened; no verb takes a scope parameter.
    offenders = [
        f"{proto.__name__}.{verb}"
        for proto in protocols
        for verb, fn in inspect.getmembers(proto, inspect.isfunction)
        if not verb.startswith("_") and any(
            p.name == "scope" or "ProfileScope" in str(p.annotation) for p in inspect.signature(fn).parameters.values()
        )
    ]
    assert offenders == []

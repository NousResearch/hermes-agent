"""Broken blob backends the conformance suite must reject (``test_mutants.py``).

Each passed an earlier version of the suite, and each is applied by the ``backend`` fixtures
when ``HERMES_BLOB_MUTANT`` names it. The local backend's holes: reads, writes and the owner
lock leaking through a tenant-blind or missing pin (``reads_skip_the_pin``,
``writes_skip_the_pin``, ``pin_ignores_tenant``), the containment checks dropped
(``traversal_unchecked``, ``symlink_followed``) and the level refusal dropped
(``level_refusal_removed``). The two-level reference backend's holes: participant records
stored without their subject (``participant_shared``), reads ignoring the participant level
(``participant_reads_ignored``), one shared namespace across agents
(``shared_namespace``), and the turn fence on conversation-file writes never checked
(``participant_writes_skip_the_fence``).
"""

LOCAL_SUITE = "tests/conformance/blobs/test_local.py"
REFERENCE_SUITE = "tests/conformance/blobs/test_reference.py"


def _local_reads_skip_the_pin(monkeypatch):
    """get, exists and list never check the pin: another agent's scope reads the store."""
    from hermes_blob_port_local import LocalBlobStore

    def get(self, key):
        _root, path = LocalBlobStore._locate(self, key)
        try:
            return path.read_bytes()
        except (IsADirectoryError, FileNotFoundError, NotADirectoryError):
            from hermes_blob_port import BlobNotFound
            raise BlobNotFound(key) from None

    def exists(self, key):
        _root, path = LocalBlobStore._locate(self, key)
        return path.is_file()

    def list(self, prefix=""):
        if prefix.startswith(("/", "\\")):
            from hermes_blob_port import BlobKeyRefused
            raise BlobKeyRefused(f"the blob prefix {prefix!r} is absolute")
        keys = LocalBlobStore._list_all(self)
        return sorted(k for k in keys if k.startswith(prefix))
    monkeypatch.setattr(LocalBlobStore, "get", get)
    monkeypatch.setattr(LocalBlobStore, "exists", exists)
    monkeypatch.setattr(LocalBlobStore, "list", list)


def _local_writes_skip_the_pin(monkeypatch):
    """put and delete never check the pin: another agent's scope writes the store."""
    from hermes_blob_port_local import LocalBlobStore

    def put(self, key, data, *, level="agent"):
        LocalBlobStore._level(self, level)
        _root, path = LocalBlobStore._locate(self, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def delete(self, key, *, level="agent"):
        LocalBlobStore._level(self, level)
        _root, path = LocalBlobStore._locate(self, key)
        try:
            path.unlink()
            return True
        except (FileNotFoundError, IsADirectoryError):
            return False
    monkeypatch.setattr(LocalBlobStore, "put", put)
    monkeypatch.setattr(LocalBlobStore, "delete", delete)


def _local_pin_ignores_tenant(monkeypatch):
    """The pin's agent half alone: another tenant naming the store's home passes."""
    from hermes_constants import get_hermes_home, hermes_home_key
    from hermes_blob_port import BlobPinRefused
    from hermes_blob_port_local import LocalBlobStore
    from hermes_scope import current_scope

    def _pin(self):
        scope = current_scope()
        agent = scope.agent if scope is not None else get_hermes_home()
        if hermes_home_key(agent) != self._home_key:
            raise BlobPinRefused("the bound agent is not this blob store's profile")
    monkeypatch.setattr(LocalBlobStore, "_pin", _pin)


def _local_traversal_unchecked(monkeypatch):
    """The key is joined onto the family root raw: no containment check, so a key with ``..``
    components (an absolute path replaces the root outright) names files outside the
    namespace."""
    from hermes_blob_port_local import _MEDIA_FAMILIES, LocalBlobStore

    def _locate(self, key):
        parts = key.split("/")
        if parts[0] == "media":
            family, rest = parts[1], parts[2:]
            if family not in _MEDIA_FAMILIES:
                raise _key_refused(f"a media key names its cache, not {family!r}")
        elif parts[0] in ("pets", "skills", "attachments"):
            family, rest = parts[0], parts[1:]
        else:
            raise _key_refused(f"a blob key starts with a family it serves, not {parts[0]!r}")
        root = LocalBlobStore._family_root(self, family).resolve()
        return root, root.joinpath(*rest)
    monkeypatch.setattr(LocalBlobStore, "_locate", _locate)


def _key_refused(message):
    from hermes_blob_port import BlobKeyRefused
    return BlobKeyRefused(message)


def _local_symlink_followed(monkeypatch):
    """The key is containment-checked lexically but never resolved: a symlink planted inside
    the tree pointing outside it is followed."""
    from hermes_blob_port import parse_key
    from hermes_blob_port_local import _MEDIA_FAMILIES, LocalBlobStore

    def _locate(self, key):
        components = parse_key(key)
        if components[0] == "media":
            family, rest = components[1], components[2:]
            if family not in _MEDIA_FAMILIES:
                raise _key_refused(f"a media key names its cache, not {family!r}")
        elif components[0] in ("pets", "skills", "attachments"):
            family, rest = components[0], components[1:]
        else:
            raise _key_refused(f"a blob key starts with a family it serves, not {components[0]!r}")
        root = LocalBlobStore._family_root(self, family).resolve()
        return root, root.joinpath(*rest)
    monkeypatch.setattr(LocalBlobStore, "_locate", _locate)


def _local_level_refusal_removed(monkeypatch):
    """Any level is accepted and stored: a participant-level write lands at the agent level
    instead of being refused."""
    from hermes_blob_port_local import LocalBlobStore
    monkeypatch.setattr(LocalBlobStore, "_level", lambda self, level: None)


def _reference_participant_shared(monkeypatch):
    """Participant records are stored in the agent's own directory, without their subject:
    every member of the agent reads every other member's conversation files."""
    from tests.conformance.blobs.reference import ReferenceBlobStore
    monkeypatch.setattr(ReferenceBlobStore, "_level_dir",
                        lambda self, level, subject: self.root / self.tenant / "agent" / self.agent)


def _reference_participant_reads_ignored(monkeypatch):
    """Reads never resolve the participant level: a member's own conversation file is
    invisible to them."""
    from tests.conformance.blobs.reference import ReferenceBlobStore
    monkeypatch.setattr(ReferenceBlobStore, "_read_dirs",
                        lambda self: [ReferenceBlobStore._level_dir(self, "agent", "")])


def _reference_shared_namespace(monkeypatch):
    """Every agent writes one shared namespace: one agent's put overwrites another's blob at
    the same key, and a foreign blob is visible."""
    from tests.conformance.blobs.reference import ReferenceBlobStore
    monkeypatch.setattr(ReferenceBlobStore, "_level_dir", lambda self, level, subject: self.root / "shared")


def _reference_participant_writes_skip_the_fence(monkeypatch):
    """The turn fence on conversation-file writes is never checked: a write with no turn bound,
    or one under a fence its session's lease no longer holds, lands anyway."""
    from tests.conformance.blobs.reference import ReferenceBlobStore
    monkeypatch.setattr(ReferenceBlobStore, "_write_fence", lambda self, level: None)


#: name → (how to apply it, which suite file kills it, its own-agent control arguments).
MUTANTS = {
    "reads_skip_the_pin": (_local_reads_skip_the_pin, LOCAL_SUITE,
                           [LOCAL_SUITE, "-k", "succeeds_for_its_own_agent"]),
    "writes_skip_the_pin": (_local_writes_skip_the_pin, LOCAL_SUITE,
                            [LOCAL_SUITE, "-k", "succeeds_for_its_own_agent"]),
    "pin_ignores_tenant": (_local_pin_ignores_tenant, LOCAL_SUITE,
                           [LOCAL_SUITE, "-k", "succeeds_for_its_own_agent"]),
    "traversal_unchecked": (_local_traversal_unchecked, LOCAL_SUITE,
                            [LOCAL_SUITE, "-k", "succeeds_for_its_own_agent"]),
    "symlink_followed": (_local_symlink_followed, LOCAL_SUITE,
                         [LOCAL_SUITE, "-k", "succeeds_for_its_own_agent"]),
    "level_refusal_removed": (_local_level_refusal_removed, LOCAL_SUITE,
                              [LOCAL_SUITE, "-k", "succeeds_for_its_own_agent"]),
    "participant_shared": (_reference_participant_shared, REFERENCE_SUITE,
                           [REFERENCE_SUITE, "-k",
                            "succeeds_for_its_own_agent or reads_the_agents_own_files"]),
    "participant_reads_ignored": (_reference_participant_reads_ignored, REFERENCE_SUITE,
                                  [REFERENCE_SUITE, "-k",
                                   "succeeds_for_its_own_agent or reads_the_agents_own_files"]),
    "shared_namespace": (_reference_shared_namespace, REFERENCE_SUITE,
                         [REFERENCE_SUITE, "-k",
                          "succeeds_for_its_own_agent or reads_the_agents_own_files"]),
    "participant_writes_skip_the_fence": (_reference_participant_writes_skip_the_fence, REFERENCE_SUITE,
                                          [REFERENCE_SUITE, "-k",
                                           "succeeds_for_its_own_agent or reads_the_agents_own_files"]),
}


def apply_mutant(monkeypatch):
    import os
    name = os.environ.get("HERMES_BLOB_MUTANT")
    if name:
        apply, _suite, _control = MUTANTS[name]
        apply(monkeypatch)

"""Public raw user-config mutations use the selected profile backend and guarded writer."""

import pytest

from hermes_cli import config
from hermes_cli import config_backend as cb


def test_update_preserves_raw_values_comments_and_profile_scope(tmp_path, monkeypatch):
    home = tmp_path / "work"
    home.mkdir()
    path = home / "config.yaml"
    path.write_text('# keep\nmodel:\n  default: "${MODEL}"  # literal\n')
    monkeypatch.setenv("HERMES_HOME", str(home))
    config.update_user_config(lambda doc: doc.update(delegation={"backend": "model_map"}))
    assert cb.read_config_doc(path) == {
        "model": {"default": "${MODEL}"}, "delegation": {"backend": "model_map"}}
    assert "# keep" in path.read_text() and "# literal" in path.read_text()
    assert not (tmp_path / "config.yaml").exists()


@pytest.mark.parametrize("failure", ["callback", "omission", "managed", "malformed", "unreadable"])
def test_refusals_leave_original_document(tmp_path, monkeypatch, failure):
    home = tmp_path / "home"
    home.mkdir()
    path = home / "config.yaml"
    path.write_text("keep: 1\n" if failure != "malformed" else "keep: [broken\n")
    before = path.read_bytes()
    monkeypatch.setenv("HERMES_HOME", str(home))
    called = []

    def mutate(doc):
        called.append(True)
        doc["new"] = 2
        if failure == "callback":
            raise ValueError("callback refused")
        if failure == "omission":
            del doc["keep"]

    if failure == "managed":
        monkeypatch.setattr(config, "is_managed", lambda: True)
    if failure == "unreadable":
        def deny(self, home):
            raise PermissionError("read denied")
        monkeypatch.setattr(cb.FileBackend, "read_user_doc", deny)
    with pytest.raises((RuntimeError, ValueError)):
        config.update_user_config(mutate)
    assert bool(called) == (failure in ("callback", "omission"))
    assert path.read_bytes() == before


def test_backend_receives_bound_home_and_one_document_without_local_file(tmp_path, monkeypatch):
    from copy import deepcopy
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    class MemoryBackend(cb.FileBackend):
        def __init__(self):
            self.docs = {}
            self.writes = []

        def version(self, home):
            return (len(self.writes), 0, 0, 0)

        def read_user_doc(self, home):
            return deepcopy(self.docs.get(home, {"keep": "${LITERAL}"}))

        def probe_readable(self, home):
            pass

        def write_changes(self, home, changes):
            self.writes.append(home)
            self.docs[home] = deepcopy(changes.document)

    backend = MemoryBackend()
    monkeypatch.setattr(cb, "get_config_backend", lambda: backend)
    launch, other = tmp_path / "launch", tmp_path / "other"
    monkeypatch.setenv("HERMES_HOME", str(launch))
    token = set_hermes_home_override(other)
    try:
        config.update_user_config(lambda doc: doc.update(selected="other"))
    finally:
        reset_hermes_home_override(token)
    config.update_user_config(lambda doc: doc.update(selected="launch"))
    assert backend.writes == [other, launch]
    assert backend.docs[other] == {"keep": "${LITERAL}", "selected": "other"}
    assert backend.docs[launch] == {"keep": "${LITERAL}", "selected": "launch"}
    assert not list(tmp_path.rglob("config.yaml"))


def test_concurrent_mutations_reconcile_fresh_document(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    home = tmp_path / "home"
    home.mkdir()
    path = home / "config.yaml"
    path.write_text("count: 0\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    def increment(_):
        config.update_user_config(lambda doc: doc.update(count=doc["count"] + 1))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(increment, range(32)))
    assert cb.read_config_doc(path) == {"count": 32}


def test_dashboard_backend_write_serializes_with_public_preset_update(tmp_path, monkeypatch):
    """A preset saved after the dashboard read must not be clobbered by its save.

    Events gate the real dashboard RMW after its read. A nonblocking lock probe
    distinguishes serialization from the old race without sleeps or scheduling
    assumptions; on the broken implementation the preset commits first.
    """
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from hermes_cli.web_routers.delegation import MODEL_MAP, _write_backend_block

    home = tmp_path / "home"
    home.mkdir()
    path = home / "config.yaml"
    path.write_text("delegation:\n  backends:\n    model_map:\n      options: []\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    dashboard_read = Event()
    release_dashboard = Event()
    probed = Event()
    preset_committed = Event()
    acquired = []

    def build(current):
        dashboard_read.set()
        assert release_dashboard.wait(10), "dashboard was not released"
        return {"options": [{"name": "dashboard", "model": "test/model"}]}

    def update_preset():
        can_acquire = config._CONFIG_LOCK.acquire(blocking=False)
        acquired.append(can_acquire)
        if can_acquire:
            config._CONFIG_LOCK.release()
        probed.set()

        def mutate(doc):
            block = doc["delegation"]["backends"][MODEL_MAP]
            block["presets"] = [{"name": "saved", "options": []}]
            block["active_preset"] = "saved"

        config.update_user_config(mutate)
        preset_committed.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        dashboard = pool.submit(_write_backend_block, None, MODEL_MAP, build)
        try:
            assert dashboard_read.wait(10), "dashboard never read config"
            preset = pool.submit(update_preset)
            assert probed.wait(10), "public writer never probed the lock"
            if acquired[0]:
                assert preset_committed.wait(10), "preset never committed in race window"
        finally:
            release_dashboard.set()
        dashboard.result(timeout=10)
        preset.result(timeout=10)

    block = cb.read_config_doc(path)["delegation"]["backends"][MODEL_MAP]
    assert block["options"] == [{"name": "dashboard", "model": "test/model"}]
    assert block.get("presets") == [{"name": "saved", "options": []}]
    assert block.get("active_preset") == "saved"
    assert acquired == [False], "dashboard must hold core's config lock across its full RMW"

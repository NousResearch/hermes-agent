"""An external DFlash draft on a recurrent model: replaces MTP when it fits, on one slot.

Speculative decoding keeps one copy of a recurrent model's state per slot per draft position, so a
depth-7 draft on the router's four slots holds 32 copies (4.7 GiB on Qwen3.8 27B, enough for
llama.cpp's fit to push weights to host memory on Nemotron 3.5 Bolt). One slot holds 8. The draft
is optional: a missing file or an older engine leaves the model on its own MTP heads.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import bootstrap, catalog, presets
from hermes_cli.local_runtime.catalog import AssetFile, CatalogEntry, QuantVariant
from hermes_cli.local_runtime.estimator import HardwareBudget
from tests.hermes_cli import test_local_download_jobs
from tests.hermes_cli.test_local_download_jobs import poll

client = test_local_download_jobs.client  # the download-jobs app, for this file's route tests

GIB = 1 << 30
MIB = 1 << 20
NATIVE = 262144
BUILD = QuantVariant(quant="Q4_K_M", files=(AssetFile("Hybrid-Q4_K_M.gguf", 16 * GIB),))
ENTRY = CatalogEntry(
    id="hybrid", display_name="Hybrid", description="", repo="org/Hybrid-GGUF", variants=(BUILD,),
    n_ctx_train=NATIVE, full_layers=16, recurrent_layers=48, per_layer_f16=4096, mtp=True,
    n_vocab=248320, mtp_draft_depth=2,
    draft=AssetFile("Hybrid-DFlash.gguf", GIB, repo="lab/Hybrid-DFlash-GGUF", revision="0123abc"),
    draft_spec_type="draft-dflash", draft_depth=7, draft_min_engine="b11370")
ROOMY = HardwareBudget(usable_vram_bytes=int(128 * GIB * 0.8), total_device_bytes=128 * GIB,
                       ram_available_bytes=0, uma=True)


@pytest.fixture
def model(tmp_path, monkeypatch):
    """A staged hybrid model; returns (gguf path, draft path, setter for the active entry)."""
    active = {"entry": ENTRY, "state": 150 * MIB}
    monkeypatch.setattr(catalog, "entry_for_model",
                        lambda model_id: active["entry"] if model_id == BUILD.model_id else None)
    monkeypatch.setattr(presets, "read_gguf_header", lambda p: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda h: replace(
        active["entry"].profile(BUILD), kv_scale=1.0, recurrent_state_bytes=active["state"]))
    draft = bootstrap.assets_dir() / ENTRY.draft.local_name
    draft.parent.mkdir(parents=True, exist_ok=True)
    draft.touch()

    def use(entry=None, *, state=None):
        active["entry"] = entry or active["entry"]
        active["state"] = active["state"] if state is None else state
    return tmp_path / f"{BUILD.model_id}.gguf", draft, use


def _discrete(gib: float) -> HardwareBudget:
    return HardwareBudget(usable_vram_bytes=int(gib * GIB), total_device_bytes=int(gib * GIB),
                          ram_available_bytes=64 * GIB)


def _drafts(preset) -> bool:
    return preset.keys.get("spec-type") == "draft-dflash"


def test_the_draft_replaces_mtp_on_one_slot_when_it_fits(model):
    gguf, draft, _ = model

    preset = presets.preset_for_model(gguf, ROOMY, set(), engine_tag="b11671")

    assert preset.window == NATIVE and not preset.spilled
    assert preset.keys["spec-type"] == "draft-dflash"
    assert preset.keys["spec-draft-n-max"] == "7"
    assert preset.keys["model-draft"] == str(draft)
    assert preset.keys["parallel"] == "1"


def test_a_missing_draft_or_an_older_engine_keeps_mtp(model):
    gguf, draft, _ = model

    for tag in (None, "b11000"):
        preset = presets.preset_for_model(gguf, ROOMY, set(), engine_tag=tag)
        assert preset.keys["spec-type"] == "draft-mtp" and "parallel" not in preset.keys
    draft.unlink()
    preset = presets.preset_for_model(gguf, ROOMY, set(), engine_tag="b11671")
    assert preset.keys["spec-type"] == "draft-mtp" and "model-draft" not in preset.keys


def test_the_draft_may_shrink_the_window_to_half_native_and_no_further(model):
    """Each budget either keeps the MTP preset exactly, or drafts resident at no less than the
    smaller of the MTP window and half the native window. Growth past it drops the draft."""
    gguf, _, _ = model
    shrank = declined_with_room = 0
    for tenths in range(160, 400, 2):
        budget = _discrete(tenths / 10)
        mtp = presets.preset_for_model(gguf, budget, set())
        preset = presets.preset_for_model(gguf, budget, set(), engine_tag="b11671")
        if mtp.refusal:
            continue
        if _drafts(preset):
            assert not preset.spilled
            assert preset.window >= min(mtp.window, NATIVE // 2)
            assert preset.keys["parallel"] == "1"
            shrank += preset.window < mtp.window
        else:
            assert preset == mtp
            declined_with_room += not mtp.spilled
    assert shrank and declined_with_room


def test_the_draft_runs_only_where_it_fits_beside_other_programs(model):
    """A card too busy for even the floor window holds the model at the floor rather than move
    weights off it on a live reading. The draft never rides on that: it runs only where the model
    and the draft fit the free memory on their own."""
    gguf, _, _ = model
    capacity = _discrete(32)
    busy = drafted = 0
    for quarters in range(56, 120):
        live = replace(capacity, usable_vram_bytes=int(quarters / 4 * GIB))
        preset = presets.preset_for_model(gguf, capacity, set(), live=live, engine_tag="b11671")
        alone = presets.preset_for_model(gguf, live, set(), engine_tag="b11671")
        busy += alone.spilled
        if _drafts(preset):
            drafted += 1
            assert _drafts(alone) and not alone.spilled, quarters / 4
    assert busy and drafted


def test_on_windows_unified_memory_the_draft_keeps_the_smallest_prompt_cache(model):
    """On one slot a title or other side request takes the chat's slot, and llama-server's prompt
    cache is what brings the chat back. Where free commit sizes that cache, the draft runs only
    with the smallest useful cache beside it."""
    from hermes_cli.local_runtime import hardware

    gguf, _, _ = model
    capacity = HardwareBudget(usable_vram_bytes=48 * GIB, total_device_bytes=48 * GIB,
                              ram_available_bytes=64 * GIB, uma=True)
    drafted = 0
    for quarters in range(64, 160):
        room = int(quarters / 4 * GIB)
        live = replace(capacity, usable_vram_bytes=min(capacity.usable_vram_bytes, int(room * 0.8)),
                       commit_room_bytes=room)
        preset = presets.preset_for_model(gguf, capacity, set(), live=live, engine_tag="b11671")
        if _drafts(preset):
            drafted += 1
            assert int(preset.keys["cache-ram"]) << 20 >= hardware._PROMPT_CACHE_MIN, quarters / 4
    assert drafted


def test_a_grown_window_wins_over_the_draft(model):
    """Growth restored the native window on a roomy card, and other programs now hold part of it:
    whatever window still fits goes to the model as it was, not to the draft."""
    gguf, _, _ = model
    live = next(b for b in (_discrete(t / 10) for t in range(160, 400, 2))
                if _drafts(p := presets.preset_for_model(gguf, b, set(), engine_tag="b11671"))
                and p.window < presets.preset_for_model(gguf, b, set()).window)
    mtp = presets.preset_for_model(gguf, ROOMY, set(), requested_window=NATIVE, live=live)

    grown = presets.preset_for_model(gguf, ROOMY, set(), requested_window=NATIVE, live=live,
                                     engine_tag="b11671")

    assert grown == mtp and grown.keys["spec-type"] == "draft-mtp"


def test_each_draft_position_prices_a_copy_of_the_recurrent_state(model):
    """The same card that fits the draft beside 150 MiB of state per copy declines it at 1 GiB
    per copy: eight copies are 7 GiB more."""
    gguf, _, use = model
    budget = _discrete(32)
    assert _drafts(presets.preset_for_model(gguf, budget, set(), engine_tag="b11671"))

    use(state=GIB)

    assert not _drafts(presets.preset_for_model(gguf, budget, set(), engine_tag="b11671"))


def test_growth_counts_the_outgoing_draft_as_freed_memory(model, monkeypatch):
    """The grown instance loads after the drafting one exits, so all of that instance's memory,
    draft included, counts as free when growth fits the next window."""
    from hermes_cli.local_runtime import estimator, gguf as gguf_module, growth, hardware

    path, _, _ = model
    budget = _discrete(28)
    running = presets.preset_for_model(path, budget, set(), engine_tag="b11671")
    assert _drafts(running) and running.window < NATIVE
    monkeypatch.setattr(bootstrap, "staged_models", lambda: [path])
    monkeypatch.setattr(bootstrap, "get_supervisor", lambda: SimpleNamespace(is_idle=lambda m: True))
    monkeypatch.setattr(growth, "is_managed_endpoint", lambda url: True)
    monkeypatch.setattr(gguf_module, "read_gguf_header", presets.read_gguf_header)
    monkeypatch.setattr(estimator, "profile_from_gguf", presets.profile_from_gguf)
    monkeypatch.setattr(hardware, "probe_budget", lambda **kw: budget)
    monkeypatch.setattr(presets, "read_preset_decisions", lambda *a: {BUILD.model_id: running})
    freed = []
    monkeypatch.setattr(bootstrap, "_launch_budget", lambda b, own_bytes=0: freed.append(own_bytes) or b)
    monkeypatch.setattr(bootstrap, "refresh_local_runtime", lambda: False)

    growth.maybe_grow_window(BUILD.model_id, base_url="http://127.0.0.1:1/v1",
                             session_tokens=running.window, current_window=running.window)

    drafting = presets.resident_footprint(path, budget, running.window, drafting=True)
    assert freed == [drafting]
    assert drafting > presets.resident_footprint(path, budget, running.window)


def test_a_spilled_model_never_takes_the_draft(model):
    gguf, _, _ = model

    preset = presets.preset_for_model(gguf, _discrete(12), set(), engine_tag="b11671")

    assert preset.spilled and not _drafts(preset)


def test_a_model_without_mtp_drafts_without_backend_sampling(model):
    """Bolt's shape: no MTP heads, so nothing to replace and no backend sampling to keep."""
    gguf, _, use = model
    use(replace(ENTRY, mtp=False, draft_min_engine="b11671"))

    preset = presets.preset_for_model(gguf, ROOMY, set(), engine_tag="b11671")

    assert _drafts(preset) and preset.keys["parallel"] == "1"
    assert "backend-sampling" not in preset.keys


def test_the_residency_price_is_the_launch_the_model_runs(model):
    """The preset runs the draft in place of MTP, so the model holds the larger of the two launches,
    never both added together. An engine that can't load the draft prices MTP alone."""
    gguf, _, use = model
    larger = set()
    for state in (150 * MIB, 600 * MIB):  # MTP's own layers outweigh the draft, then the reverse
        use(state=state)
        mtp = presets.resident_footprint(gguf, ROOMY, NATIVE)
        drafting = presets.resident_footprint(gguf, ROOMY, NATIVE, drafting=True)
        larger.add("draft" if drafting > mtp else "mtp")

        assert presets._launch_footprint(gguf, ROOMY, engine_tag="b11671") == max(mtp, drafting)
        assert presets._launch_footprint(gguf, ROOMY, engine_tag="b11000") == mtp
        assert presets._launch_footprint(gguf, ROOMY) == mtp
    assert larger == {"draft", "mtp"}


def test_two_models_stay_resident_where_two_drafted_launches_fit(model):
    gguf, _, _ = model
    gguf.touch()
    held = max(presets.resident_footprint(gguf, ROOMY, NATIVE),
               presets.resident_footprint(gguf, ROOMY, NATIVE, drafting=True))
    card = replace(ROOMY, usable_vram_bytes=2 * held + GIB)

    assert presets.admitted_residency_count(gguf.parent, card, 4, engine_tag="b11671") == 2


def _staged(monkeypatch, *entries):
    from hermes_cli.web_routers import local_models as lm

    monkeypatch.setattr(lm.catalog, "CATALOG", entries)
    monkeypatch.setattr(lm.bootstrap, "staged_model_ids", lambda: [e.variants[0].model_id for e in entries])
    return lm


def test_installed_models_missing_their_draft_get_it_from_the_engine_update(monkeypatch):
    """A Spark that downloaded the 27B before its draft joined the catalog: the update the app
    already offers fetches the draft. Models not on disk, drafts already on disk, and drafts the
    engine can't load are left alone."""
    unstaged = replace(ENTRY, id="other", variants=(replace(BUILD, files=(AssetFile("Other.gguf", GIB),)),))
    newer = replace(ENTRY, id="newer", variants=(replace(BUILD, files=(AssetFile("Newer.gguf", GIB),)),),
                    draft=AssetFile("Newer-DFlash.gguf", GIB), draft_min_engine="b99999")
    lm = _staged(monkeypatch, ENTRY, newer)
    monkeypatch.setattr(lm.catalog, "CATALOG", (ENTRY, newer, unstaged))
    dest = lm.bootstrap.assets_dir() / ENTRY.draft.local_name

    plan = lm._missing_drafts("b11671")

    assert plan == [("https://huggingface.co/lab/Hybrid-DFlash-GGUF/resolve/0123abc/Hybrid-DFlash.gguf",
                     dest, GIB)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.touch()
    assert lm._missing_drafts("b11671") == []


def test_the_update_job_downloads_missing_drafts_then_restarts_the_server(client, monkeypatch):
    lm = _staged(monkeypatch, ENTRY)
    engine = SimpleNamespace(tag="b11671", backend="cpu")
    monkeypatch.setattr(lm, "_runtime_target", lambda backend=None: ("b11671", "cpu"))
    monkeypatch.setattr(lm.binaries, "installed_engine", lambda backend: engine)
    monkeypatch.setattr(lm, "_install_engine_job", lambda job, backend: engine)
    fetched, events = [], []
    monkeypatch.setattr(lm, "_download_job", lambda job, plan: fetched.extend(plan))
    monkeypatch.setattr(lm.bootstrap, "get_supervisor", lambda: object())
    monkeypatch.setattr(lm.bootstrap, "refresh_local_runtime", lambda: events.append("restart"))

    job_id = client.post("/api/local-models/runtime/install", json={"backend": "cpu"}).json()["job_id"]

    assert poll(client, job_id, until=("done", "error"))["status"] == "done"
    assert [dest.name for _, dest, _ in fetched] == [ENTRY.draft.local_name]
    assert events == ["restart"]


def test_a_draft_that_fails_to_download_leaves_the_model_downloaded(client, monkeypatch, tmp_path):
    """The draft only speeds generation up: a 404 on it costs the user the draft, never the model
    it was meant to speed up."""
    lm = _staged(monkeypatch, ENTRY)
    models, assets = tmp_path / "models", tmp_path / "assets"
    monkeypatch.setattr(lm.bootstrap, "models_dir", lambda: models)
    monkeypatch.setattr(lm.bootstrap, "assets_dir", lambda: assets)
    monkeypatch.setattr(lm, "_refresh_runtime", lambda *args: None)

    def download(job, plan):
        if any(dest.name == ENTRY.draft.local_name for _, dest, _ in plan):
            raise RuntimeError("download failed from https://huggingface.co/lab/Hybrid-DFlash-GGUF: HTTP Error 404")
        for _, dest, _ in plan:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.touch()

    monkeypatch.setattr(lm, "_download_job", download)

    job_id = client.post("/api/local-models/download", json={"model_id": BUILD.model_id}).json()["job_id"]

    assert poll(client, job_id, until=("done", "error"))["status"] == "done"
    assert (models / BUILD.files[0].local_name).is_file()
    assert not (assets / ENTRY.draft.local_name).exists()


def _update_job(monkeypatch, *, serving: str, on_download=None):
    """The engine update over a running server on ``serving``, with b11671 already installed."""
    lm = _staged(monkeypatch, ENTRY)
    engine = SimpleNamespace(tag="b11671", backend="cpu")
    monkeypatch.setattr(lm, "_runtime_target", lambda backend=None: ("b11671", "cpu"))
    monkeypatch.setattr(lm.binaries, "installed_engine", lambda backend: engine)
    monkeypatch.setattr(lm, "_install_engine_job", lambda job, backend: engine)
    monkeypatch.setattr(lm, "_download_job", on_download or (lambda job, plan: None))
    monkeypatch.setattr(lm.bootstrap, "get_supervisor", lambda: object())
    monkeypatch.setattr(lm.bootstrap, "serving_engine", lambda: SimpleNamespace(tag=serving, backend="cpu"))
    events = []
    monkeypatch.setattr(lm.bootstrap, "refresh_local_runtime", lambda: events.append("restart"))
    return lm, events


def test_a_pause_between_the_drafts_and_the_restart_still_restarts_on_resume(client, monkeypatch):
    """The pause lands as the last draft finishes. On resume no draft is missing, and the server
    still needs the restart that loads them."""
    def download(job, plan):
        for _, dest, _ in plan:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.touch()
        lm._RUNNING[job["job_id"]]["pause"].set()

    lm, events = _update_job(monkeypatch, serving="b11671", on_download=download)
    job_id = client.post("/api/local-models/runtime/install", json={"backend": "cpu"}).json()["job_id"]
    assert poll(client, job_id, until=("paused", "done", "error"))["status"] == "paused"
    assert events == []

    assert client.post("/api/local-models/download/resume", json={"job_id": job_id}).json()["resumed"]

    assert poll(client, job_id, until=("done", "error"))["status"] == "done"
    assert events == ["restart"]


def _failing_download(job, plan):
    raise RuntimeError("download failed from https://huggingface.co/lab/Hybrid-DFlash-GGUF: HTTP Error 404")


def test_a_failed_draft_download_still_switches_the_server_to_the_new_build(client, monkeypatch):
    """The draft only speeds generation up. A draft that fails to download leaves the model on MTP,
    and the server still moves to the build the update installed."""
    _, events = _update_job(monkeypatch, serving="b11370", on_download=_failing_download)

    job_id = client.post("/api/local-models/runtime/install", json={"backend": "cpu"}).json()["job_id"]

    assert poll(client, job_id, until=("done", "error"))["status"] == "done"
    assert events == ["restart"]


def test_a_failed_draft_download_alone_leaves_the_server_running(client, monkeypatch):
    """Nothing new landed for a restart to load, so a server already on the installed build keeps
    serving."""
    _, events = _update_job(monkeypatch, serving="b11671", on_download=_failing_download)

    job_id = client.post("/api/local-models/runtime/install", json={"backend": "cpu"}).json()["job_id"]

    assert poll(client, job_id, until=("done", "error"))["status"] == "done"
    assert events == []


def test_a_pause_during_the_draft_download_pauses_the_update(client, monkeypatch):
    from pm.downloader import DownloadPaused

    def paused(job, plan):
        raise DownloadPaused("download paused")

    _, events = _update_job(monkeypatch, serving="b11370", on_download=paused)

    job_id = client.post("/api/local-models/runtime/install", json={"backend": "cpu"}).json()["job_id"]

    assert poll(client, job_id, until=("paused", "done", "error"))["status"] == "paused"
    assert events == []


def test_the_update_restarts_a_server_still_on_an_older_build(client, monkeypatch):
    """An earlier update installed b11671 and stopped before its restart: the server still runs
    the old build, and the next update switches it over."""
    draft = bootstrap.assets_dir() / ENTRY.draft.local_name
    draft.parent.mkdir(parents=True, exist_ok=True)
    draft.touch()
    _, events = _update_job(monkeypatch, serving="b11370")

    job_id = client.post("/api/local-models/runtime/install", json={"backend": "cpu"}).json()["job_id"]

    assert poll(client, job_id, until=("done", "error"))["status"] == "done"
    assert events == ["restart"]


def test_an_installed_model_without_its_draft_still_reads_as_downloaded(monkeypatch):
    lm = _staged(monkeypatch, ENTRY)
    model = lm.bootstrap.models_dir() / BUILD.files[0].local_name
    model.parent.mkdir(parents=True, exist_ok=True)
    model.touch()

    row = lm._catalog_row(ENTRY, ROOMY, None, None, {BUILD.model_id})

    assert row["downloaded"] and row["size_bytes"] == ENTRY.download_bytes(BUILD)

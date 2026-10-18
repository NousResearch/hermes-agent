"""Durable setup intent; the existing installer and download paths do the work."""
from __future__ import annotations

from fastapi import HTTPException

from hermes_cli.local_runtime import download_plans


def _engine_urls(backend):
    import pm
    from pm.artifact_mirror import pinned_source
    from pm.registry import walk
    from hermes_cli.local_runtime.binaries import BACKEND_PACKAGES

    lock = pm.Lockfile(pm.paths.lockfile_path())
    urls = []
    for package in walk([BACKEND_PACKAGES[backend]]):
        for artifact in lock.artifacts(package.name, pm.current_target()):
            source = pinned_source(artifact["url"], pm.paths.partials_root(), artifact["sha256"])
            urls.extend((source.url, *source.fallbacks))
    return tuple(dict.fromkeys(urls))


def _save(job, plan):
    job["partial_urls"] = tuple(dict.fromkeys((*job.get("partial_urls", ()), *_engine_urls(job["setup"]["backend"]))))
    download_plans.save_plan(job, plan)
    job["plan_saved"] = True
    previous = job.pop("adopted_plan", None)
    if previous:
        download_plans.remove_plan(previous)


def _run_setup(job):
    from hermes_cli.web_routers import local_models as lm

    intent = job["setup"]
    body = lm.QuickstartBody(model_id=intent["model_id"])
    with lm._config_profile_scope(intent["profile"]):
        lm._check_job_pause(job)
        entry = lm.catalog.catalog_by_id().get(intent["model_id"])
        plan = job["download_plan"]
        missing_engine = lm.binaries.installed_engine(intent["backend"]) is None
        _save(job, plan)
        if missing_engine:
            lm._install_engine_job(job, intent["backend"])
        if intent["variant"] is None:
            entry, variant = lm._quickstart_target(body, lm.hardware.probe_budget(planning=True))
            job.update(target=entry.display_name, model_id=entry.id)
            intent["variant"] = variant.model_id
            plan = lm._download_plan(entry, variant)
            job["download_plan"] = plan
            download_plans.save_plan(job, plan)
        if any(not destination.is_file() for _, destination, _ in plan):
            if job["phase"] not in lm._MODEL_DOWNLOAD_PHASES:
                job.update(done_bytes=0, total_bytes=None, ranges={})
            lm._run_download_plan(job, plan, job["target"], entry)
        lm._step(job, "starting-server", "Starting the local server")
        lm._ensure_server(job, lm._set_runtime_enabled(True), intent["variant"],
                          fail_detail="The local server could not start — open Local Models for details",
                          skip_msg="quickstart rescan check skipped")
        lm._assign_default(job, intent["variant"])
        lm._finish(job, f"{job['target']} is ready — new chats use it")


def _run_runtime(job):
    from hermes_cli.web_routers import local_models as lm

    backend = job["setup"]["backend"]
    with lm._config_profile_scope(job["setup"]["profile"]):
        lm._check_job_pause(job)
        _save(job, [])
        previous = lm.binaries.installed_engine(backend)
        engine = lm._install_engine_job(job, backend)
        drafts = lm._missing_drafts(engine.tag)
        if drafts:
            download_plans.save_plan(job, drafts)
            before, job["restart_runtime"] = job.get("restart_runtime", False), True
            lm._step(job, "downloading", "Downloading draft models that speed up generation")
            if not lm._download_optional(job, drafts) and len(lm._missing_drafts(engine.tag)) == len(drafts):
                job["restart_runtime"] = before
        running = lm.bootstrap.get_supervisor()
        serving = lm.bootstrap.serving_engine()
        stale = previous != engine or (serving is not None and serving.tag != engine.tag)
        if running is not None and (stale or job.get("restart_runtime")):
            lm._step(job, "restarting", "Switching the running server to the pinned build"
                     if stale else "Restarting the local server")
            lm.bootstrap.refresh_local_runtime()
            job["restart_runtime"] = False
    lm._finish(job, f"llama.cpp {engine.tag} ready ({backend})")


def _spawn(job, *, restored=False):
    from hermes_cli.web_routers import local_models as lm

    def release():
        if job.get("plan_saved"):
            download_plans.remove_plan(job["plan_key"])
        if job["kind"] == "quickstart":
            lm._QUICKSTART_LOCK.release()

    run = {"quickstart": _run_setup, "runtime-install": _run_runtime}[job["kind"]]
    lm._spawn_job(job, "lr-local-setup", lambda: run(job), fail_msg="local setup failed: %s",
                  on_exit=release, resumable=True, start_paused=restored,
                  ownership=lambda: download_plans.plan_lock(job["plan_key"], wait=True,
                      cancelled=lm._RUNNING[job["job_id"]]["pause"].is_set))


def begin(body, profile):
    from hermes_constants import get_hermes_home, profile_name_for_home
    from hermes_cli.web_routers import local_models as lm

    with lm._config_profile_scope(profile):
        owner = profile_name_for_home(get_hermes_home()) or "default"
        entry, variant = lm._quickstart_target(body, lm.hardware.probe_budget(planning=True))
        _, backend = lm._runtime_target()
        need_runtime = lm.binaries.installed_engine(backend) is None
        plan = lm._download_plan(entry, variant)
    need_download = any(not destination.is_file() for _, destination, _ in plan)
    if not lm._QUICKSTART_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Setup is already running")
    try:
        job = lm._quickstart_job(entry, plan if need_download else [])
        if job.get("plan_saved"):
            job["adopted_plan"] = job["plan_key"]
        job["setup"] = {"model_id": entry.id, "profile": owner, "backend": backend,
                        "variant": None if need_runtime else variant.model_id}
        job["download_key"] = download_plans.plan_key(plan)
        job["download_plan"] = plan
        job["plan_key"] = download_plans.plan_key(plan, job["setup"])
    except Exception:
        lm._QUICKSTART_LOCK.release()
        raise
    _spawn(job)
    return {"job_id": job["job_id"], "model_id": entry.id, "display_name": entry.display_name,
            "needs_runtime": need_runtime, "needs_download": need_download,
            "download_bytes": sum(row[2] for row in plan) if need_download else 0}


def begin_runtime(body, profile):
    from hermes_constants import get_hermes_home, profile_name_for_home
    from hermes_cli.web_routers import local_models as lm

    with lm._config_profile_scope(profile):
        owner = profile_name_for_home(get_hermes_home()) or "default"
        tag, backend = lm._runtime_target(body.backend)
    job = lm._job("runtime-install", f"llama.cpp {tag} ({backend})")
    job["setup"] = {"model_id": None, "profile": owner, "backend": backend, "variant": None}
    job["plan_key"] = download_plans.plan_key([], job["setup"])
    _spawn(job)
    return {"job_id": job["job_id"], "backend": backend, "tag": tag}


def restore(key, record, plan):
    from hermes_cli.web_routers import local_models as lm

    with lm._JOBS_LOCK:
        if any(job.get("plan_key") == key for job in lm._JOBS.values()):
            return
    kind = "runtime-install" if record["setup"]["model_id"] is None else "quickstart"
    if kind == "quickstart" and not lm._QUICKSTART_LOCK.acquire(blocking=False):
        return
    job = lm._job(kind, record["target"], model_id=record["setup"]["model_id"])
    job.update(setup=record["setup"], plan_key=key, download_plan=plan,
               download_key=download_plans.plan_key(plan), plan_saved=True,
               partial_urls=tuple(record.get("partial_urls", ())), phase="interrupted",
               detail="Setup interrupted. Resume to finish setup")
    for name in ("done_bytes", "total_bytes", "ranges", "restart_runtime"):
        if name in record.get("state", {}):
            job[name] = record["state"][name]
    _spawn(job, restored=True)

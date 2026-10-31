"""oMLX control plane. Shares local-model jobs and profile-safe config writes."""
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hermes_cli import config as config_mod
from hermes_cli.local_runtime import omlx
from hermes_cli.web_routers._common import _CONFIG_MUTATION_LOCK, _config_profile_scope
from hermes_cli.web_routers import local_models as jobs

router = APIRouter(prefix="/api/local-models/omlx")


class EngineBody(BaseModel):
    engine: Literal["llamacpp", "omlx"]


class ServerBody(BaseModel):
    action: Literal["start", "stop"]


class ActivateBody(BaseModel):
    model_id: str


class DownloadBody(BaseModel):
    repo: str


def _supported():
    if not omlx.supported():
        raise HTTPException(400, "oMLX requires macOS 15 or newer on Apple Silicon (arm64)")


def _save_runtime(**values):
    with _CONFIG_MUTATION_LOCK:
        config = config_mod.load_config()
        config.setdefault("local_runtime", {}).update(values)
        config_mod.save_config(config)


@router.get("/status")
def status(profile: str | None = None):
    with _config_profile_scope(profile):
        return omlx.status(config_mod.load_config_readonly())


@router.post("/select")
def select(body: EngineBody, profile: str | None = None):
    # Browsing never changes the current cloud/custom provider or starts an engine.
    if body.engine == "omlx":
        _supported()
    with _config_profile_scope(profile):
        _save_runtime(engine=body.engine)
    return {"ok": True}


@router.post("/install")
def install():
    _supported()
    job = jobs._job("runtime-install", "oMLX")
    def run():
        jobs._step(job, "installing-runtime", "Installing the self-contained oMLX runtime")
        omlx.install(progress=jobs._download_progress_hook(job))
        jobs._finish(job, "oMLX ready")
    jobs._spawn_job(job, "omlx-install", run)
    return {"job_id": job["job_id"]}


@router.post("/server")
def server(body: ServerBody, profile: str | None = None):
    _supported()
    with _config_profile_scope(profile), jobs._http_error(409):
        {"start": omlx.start, "stop": omlx.stop}[body.action]()
        _save_runtime(omlx_enabled=body.action == "start")
    return {"ok": True}


@router.post("/activate")
def activate(body: ActivateBody, profile: str | None = None):
    _supported()
    with _config_profile_scope(profile):
        with jobs._http_error(409):
            omlx.validate_model(body.model_id)
            from hermes_cli.web_server_config import _prepare_main_assignment, _apply_model_assignment_sync
            key = omlx.read_state()["api_key"]
            prepared = _prepare_main_assignment(config_mod.load_config(), "custom", body.model_id, omlx.BASE_URL, key)
            with _CONFIG_MUTATION_LOCK:
                _apply_model_assignment_sync("main", "custom", body.model_id, "", omlx.BASE_URL, key,
                                             prepared=prepared)
                _save_runtime(engine="omlx", omlx_enabled=True)
    return {"ok": True}


@router.post("/download")
def download(body: DownloadBody):
    _supported()
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", body.repo):
        raise HTTPException(422, "Use a Hugging Face owner/model repository ID")
    if not omlx.cli().is_file():
        raise HTTPException(409, "Install the managed oMLX runtime first")
    mid = body.repo.replace("/", "--")
    job = jobs._job("model-download", f"oMLX · {body.repo}", model_id=mid)
    def run():
        jobs._step(job, "downloading", f"Downloading {body.repo}")
        omlx.download(body.repo)
        jobs._finish(job, f"oMLX · {body.repo} ready")
    jobs._spawn_job(job, "omlx-download", run)
    return {"job_id": job["job_id"]}

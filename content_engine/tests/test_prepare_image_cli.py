"""Provider-free CLI consumption of prepared ad-hoc image requests."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_engine_script(monkeypatch) -> ModuleType:
    """Load the executable module, never the similarly named package."""
    engine_path = Path(__file__).resolve().parents[1] / "content_engine.py"
    monkeypatch.syspath_prepend(str(engine_path.parent))
    spec = importlib.util.spec_from_file_location("_content_engine_script", engine_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_image_cli_returns_a_plan_without_opening_the_content_database(
    monkeypatch, capsys
) -> None:
    engine = _load_engine_script(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "content_engine.py",
            "prepare-image",
            "--prompt",
            "A high-density map of the agent runtime.",
            "--style",
            "Data Atlas",
            "--reference",
            "https://example.com/brief.pdf",
        ],
    )

    def _database_must_not_open() -> None:
        raise AssertionError("provider-free prepare-image must not open the content database")

    monkeypatch.setattr(engine, "init_db", _database_must_not_open)

    assert engine.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "codex"
    assert payload["style_id"] == "data-atlas"
    assert payload["references"] == ["https://example.com/brief.pdf"]


def test_prepare_image_cli_can_explicitly_stage_and_plan_without_a_provider(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    engine = _load_engine_script(monkeypatch)
    observed: dict[str, object] = {}

    def _stage(request, *, staging_root: Path, job_id: str):
        observed.update({"request": request, "staging_root": staging_root, "job_id": job_id})
        return SimpleNamespace(
            staged_references=(SimpleNamespace(sha256="b" * 64),),
            plan=SimpleNamespace(
                backend=request.backend,
                execution_enabled=False,
                reason="provider execution deliberately disabled",
            ),
        )

    monkeypatch.setattr(engine, "stage_and_plan_image_job", _stage, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "content_engine.py",
            "prepare-image",
            "--prompt",
            "A compact visual system map.",
            "--style",
            "Data Atlas",
            "--reference",
            "https://example.com/source",
            "--stage-root",
            str(tmp_path / "staging"),
            "--job-id",
            "job-cli",
        ],
    )
    monkeypatch.setattr(engine, "init_db", lambda: (_ for _ in ()).throw(AssertionError("no DB")))

    assert engine.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert observed["job_id"] == "job-cli"
    assert payload["staging"]["reference_count"] == 1
    assert payload["staging"]["execution_enabled"] is False


def test_generate_image_cli_executes_one_private_job_without_opening_the_content_database(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    engine = _load_engine_script(monkeypatch)
    observed: dict[str, object] = {}

    def _stage(request, *, staging_root: Path, job_id: str):
        observed.update({"request": request, "staging_root": staging_root, "job_id": job_id})
        return SimpleNamespace(plan=SimpleNamespace(backend=request.backend), staged_references=())

    def _execute(request, staged_job, *, staging_root: Path, job_id: str, aspect_ratio: str):
        observed.update(
            {
                "executed_request": request,
                "staged_job": staged_job,
                "execution_root": staging_root,
                "execution_job_id": job_id,
                "aspect_ratio": aspect_ratio,
            }
        )
        return SimpleNamespace(
            output_path=staging_root / job_id / "generated.png",
            completion_path=staging_root / job_id / "image-completion.json",
            sha256="c" * 64,
            provider="openai-codex",
            model="gpt-image-2-medium",
        )

    monkeypatch.setattr(engine, "stage_and_plan_image_job", _stage)
    monkeypatch.setattr(engine, "execute_staged_image_job", _execute, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "content_engine.py",
            "generate-image",
            "--prompt",
            "A compact visual system map.",
            "--style",
            "Data Atlas",
            "--stage-root",
            str(tmp_path / "staging"),
            "--job-id",
            "job-cli",
            "--aspect-ratio",
            "square",
        ],
    )
    monkeypatch.setattr(engine, "init_db", lambda: (_ for _ in ()).throw(AssertionError("no DB")))

    assert engine.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert observed["job_id"] == "job-cli"
    assert observed["aspect_ratio"] == "square"
    assert payload == {
        "backend": {"model": "gpt-image-2-medium", "provider": "openai-codex"},
        "completion_path": str(tmp_path / "staging" / "job-cli" / "image-completion.json"),
        "job_id": "job-cli",
        "output_path": str(tmp_path / "staging" / "job-cli" / "generated.png"),
        "sha256": "c" * 64,
    }


def test_generate_image_cli_rejects_local_before_source_staging(monkeypatch, capsys, tmp_path: Path) -> None:
    engine = _load_engine_script(monkeypatch)
    monkeypatch.setattr(
        engine,
        "stage_and_plan_image_job",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local must not stage or fetch")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "content_engine.py",
            "generate-image",
            "--prompt",
            "A local evaluation sheet.",
            "--style",
            "Data Atlas",
            "--backend",
            "local",
            "--stage-root",
            str(tmp_path / "staging"),
            "--job-id",
            "job-local",
        ],
    )
    monkeypatch.setattr(engine, "init_db", lambda: (_ for _ in ()).throw(AssertionError("no DB")))

    assert engine.main() == 2
    assert "Local execution is disabled" in capsys.readouterr().out

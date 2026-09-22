"""Behaviour coverage for the #pipeline Discord projection publisher."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "pipeline_projection.py"
SPEC = importlib.util.spec_from_file_location("pipeline_projection", SCRIPT)
assert SPEC and SPEC.loader
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_publish_creates_once_then_edits_same_canonical_message(tmp_path):
    requests = []

    def opener(request, timeout):
        requests.append(request)
        return Response({"id": "9001"})

    projection = pipeline.render_projection({
        "model": "card",
        "id": 444,
        "title": "Pipeline operacional",
        "state": "em execução",
        "objective": "Expor uma projeção de leitura.",
        "last_transition": "#506 em execução",
        "links": {"github": "https://github.com/CerteiroDevTeam/hermes-agent/issues/506"},
    })
    ledger = tmp_path / "projection.json"

    assert pipeline.publish("123", projection, ledger, "token", opener) == ("created", "9001")
    assert pipeline.publish("123", projection, ledger, "token", opener) == ("edited", "9001")

    assert [request.get_method() for request in requests] == ["POST", "PATCH"]
    assert requests[1].full_url.endswith("/channels/123/messages/9001")
    assert json.loads(ledger.read_text()) == {"epic:444": "9001"}


def test_waiting_pedro_is_rendered_without_inventing_missing_links():
    projection = pipeline.render_projection({
        "model": "waiting_pedro",
        "id": 444,
        "title": "Pipeline operacional",
        "state": "aguardando Pedro",
        "objective": "Aguardar decisão explícita.",
        "last_transition": "DevOps concluiu a projeção.",
    })

    assert projection.key == "epic:444"
    assert "## Aguardando Pedro" in projection.content
    assert "GitHub: evidência indisponível" in projection.content
    assert "Ação de Pedro: evidência indisponível" in projection.content


def test_rejects_unknown_model():
    with pytest.raises(pipeline.ProjectionError, match="model must be one of"):
        pipeline.render_projection({"model": "action_button"})
